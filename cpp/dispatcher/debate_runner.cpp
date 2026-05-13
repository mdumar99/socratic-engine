/**
 * debate_runner.cpp — production debate binary
 *
 * Reads query from stdin or --query argument.
 * Outputs structured JSON to stdout, one object per round.
 * Stderr gets llama.cpp logs (silenced by default).
 *
 * Output format (one JSON line per event):
 *   {"event":"start","query":"...","kg_nodes":N}
 *   {"event":"round","round":0,"role":"Proposer","output":"...","chars":N}
 *   {"event":"round","round":1,"role":"Critic","output":"...","chars":N}
 *   {"event":"round","round":2,"role":"Retriever","output":"...","chars":N}
 *   {"event":"round","round":3,"role":"Synthesizer","output":"...","chars":N}
 *   {"event":"done","elapsed":63.4,"success":true}
 *   {"event":"error","message":"..."}
 */
#include "debate_orchestrator.hpp"
#include "../knowledge_graph/kg_types.hpp"
#include <iostream>
#include <fstream>
#include <sstream>
#include <string>
#include <filesystem>
#include "ggml.h"

using namespace se::dispatcher;
using namespace se::kg;

static void null_log(ggml_log_level, const char*, void*) {}

/// Escape a string for JSON output
static std::string json_escape(const std::string& s) {
    std::ostringstream oss;
    for (unsigned char c : s) {
        switch (c) {
            case '"':  oss << "\\\""; break;
            case '\\': oss << "\\\\"; break;
            case '\n': oss << "\\n";  break;
            case '\r': oss << "\\r";  break;
            case '\t': oss << "\\t";  break;
            default:
                if (c < 0x20) oss << "\\u00" << std::hex << (int)c;
                else oss << c;
        }
    }
    return oss.str();
}

static void emit(const std::string& json) {
    std::cout << json << "\n" << std::flush;
}

static void emit_error(const std::string& msg) {
    emit("{\"event\":\"error\",\"message\":\"" + json_escape(msg) + "\"}");
}

int main(int argc, char* argv[]) {
    llama_log_set(null_log, nullptr);

    // Parse args
    std::string query;
    std::string model_path = std::string("/home/") + getenv("USER") + "/models/phi3-mini-q4.gguf";
    std::string kg_path    = "/tmp/se_runtime_kg";
    int         max_tokens = 150;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--query" && i + 1 < argc)        query      = argv[++i];
        else if (arg == "--model" && i + 1 < argc)   model_path = argv[++i];
        else if (arg == "--kg" && i + 1 < argc)      kg_path    = argv[++i];
        else if (arg == "--tokens" && i + 1 < argc)  max_tokens = std::stoi(argv[++i]);
    }

    // Read query from stdin if not provided
    if (query.empty()) {
        std::getline(std::cin, query);
    }

    if (query.empty()) {
        emit_error("No query provided. Use --query or pipe via stdin.");
        return 1;
    }

    // Load model
    LlamaEngine* engine = nullptr;
    try {
        engine = new LlamaEngine(model_path, 99);
    } catch (const std::exception& e) {
        emit_error(std::string("Model load failed: ") + e.what());
        return 1;
    }

    // Build default KG if path doesn't exist
    bool created_kg = false;
    if (!std::filesystem::exists(kg_path)) {
        created_kg = true;
        KnowledgeGraph kg_init(kg_path);
        kg_init.upsert_node(make_node_helper(1, "knowledge",
            "General knowledge base — add domain-specific nodes via kg_builder"));
    }

    // Run debate with JSON output per round
    try {
        KnowledgeGraph kg(kg_path);

        // Custom orchestrator that emits JSON per round
        auto kg_nodes = kg.bfs(1, 2, 16);
        std::string base_context = kg.to_context_string(kg_nodes);

        emit("{\"event\":\"start\",\"query\":\"" + json_escape(query) +
             "\",\"kg_nodes\":" + std::to_string(kg_nodes.size()) + "}");

        // Pre-create contexts
        std::array<llama_context*, 4> contexts{};
        for (int i = 0; i < 4; ++i) {
            contexts[i] = engine->make_context(2048, 2);
        }

        
        std::array<std::string, 4> outputs;

        auto make_job = [&](const std::string& q, const std::string& ctx,
                            AgentRole role, uint8_t round) {
            AgentJob job{};
            job.role  = role;
            job.round = round;
            job.query_len = static_cast<uint16_t>(
                std::min(q.size(), se::ipc::MAX_QUERY_BYTES));
            std::memcpy(job.query, q.data(), job.query_len);
            job.context_len = static_cast<uint16_t>(
                std::min(ctx.size(), se::ipc::MAX_CONTEXT_BYTES));
            std::memcpy(job.context, ctx.data(), job.context_len);
            return job;
        };

        auto trunc = [](const std::string& s, size_t n = 350) {
            return s.size() <= n ? s : s.substr(0, n) + "...";
        };

        auto t_start = std::chrono::steady_clock::now();

        // Round 0: Proposer
        {
            auto job = make_job(query, base_context, AgentRole::Proposer, 0);
            outputs[0] = engine->generate(contexts[0], build_prompt(job), max_tokens);
            emit("{\"event\":\"round\",\"round\":0,\"role\":\"Proposer\","
                 "\"output\":\"" + json_escape(outputs[0]) + "\","
                 "\"chars\":" + std::to_string(outputs[0].size()) + "}");
        }

        // Round 1: Critic
        {
            std::string ctx = base_context + "\n[Proposer]\n" + outputs[0];
            auto job = make_job(query, ctx, AgentRole::Critic, 1);
            outputs[1] = engine->generate(contexts[1], build_prompt(job), max_tokens);
            emit("{\"event\":\"round\",\"round\":1,\"role\":\"Critic\","
                 "\"output\":\"" + json_escape(outputs[1]) + "\","
                 "\"chars\":" + std::to_string(outputs[1].size()) + "}");
        }

        // Round 2: Retriever
        {
            std::string ctx = base_context
                + "\n[Proposer]\n" + outputs[0]
                + "\n[Critic]\n"   + outputs[1];
            auto job = make_job(query, ctx, AgentRole::Retriever, 2);
            outputs[2] = engine->generate(contexts[2], build_prompt(job), max_tokens);
            emit("{\"event\":\"round\",\"round\":2,\"role\":\"Retriever\","
                 "\"output\":\"" + json_escape(outputs[2]) + "\","
                 "\"chars\":" + std::to_string(outputs[2].size()) + "}");
        }

        // Round 3: Synthesizer
        {
            std::string ctx = base_context
                + "\n[Proposer]\n"  + trunc(outputs[0])
                + "\n[Critic]\n"    + trunc(outputs[1])
                + "\n[Retriever]\n" + trunc(outputs[2]);
            auto job = make_job(query, ctx, AgentRole::Synthesizer, 3);
            outputs[3] = engine->generate(contexts[3], build_prompt(job), max_tokens);
            emit("{\"event\":\"round\",\"round\":3,\"role\":\"Synthesizer\","
                 "\"output\":\"" + json_escape(outputs[3]) + "\","
                 "\"chars\":" + std::to_string(outputs[3].size()) + "}");
        }

        double elapsed = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();

        emit("{\"event\":\"done\",\"elapsed\":" +
             std::to_string(elapsed) + ",\"success\":true}");

        for (int i = 0; i < 4; ++i) llama_free(contexts[i]);
        if (created_kg) std::filesystem::remove_all(kg_path);

    } catch (const std::exception& e) {
        emit_error(e.what());
        delete engine;
        return 1;
    }

    delete engine;
    return 0;
}
