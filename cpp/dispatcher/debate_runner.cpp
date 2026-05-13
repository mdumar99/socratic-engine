/**
 * debate_runner.cpp — production debate binary with token streaming
 *
 * Output format (one JSON line per event):
 *   {"event":"start","query":"...","kg_nodes":N}
 *   {"event":"round_start","round":0,"role":"Proposer"}
 *   {"event":"token","round":0,"role":"Proposer","token":"..."}
 *   {"event":"round_end","round":0,"role":"Proposer","chars":N}
 *   ... repeat for rounds 1-3 ...
 *   {"event":"done","elapsed":63.4,"success":true}
 *   {"event":"error","message":"..."}
 */
#include "debate_orchestrator.hpp"
#include "../knowledge_graph/kg_types.hpp"
#include <iostream>
#include <string>
#include <filesystem>
#include <sstream>
#include "ggml.h"

using namespace se::dispatcher;
using namespace se::kg;

static void null_log(ggml_log_level, const char*, void*) {}

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

/// Run one agent round with per-token streaming
static std::string run_round(
    LlamaEngine&    engine,
    llama_context*  ctx,
    const AgentJob& job,
    int             round,
    const char*     role,
    int             max_tokens)
{
    emit("{\"event\":\"round_start\",\"round\":" + std::to_string(round) +
         ",\"role\":\"" + role + "\"}");

    std::string output;
    output.reserve(max_tokens * 4);

    engine.generate_stream(
        ctx,
        build_prompt(job),
        [&](std::string_view piece) -> bool {
            output.append(piece);
            // Emit token event
            std::string tok(piece);
            emit("{\"event\":\"token\",\"round\":" + std::to_string(round) +
                 ",\"role\":\"" + role + "\",\"token\":\"" +
                 json_escape(tok) + "\"}");
            return true;  // continue generating
        },
        max_tokens,
        0.7f
    );

    // Trim artifacts
    for (const auto& art : {"<|end|>", "<|assistant|>", "<|user|>", "<|system|>"}) {
        size_t pos;
        while ((pos = output.find(art)) != std::string::npos)
            output.erase(pos, strlen(art));
    }
    // Trim whitespace
    const auto s = output.find_first_not_of(" \t\n\r");
    const auto e = output.find_last_not_of(" \t\n\r");
    if (s != std::string::npos) output = output.substr(s, e - s + 1);

    emit("{\"event\":\"round_end\",\"round\":" + std::to_string(round) +
         ",\"role\":\"" + role + "\",\"chars\":" +
         std::to_string(output.size()) + "}");

    return output;
}

int main(int argc, char* argv[]) {
    llama_log_set(null_log, nullptr);

    std::string query;
    std::string model_path = std::string("/home/") + getenv("USER") +
                             "/models/phi3-mini-q4.gguf";
    std::string kg_path    = "/tmp/se_runtime_kg";
    int         max_tokens = 400;

    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if      (arg == "--query"  && i+1 < argc) query      = argv[++i];
        else if (arg == "--model"  && i+1 < argc) model_path = argv[++i];
        else if (arg == "--kg"     && i+1 < argc) kg_path    = argv[++i];
        else if (arg == "--tokens" && i+1 < argc) max_tokens = std::stoi(argv[++i]);
    }

    if (query.empty()) std::getline(std::cin, query);
    if (query.empty()) {
        emit_error("No query provided.");
        return 1;
    }

    LlamaEngine* engine = nullptr;
    try {
        engine = new LlamaEngine(model_path, 99);
    } catch (const std::exception& e) {
        emit_error(std::string("Model load failed: ") + e.what());
        return 1;
    }

    bool created_kg = false;
    if (!std::filesystem::exists(kg_path)) {
        created_kg = true;
        KnowledgeGraph kg_init(kg_path);
        kg_init.upsert_node(make_node_helper(1, "knowledge",
            "General knowledge base — add domain nodes via kg_builder"));
    }

    try {
        KnowledgeGraph kg(kg_path);
        auto kg_nodes    = kg.bfs(1, 2, 16);
        std::string base = kg.to_context_string(kg_nodes);

        emit("{\"event\":\"start\",\"query\":\"" + json_escape(query) +
             "\",\"kg_nodes\":" + std::to_string(kg_nodes.size()) + "}");

        // Pre-create 4 contexts
        std::array<llama_context*, 4> contexts{};
        for (int i = 0; i < 4; ++i)
            contexts[i] = engine->make_context(3072, 2);

        auto trunc = [](const std::string& s, size_t n = 350) {
            return s.size() <= n ? s : s.substr(0, n) + "...";
        };

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

        auto t_start = std::chrono::steady_clock::now();
        std::array<std::string, 4> outputs;

        // Round 0: Proposer
        outputs[0] = run_round(*engine, contexts[0],
            make_job(query, base, AgentRole::Proposer, 0),
            0, "Proposer", max_tokens);

        // Round 1: Critic
        outputs[1] = run_round(*engine, contexts[1],
            make_job(query,
                base + "\n[Proposer]\n" + outputs[0],
                AgentRole::Critic, 1),
            1, "Critic", max_tokens);

        // Round 2: Retriever
        outputs[2] = run_round(*engine, contexts[2],
            make_job(query,
                base + "\n[Proposer]\n" + outputs[0]
                     + "\n[Critic]\n"   + outputs[1],
                AgentRole::Retriever, 2),
            2, "Retriever", max_tokens);

        // Round 3: Synthesizer
        outputs[3] = run_round(*engine, contexts[3],
            make_job(query,
                base + "\n[Proposer]\n"  + trunc(outputs[0])
                     + "\n[Critic]\n"    + trunc(outputs[1])
                     + "\n[Retriever]\n" + trunc(outputs[2]),
                AgentRole::Synthesizer, 3),
            3, "Synthesizer", max_tokens);

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
