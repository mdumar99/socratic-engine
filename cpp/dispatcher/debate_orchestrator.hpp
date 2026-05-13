#pragma once
#include "dispatcher.hpp"
#include "../agents/agent_runner.hpp"
#include "../agents/llama_engine.hpp"
#include "../knowledge_graph/knowledge_graph.hpp"
#include <string>
#include <vector>
#include <mutex>
#include <condition_variable>
#include <chrono>
#include <iostream>
#include <algorithm>
#include <array>

namespace se::dispatcher {

using namespace se::agents;
using namespace se::kg;

struct DebateResult {
    std::string query;
    std::string final_answer;
    std::string proposer_output;
    std::string critic_output;
    std::string retriever_output;
    double      elapsed_seconds;
    bool        success;
};

/**
 * DebateOrchestrator — 4-round Socratic debate.
 * Pre-creates one llama_context per agent at construction time.
 * Contexts are reused across rounds — no per-round VRAM allocation.
 */
class DebateOrchestrator {
public:
    DebateOrchestrator(LlamaEngine&    engine,
                       KnowledgeGraph& kg,
                       int             max_tokens     = 200,
                       float           temp           = 0.7f)
        : engine_(engine)
        , kg_(kg)
        , max_tokens_(max_tokens)
        , temp_(temp)
    {
        // Pre-create one context per agent — done ONCE at startup
        std::cout << "  Pre-creating 4 agent contexts...\n";
        for (int i = 0; i < 4; ++i) {
            contexts_[i] = engine_.make_context(2048, 2);
            std::cout << "  Context " << i << " ready\n";
        }
        std::cout << "  All contexts ready.\n\n";
    }

    ~DebateOrchestrator() {
        for (int i = 0; i < 4; ++i) {
            if (contexts_[i]) {
                llama_free(contexts_[i]);
                contexts_[i] = nullptr;
            }
        }
    }

    DebateOrchestrator(const DebateOrchestrator&) = delete;
    DebateOrchestrator& operator=(const DebateOrchestrator&) = delete;

    DebateResult run(const std::string& query, uint64_t kg_seed_node = 1) {
        DebateResult result;
        result.query   = query;
        result.success = false;

        auto t_start = std::chrono::steady_clock::now();

        // KG context via BFS
        auto kg_nodes    = kg_.bfs(kg_seed_node, 2, 16);
        std::string base = kg_.to_context_string(kg_nodes);
        std::cout << "  KG: " << kg_nodes.size() << " nodes\n\n";

        // ── Round 0: Proposer ──
        std::cout << "[Round 0] Proposer...\n";
        AgentJob job0 = make_job(query, base, AgentRole::Proposer, 0);
        result.proposer_output = engine_.generate(
            contexts_[0], build_prompt(job0), max_tokens_, temp_);
        std::cout << "  done (" << result.proposer_output.size() << " chars)\n\n";

        // ── Round 1: Critic ──
        std::cout << "[Round 1] Critic...\n";
        std::string ctx1 = base
            + "\n[Proposer]\n" + result.proposer_output;
        AgentJob job1 = make_job(query, ctx1, AgentRole::Critic, 1);
        result.critic_output = engine_.generate(
            contexts_[1], build_prompt(job1), max_tokens_, temp_);
        std::cout << "  done (" << result.critic_output.size() << " chars)\n\n";

        // ── Round 2: Retriever ──
        std::cout << "[Round 2] Retriever...\n";
        std::string ctx2 = base
            + "\n[Proposer]\n"  + result.proposer_output
            + "\n[Critic]\n"    + result.critic_output;
        AgentJob job2 = make_job(query, ctx2, AgentRole::Retriever, 2);
        result.retriever_output = engine_.generate(
            contexts_[2], build_prompt(job2), max_tokens_, temp_);
        std::cout << "  done (" << result.retriever_output.size() << " chars)\n\n";

        // ── Round 3: Synthesizer ──
        std::cout << "[Round 3] Synthesizer...\n";
        std::string ctx3 = base
            + "\n[Proposer]\n"   + trunc(result.proposer_output, 300)
            + "\n[Critic]\n"     + trunc(result.critic_output, 300)
            + "\n[Retriever]\n"  + trunc(result.retriever_output, 300);
        AgentJob job3 = make_job(query, ctx3, AgentRole::Synthesizer, 3);
        result.final_answer = engine_.generate(
            contexts_[3], build_prompt(job3), max_tokens_, temp_);
        std::cout << "  done (" << result.final_answer.size() << " chars)\n\n";

        result.elapsed_seconds = std::chrono::duration<double>(
            std::chrono::steady_clock::now() - t_start).count();
        result.success = !result.final_answer.empty();
        return result;
    }

private:

    /// Truncate a string to max_chars, keeping whole words where possible
    static std::string trunc(const std::string& s, size_t max_chars = 400) {
        if (s.size() <= max_chars) return s;
        return s.substr(0, max_chars) + "...";
    }

    /// Build an AgentJob struct directly (no ring buffer needed here)
    static AgentJob make_job(const std::string& query,
                             const std::string& context,
                             AgentRole          role,
                             uint8_t            round)
    {
        AgentJob job{};
        job.role  = role;
        job.round = round;

        job.query_len = static_cast<uint16_t>(
            std::min(query.size(), se::ipc::MAX_QUERY_BYTES));
        std::memcpy(job.query, query.data(), job.query_len);

        job.context_len = static_cast<uint16_t>(
            std::min(context.size(), se::ipc::MAX_CONTEXT_BYTES));
        std::memcpy(job.context, context.data(), job.context_len);

        return job;
    }

    LlamaEngine&    engine_;
    KnowledgeGraph& kg_;
    int             max_tokens_;
    float           temp_;
    std::array<llama_context*, 4> contexts_{};
};

} // namespace se::dispatcher
