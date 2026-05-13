#pragma once
#include "llama_engine.hpp"
#include "../ipc/spsc_ring.hpp"
#include "../ipc/job.hpp"
#include <string>
#include <string_view>
#include <functional>
#include <atomic>
#include <thread>
#include <chrono>
#include <sstream>

namespace se::agents {

using namespace se::ipc;

// Ring buffer type for agent communication
using AgentRing = SPSCRing<AgentJob, 16>;

/// Result produced by an agent after processing one job
struct AgentResult {
    uint64_t    job_id;
    AgentRole   role;
    uint8_t     round;
    std::string output;
    bool        success;
};

/// Callback invoked when an agent finishes a job
using ResultCallback = std::function<void(AgentResult)>;

/// Role-specific system prompts
inline std::string build_system_prompt(AgentRole role, uint8_t /*round*/) {
    switch (role) {
        case AgentRole::Proposer:
            return "You are the Proposer agent in a Socratic reasoning system. "
                   "Generate a clear, well-structured hypothesis to the question. "
                   "State your reasoning explicitly. Keep your response under 150 words.";
        case AgentRole::Critic:
            return "You are the Critic agent in a Socratic reasoning system. "
                   "Identify flaws, gaps, or contradictions in the proposed answer. "
                   "Be specific. If the answer is sound, say so briefly. Under 150 words.";
        case AgentRole::Retriever:
            return "You are the Retriever agent in a Socratic reasoning system. "
                   "Extract the most relevant facts from the provided context. "
                   "List key facts that support or refute the hypothesis. Under 150 words.";
        case AgentRole::Synthesizer:
            return "You are the Synthesizer agent in a Socratic reasoning system. "
                   "Produce a final refined answer integrating hypothesis, critique, "
                   "and evidence. Resolve contradictions. Be definitive. Under 200 words.";
        default:
            return "You are a reasoning agent. Answer the question carefully.";
    }
}

/// Build full Phi-3 chat format prompt
inline std::string build_prompt(const AgentJob& job) {
    std::ostringstream oss;
    oss << "<|system|>\n"
        << build_system_prompt(job.role, job.round)
        << "\n<|end|>\n"
        << "<|user|>\n";

    if (job.context_len > 0) {
        oss << "Context from knowledge graph:\n"
            << std::string(job.context, job.context_len)
            << "\n\n";
    }

    oss << "Question: "
        << std::string(job.query, job.query_len)
        << "\n<|end|>\n"
        << "<|assistant|>\n";

    return oss.str();
}

/**
 * AgentRunner — one per agent role.
 * Owns its own llama_context. Shares the LlamaEngine (model weights).
 * Runs on its own thread, consuming jobs from its ring buffer.
 */
class AgentRunner {
public:
    AgentRunner(AgentRing&          ring,
                const LlamaEngine&  engine,
                ResultCallback      result_cb,
                int                 max_tokens = 256,
                float               temp       = 0.7f)
        : ring_(ring)
        , engine_(engine)
        , result_cb_(std::move(result_cb))
        , max_tokens_(max_tokens)
        , temp_(temp)
        , running_(false)
        , ctx_(nullptr)
    {}

    ~AgentRunner() { stop(); }

    AgentRunner(const AgentRunner&) = delete;
    AgentRunner& operator=(const AgentRunner&) = delete;

    void start() {
        running_.store(true, std::memory_order_release);
        thread_ = std::thread(&AgentRunner::run_loop, this);
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
        if (ctx_) {
            llama_free(ctx_);
            ctx_ = nullptr;
        }
    }

    bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

private:
    void run_loop() {
        // Each agent creates its own context on its own thread
        ctx_ = engine_.make_context();

        while (running_.load(std::memory_order_acquire)) {
            auto job = ring_.pop();
            if (!job) {
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            const std::string prompt = build_prompt(*job);
            const std::string output = engine_.generate(ctx_, prompt, max_tokens_, temp_);

            AgentResult result{
                job->job_id,
                job->role,
                job->round,
                output,
                !output.empty()
            };

            if (result_cb_) result_cb_(std::move(result));
        }
    }

    AgentRing&          ring_;
    const LlamaEngine&  engine_;
    ResultCallback      result_cb_;
    int                 max_tokens_;
    float               temp_;
    std::atomic<bool>   running_;
    std::thread         thread_;
    llama_context*      ctx_;
};

} // namespace se::agents
