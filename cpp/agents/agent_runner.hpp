#pragma once
#include "../ipc/spsc_ring.hpp"
#include "../ipc/job.hpp"
#include <string>
#include <string_view>
#include <functional>
#include <atomic>
#include <thread>
#include <sstream>
#include <cstdio>
#include <array>
#include <stdexcept>

namespace se::agents {

using namespace se::ipc;

// Ring buffer type for agent communication
using AgentRing = SPSCRing<AgentJob, 16>;

/// Result produced by an agent after processing one job
struct AgentResult {
    uint64_t    job_id;
    AgentRole   role;
    uint8_t     round;
    std::string output;     // raw text from the model
    bool        success;
};

/// Callback invoked when an agent finishes a job
using ResultCallback = std::function<void(AgentResult)>;

/**
 * Builds the role-specific system prompt for each agent.
 * This is the core of the Socratic debate — each agent has
 * a distinct persona and instruction set.
 */
inline std::string build_system_prompt(AgentRole role, uint8_t round) {
    switch (role) {
        case AgentRole::Proposer:
            return "You are the Proposer agent in a Socratic reasoning system. "
                   "Your job is to generate a clear, well-structured hypothesis or answer "
                   "to the question. Be confident but not overreaching. "
                   "State your reasoning explicitly. Keep your response under 150 words.";

        case AgentRole::Critic:
            return "You are the Critic agent in a Socratic reasoning system. "
                   "Your job is to identify flaws, gaps, contradictions, or unsupported "
                   "claims in the proposed answer. Be specific about what is wrong and why. "
                   "If the answer is sound, say so briefly. Keep your response under 150 words.";

        case AgentRole::Retriever:
            return "You are the Retriever agent in a Socratic reasoning system. "
                   "Your job is to extract and highlight the most relevant facts from the "
                   "provided context. Focus on evidence that either supports or refutes "
                   "the hypothesis. List key facts clearly. Keep your response under 150 words.";

        case AgentRole::Synthesizer:
            return "You are the Synthesizer agent in a Socratic reasoning system. "
                   "Your job is to produce a final, refined answer by integrating the "
                   "hypothesis, critique, and retrieved evidence. Resolve contradictions. "
                   "Be definitive and clear. Keep your response under 200 words.";

        default:
            return "You are a reasoning agent. Answer the question carefully.";
    }
}

/**
 * Builds the full prompt string for a given job.
 * Format: [SYSTEM]\n{system}\n[CONTEXT]\n{kg_context}\n[QUESTION]\n{query}
 */
inline std::string build_prompt(const AgentJob& job) {
    std::ostringstream oss;

    oss << "<|system|>\n"
        << build_system_prompt(job.role, job.round)
        << "\n<|end|>\n";

    if (job.context_len > 0) {
        oss << "<|user|>\n"
            << "Context from knowledge graph:\n"
            << std::string(job.context, job.context_len)
            << "\n\nQuestion: "
            << std::string(job.query, job.query_len)
            << "\n<|end|>\n";
    } else {
        oss << "<|user|>\n"
            << "Question: "
            << std::string(job.query, job.query_len)
            << "\n<|end|>\n";
    }

    oss << "<|assistant|>\n";
    return oss.str();
}

/**
 * Runs llama-cli as a subprocess for one inference call.
 * Returns the model output as a string, or empty on failure.
 *
 * @param model_path   Path to the GGUF model file
 * @param prompt       Full prompt string
 * @param n_gpu_layers Number of layers to offload to GPU
 * @param max_tokens   Maximum tokens to generate
 */
inline std::string run_llama_inference(
    const std::string& model_path,
    const std::string& prompt,
    int n_gpu_layers = 99,
    int max_tokens   = 256)
{
    // Write prompt to a temp file to avoid shell escaping issues
    const std::string prompt_file = "/tmp/se_prompt_" +
        std::to_string(std::hash<std::thread::id>{}(std::this_thread::get_id())) + ".txt";

    {
        FILE* f = fopen(prompt_file.c_str(), "w");
        if (!f) return "";
        fwrite(prompt.data(), 1, prompt.size(), f);
        fclose(f);
    }

    // Build the llama-cli command
    std::ostringstream cmd;
    cmd << "/home/" << getenv("USER") << "/llama.cpp/build/bin/llama-cli"
        << " -m " << model_path
        << " -ngl " << n_gpu_layers
        << " -n " << max_tokens
        << " --temp 0.7"
        << " --repeat-penalty 1.1"
        << " --no-display-prompt"
        << " -f " << prompt_file
        << " 2>/dev/null";

    // Run and capture output
    std::string output;
    FILE* pipe = popen(cmd.str().c_str(), "r");
    if (!pipe) {
        remove(prompt_file.c_str());
        return "";
    }

    std::array<char, 256> buf;
    while (fgets(buf.data(), buf.size(), pipe)) {
        output += buf.data();
    }
    pclose(pipe);
    remove(prompt_file.c_str());

    // Trim trailing whitespace
    while (!output.empty() && std::isspace(output.back())) {
        output.pop_back();
    }

    return output;
}

/**
 * AgentRunner — consumes jobs from a ring buffer and runs inference.
 *
 * Runs on its own thread. Calls result_cb on the calling thread
 * context via the ring (caller must drain results).
 *
 * Usage:
 *   AgentRunner runner(ring, model_path, result_callback);
 *   runner.start();
 *   // ... dispatch jobs ...
 *   runner.stop();
 */
class AgentRunner {
public:
    AgentRunner(AgentRing&         ring,
                std::string        model_path,
                ResultCallback     result_cb)
        : ring_(ring)
        , model_path_(std::move(model_path))
        , result_cb_(std::move(result_cb))
        , running_(false)
    {}

    ~AgentRunner() { stop(); }

    void start() {
        running_.store(true, std::memory_order_release);
        thread_ = std::thread(&AgentRunner::run_loop, this);
    }

    void stop() {
        running_.store(false, std::memory_order_release);
        if (thread_.joinable()) thread_.join();
    }

    bool is_running() const {
        return running_.load(std::memory_order_acquire);
    }

private:
    void run_loop() {
        while (running_.load(std::memory_order_acquire)) {
            auto job = ring_.pop();
            if (!job) {
                // Nothing in ring — yield briefly
                std::this_thread::sleep_for(std::chrono::milliseconds(1));
                continue;
            }

            const std::string prompt = build_prompt(*job);
            const std::string output = run_llama_inference(model_path_, prompt);

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
    std::string         model_path_;
    ResultCallback      result_cb_;
    std::atomic<bool>   running_;
    std::thread         thread_;
};

} // namespace se::agents
