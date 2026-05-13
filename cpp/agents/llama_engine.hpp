#pragma once
#include <string>
#include <vector>
#include <stdexcept>
#include <cstring>

#include "llama.h"

namespace se::agents {

/**
 * LlamaEngine — wraps a single loaded llama_model.
 * One engine is shared across all agents.
 * Each agent gets its own llama_context via make_context().
 */
class LlamaEngine {
public:
    explicit LlamaEngine(const std::string& model_path, int n_gpu_layers = 99) {
        llama_model_params mparams = llama_model_default_params();
        mparams.n_gpu_layers = n_gpu_layers;

        model_ = llama_model_load_from_file(model_path.c_str(), mparams);
        if (!model_) {
            throw std::runtime_error("LlamaEngine: failed to load model: " + model_path);
        }

        vocab_ = llama_model_get_vocab(model_);
    }

    ~LlamaEngine() {
        if (model_) llama_model_free(model_);
    }

    // Non-copyable, non-movable
    LlamaEngine(const LlamaEngine&) = delete;
    LlamaEngine& operator=(const LlamaEngine&) = delete;

    /**
     * Create a lightweight context for one agent thread.
     * Each agent calls this once and owns its context.
     * Caller must call llama_free() on the returned context.
     */
    llama_context* make_context(int n_ctx = 4096, int n_threads = 2) const {
        llama_context_params cparams = llama_context_default_params();
        cparams.n_ctx      = n_ctx;
        cparams.n_threads  = n_threads;
        cparams.n_batch    = 2048;

        llama_context* ctx = llama_init_from_model(model_, cparams);
        if (!ctx) {
            throw std::runtime_error("LlamaEngine: failed to create context");
        }
        return ctx;
    }

    /**
     * Run inference on a given context with a prompt string.
     * Returns the generated text.
     *
     * @param ctx        Agent's own llama_context (not shared)
     * @param prompt     Full prompt string
     * @param max_tokens Maximum tokens to generate
     * @param temp       Sampling temperature
     */
    std::string generate(llama_context* ctx,
                         const std::string& prompt,
                         int max_tokens = 256,
                         float temp     = 0.7f) const
    {
        // Tokenize prompt
        const int n_prompt_tokens = -llama_tokenize(
            vocab_, prompt.c_str(), static_cast<int32_t>(prompt.size()),
            nullptr, 0, true, true);

        std::vector<llama_token> tokens(n_prompt_tokens);
        if (llama_tokenize(vocab_, prompt.c_str(),
                           static_cast<int32_t>(prompt.size()),
                           tokens.data(),
                           static_cast<int32_t>(tokens.size()),
                           true, true) < 0)
        {
            return "";
        }

        // Clear KV cache for fresh inference
        

        // Decode prompt tokens
        llama_batch batch = llama_batch_get_one(tokens.data(),
                                                static_cast<int32_t>(tokens.size()));
        if (llama_decode(ctx, batch) != 0) return "";

        // Set up sampler chain
        llama_sampler_chain_params sparams = llama_sampler_chain_default_params();
        llama_sampler* sampler = llama_sampler_chain_init(sparams);
        llama_sampler_chain_add(sampler, llama_sampler_init_temp(temp));
        llama_sampler_chain_add(sampler, llama_sampler_init_dist(42));

        const llama_token eos = llama_vocab_eos(vocab_);

        // Generate tokens
        std::string output;
        output.reserve(max_tokens * 4);

        for (int i = 0; i < max_tokens; ++i) {
            const llama_token token = llama_sampler_sample(sampler, ctx, -1);

            if (token == eos) break;

            // Convert token to text
            char buf[256];
            const int n = llama_token_to_piece(vocab_, token, buf, sizeof(buf), 0, true);
            if (n < 0) break;
            output.append(buf, n);

            // Decode next token
            llama_token next_token = token;
        llama_batch next = llama_batch_get_one(&next_token, 1);
            if (llama_decode(ctx, next) != 0) break;
        }

        llama_sampler_free(sampler);

        // Trim leading/trailing whitespace
        const auto start = output.find_first_not_of(" \t\n\r");
        const auto end   = output.find_last_not_of(" \t\n\r");
        if (start == std::string::npos) return "";
        return output.substr(start, end - start + 1);
    }

    const llama_model* model() const { return model_; }
    const llama_vocab* vocab() const { return vocab_; }

private:
    llama_model* model_ = nullptr;
    const llama_vocab* vocab_ = nullptr;
};

} // namespace se::agents
