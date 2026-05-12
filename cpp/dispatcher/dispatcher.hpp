#pragma once
#include "../ipc/spsc_ring.hpp"
#include "../ipc/job.hpp"
#include <array>
#include <string>
#include <string_view>
#include <atomic>
#include <cstdint>
#include <cstring>

namespace se::dispatcher {

using namespace se::ipc;

/// One ring buffer per agent
static constexpr std::size_t RING_CAPACITY = 16;
using AgentRing = SPSCRing<AgentJob, RING_CAPACITY>;

/**
 * Dispatcher — decomposes a user query into AgentJobs and
 * pushes them onto per-agent ring buffers.
 *
 * Thread safety:
 *   - dispatch() is called by the main/producer thread only.
 *   - Each ring is consumed by exactly one agent thread.
 */
class Dispatcher {
public:
    Dispatcher() {
        job_counter_.store(0, std::memory_order_relaxed);
    }

    // Non-copyable
    Dispatcher(const Dispatcher&) = delete;
    Dispatcher& operator=(const Dispatcher&) = delete;

    /**
     * Decompose query into one job per agent role and push
     * onto the corresponding ring buffer.
     *
     * @param query      The user's question
     * @param context    KG-pruned context (shared across agents for now)
     * @param round      Debate round number (0-3)
     * @return           Job ID assigned to this batch, or 0 on failure
     */
    uint64_t dispatch(std::string_view query,
                      std::string_view context,
                      uint8_t round = 0) noexcept
    {
        if (query.empty() || query.size() > MAX_QUERY_BYTES) return 0;

        const uint64_t job_id = job_counter_.fetch_add(1, std::memory_order_relaxed) + 1;

        for (uint8_t i = 0; i < static_cast<uint8_t>(AgentRole::COUNT); ++i) {
            AgentJob job{};
            job.job_id     = job_id;
            job.role       = static_cast<AgentRole>(i);
            job.round      = round;

            // Copy query
            job.query_len  = static_cast<uint16_t>(query.size());
            std::memcpy(job.query, query.data(), query.size());

            // Copy context (truncate if needed)
            const std::size_t ctx_len = std::min(context.size(), MAX_CONTEXT_BYTES);
            job.context_len = static_cast<uint16_t>(ctx_len);
            std::memcpy(job.context, context.data(), ctx_len);

            // Spin-push — in production we'd add a timeout/backpressure
            while (!rings_[i].push(job)) {
                // ring full — yield and retry
                __builtin_ia32_pause();
            }
        }

        return job_id;
    }

    /**
     * Get the ring buffer for a specific agent role.
     * Called by agent threads to obtain their consumer end.
     */
    AgentRing& ring_for(AgentRole role) noexcept {
        return rings_[static_cast<std::size_t>(role)];
    }

    uint64_t jobs_dispatched() const noexcept {
        return job_counter_.load(std::memory_order_relaxed);
    }

private:
    std::array<AgentRing, static_cast<std::size_t>(AgentRole::COUNT)> rings_;
    alignas(64) std::atomic<uint64_t> job_counter_{0};
};

} // namespace se::dispatcher
