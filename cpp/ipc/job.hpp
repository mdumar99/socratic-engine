#pragma once
#include <array>
#include <cstdint>

namespace se::ipc {

/// Agent roles in the Socratic debate
enum class AgentRole : uint8_t {
    Proposer    = 0,
    Critic      = 1,
    Retriever   = 2,
    Synthesizer = 3,
    COUNT       = 4
};

/// Maximum context size per agent slot (in bytes)
static constexpr std::size_t MAX_CONTEXT_BYTES = 4096;

/// Maximum query size (in bytes)
static constexpr std::size_t MAX_QUERY_BYTES = 512;

/**
 * A job sent from the dispatcher to an agent via the ring buffer.
 * Trivially copyable — safe to memcpy through shared memory.
 */
struct alignas(64) AgentJob {
    uint64_t    job_id;                          // unique job identifier
    AgentRole   role;                            // which agent receives this
    uint8_t     round;                           // debate round (0-3)
    uint16_t    context_len;                     // valid bytes in context[]
    uint16_t    query_len;                       // valid bytes in query[]
    char        query[MAX_QUERY_BYTES];          // the user's question
    char        context[MAX_CONTEXT_BYTES];      // KG-pruned context for this agent
};

static_assert(std::is_trivially_copyable_v<AgentJob>,
    "AgentJob must be trivially copyable to pass through ring buffer");

} // namespace se::ipc
