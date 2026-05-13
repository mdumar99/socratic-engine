#pragma once
#include <atomic>
#include <array>
#include <optional>
#include <cstddef>
#include <cassert>

namespace se::ipc {

/**
 * Lock-free Single-Producer Single-Consumer ring buffer.
 *
 * - One thread calls push(), another calls pop() — never both sides
 *   from multiple threads simultaneously.
 * - Cache-line aligned head/tail to prevent false sharing.
 * - Zero heap allocation, fixed capacity at compile time.
 *
 * @tparam T    Element type (must be trivially copyable)
 * @tparam N    Capacity (must be power of two)
 */
template<typename T, std::size_t N>
class SPSCRing {
    static_assert((N & (N - 1)) == 0, "N must be a power of two");
    static_assert(std::is_trivially_copyable_v<T>, "T must be trivially copyable");

    static constexpr std::size_t MASK = N - 1;

    // Separate cache lines to prevent false sharing between producer and consumer
    alignas(64) std::atomic<std::size_t> head_{0};  // written by producer
    alignas(64) std::atomic<std::size_t> tail_{0};  // written by consumer
    alignas(64) std::array<T, N> slots_{};

public:
    SPSCRing() = default;

    // Non-copyable, non-movable
    SPSCRing(const SPSCRing&) = delete;
    SPSCRing& operator=(const SPSCRing&) = delete;

    /**
     * Push an item. Called only by the producer thread.
     * @return true if pushed, false if ring is full
     */
    bool push(const T& item) noexcept {
        const std::size_t h = head_.load(std::memory_order_relaxed);
        const std::size_t next = (h + 1) & MASK;

        // Full if next write position == tail
        if (next == tail_.load(std::memory_order_acquire)) {
            return false;
        }

        slots_[h] = item;
        head_.store(next, std::memory_order_release);
        return true;
    }

    /**
     * Pop an item. Called only by the consumer thread.
     * @return item if available, std::nullopt if empty
     */
    std::optional<T> pop() noexcept {
        const std::size_t t = tail_.load(std::memory_order_relaxed);

        // Empty if tail == head
        if (t == head_.load(std::memory_order_acquire)) {
            return std::nullopt;
        }

        T item = slots_[t];
        tail_.store((t + 1) & MASK, std::memory_order_release);
        return item;
    }

    /**
     * Non-blocking peek — check if data is available without consuming.
     */
    bool empty() const noexcept {
        return tail_.load(std::memory_order_acquire) ==
               head_.load(std::memory_order_acquire);
    }

    /**
     * Approximate size — not exact under concurrent access.
     */
    std::size_t size_approx() const noexcept {
        const std::size_t h = head_.load(std::memory_order_acquire);
        const std::size_t t = tail_.load(std::memory_order_acquire);
        return (h - t) & MASK;
    }

    static constexpr std::size_t capacity() noexcept { return N; }
};

} // namespace se::ipc
