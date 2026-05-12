#include "../../cpp/ipc/spsc_ring.hpp"
#include "../../cpp/ipc/job.hpp"
#include <cassert>
#include <thread>
#include <iostream>
#include <atomic>

using namespace se::ipc;

void test_basic_push_pop() {
    SPSCRing<int, 8> ring;

    assert(ring.empty());
    assert(ring.push(42));
    assert(!ring.empty());

    auto val = ring.pop();
    assert(val.has_value());
    assert(val.value() == 42);
    assert(ring.empty());

    std::cout << "✓ test_basic_push_pop\n";
}

void test_full_ring() {
    SPSCRing<int, 4> ring;  // capacity 4, but only 3 usable (ring buffer property)

    assert(ring.push(1));
    assert(ring.push(2));
    assert(ring.push(3));
    assert(!ring.push(4));  // should be full

    std::cout << "✓ test_full_ring\n";
}

void test_fifo_order() {
    SPSCRing<int, 16> ring;

    for (int i = 0; i < 10; i++) ring.push(i);
    for (int i = 0; i < 10; i++) {
        auto val = ring.pop();
        assert(val.has_value());
        assert(val.value() == i);
    }

    std::cout << "✓ test_fifo_order\n";
}

void test_concurrent_producer_consumer() {
    SPSCRing<int, 1024> ring;
    std::atomic<int> sum_produced{0};
    std::atomic<int> sum_consumed{0};
    const int N = 10000;

    std::thread producer([&]() {
        for (int i = 0; i < N; i++) {
            while (!ring.push(i)) {}  // spin until space
            sum_produced.fetch_add(i, std::memory_order_relaxed);
        }
    });

    std::thread consumer([&]() {
        int count = 0;
        while (count < N) {
            auto val = ring.pop();
            if (val) {
                sum_consumed.fetch_add(val.value(), std::memory_order_relaxed);
                count++;
            }
        }
    });

    producer.join();
    consumer.join();

    assert(sum_produced.load() == sum_consumed.load());
    std::cout << "✓ test_concurrent_producer_consumer (" << N << " items)\n";
}

void test_agent_job_fits_ring() {
    SPSCRing<AgentJob, 8> ring;

    AgentJob job{};
    job.job_id = 1;
    job.role = AgentRole::Proposer;
    job.round = 0;

    const char* q = "What causes lightning?";
    job.query_len = static_cast<uint16_t>(__builtin_strlen(q));
    __builtin_memcpy(job.query, q, job.query_len);

    assert(ring.push(job));
    auto out = ring.pop();
    assert(out.has_value());
    assert(out->job_id == 1);
    assert(out->role == AgentRole::Proposer);
    assert(out->query_len == job.query_len);

    std::cout << "✓ test_agent_job_fits_ring\n";
}

int main() {
    std::cout << "Running SPSC ring buffer tests...\n";
    test_basic_push_pop();
    test_full_ring();
    test_fifo_order();
    test_concurrent_producer_consumer();
    test_agent_job_fits_ring();
    std::cout << "\nAll tests passed.\n";
    return 0;
}
