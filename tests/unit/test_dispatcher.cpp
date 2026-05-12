#include "../../cpp/dispatcher/dispatcher.hpp"
#include <cassert>
#include <iostream>
#include <thread>
#include <vector>
#include <atomic>

using namespace se::dispatcher;
using namespace se::ipc;

void test_dispatch_creates_four_jobs() {
    Dispatcher d;

    uint64_t id = d.dispatch("What causes lightning?", "Context: thunderstorm clouds...");
    assert(id == 1);

    // Each agent ring should have exactly one job
    for (uint8_t i = 0; i < static_cast<uint8_t>(AgentRole::COUNT); ++i) {
        auto job = d.ring_for(static_cast<AgentRole>(i)).pop();
        assert(job.has_value());
        assert(job->job_id == 1);
        assert(job->role == static_cast<AgentRole>(i));
        assert(job->round == 0);
        assert(std::string_view(job->query, job->query_len) == "What causes lightning?");
    }

    std::cout << "✓ test_dispatch_creates_four_jobs\n";
}

void test_job_ids_increment() {
    Dispatcher d;

    uint64_t id1 = d.dispatch("Query one", "ctx");
    uint64_t id2 = d.dispatch("Query two", "ctx");
    uint64_t id3 = d.dispatch("Query three", "ctx");

    assert(id1 == 1);
    assert(id2 == 2);
    assert(id3 == 3);
    assert(d.jobs_dispatched() == 3);

    std::cout << "✓ test_job_ids_increment\n";
}

void test_round_field_preserved() {
    Dispatcher d;

    d.dispatch("Test query", "ctx", 2);  // round 2

    for (uint8_t i = 0; i < static_cast<uint8_t>(AgentRole::COUNT); ++i) {
        auto job = d.ring_for(static_cast<AgentRole>(i)).pop();
        assert(job.has_value());
        assert(job->round == 2);
    }

    std::cout << "✓ test_round_field_preserved\n";
}

void test_context_truncation() {
    Dispatcher d;

    // Context larger than MAX_CONTEXT_BYTES should be truncated, not crash
    std::string big_context(MAX_CONTEXT_BYTES + 100, 'x');
    uint64_t id = d.dispatch("query", big_context);
    assert(id != 0);

    auto job = d.ring_for(AgentRole::Proposer).pop();
    assert(job.has_value());
    assert(job->context_len == MAX_CONTEXT_BYTES);

    std::cout << "✓ test_context_truncation\n";
}

void test_concurrent_consumers() {
    Dispatcher d;
    const int NUM_DISPATCHES = 100;
    std::atomic<int> consumed{0};

    // Dispatch 100 batches
    std::thread producer([&]() {
        for (int i = 0; i < NUM_DISPATCHES; i++) {
            d.dispatch("concurrent query", "ctx");
        }
    });

    // One consumer thread per agent role
    std::vector<std::thread> consumers;
    for (uint8_t role = 0; role < static_cast<uint8_t>(AgentRole::COUNT); ++role) {
        consumers.emplace_back([&, role]() {
            int count = 0;
            while (count < NUM_DISPATCHES) {
                auto job = d.ring_for(static_cast<AgentRole>(role)).pop();
                if (job.has_value()) count++;
            }
            consumed.fetch_add(count, std::memory_order_relaxed);
        });
    }

    producer.join();
    for (auto& t : consumers) t.join();

    assert(consumed.load() == NUM_DISPATCHES * static_cast<int>(AgentRole::COUNT));
    std::cout << "✓ test_concurrent_consumers (" 
              << NUM_DISPATCHES << " dispatches × 4 agents)\n";
}

int main() {
    std::cout << "Running dispatcher tests...\n";
    test_dispatch_creates_four_jobs();
    test_job_ids_increment();
    test_round_field_preserved();
    test_context_truncation();
    test_concurrent_consumers();
    std::cout << "\nAll tests passed.\n";
    return 0;
}
