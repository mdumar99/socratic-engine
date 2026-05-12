#include "agent_runner.hpp"
#include "../dispatcher/dispatcher.hpp"
#include "../knowledge_graph/knowledge_graph.hpp"
#include <iostream>
#include <mutex>
#include <vector>
#include <chrono>
#include <thread>
#include <algorithm>
#include <filesystem>
#include <cstring>

using namespace se::agents;
using namespace se::dispatcher;
using namespace se::kg;

int main() {
    const std::string MODEL_PATH = std::string("/home/") + getenv("USER") + "/models/phi3-mini-q4.gguf";
    const std::string KG_PATH    = "/tmp/se_smoke_kg";

    std::cout << "=== Socratic Engine Smoke Test ===\n\n";

    // 1. Build a small knowledge graph
    std::cout << "→ Building knowledge graph...\n";
    KnowledgeGraph kg(KG_PATH);

    kg.upsert_node(make_node_helper(1, "lightning",
        "An electrostatic discharge between charged regions of a cloud or between cloud and ground"));
    kg.upsert_node(make_node_helper(2, "electric charge separation",
        "Occurs when ice crystals and water droplets collide inside thunderstorm clouds"));
    kg.upsert_node(make_node_helper(3, "stepped leader",
        "Invisible channel of ionised air that propagates from cloud toward ground before lightning"));
    kg.upsert_node(make_node_helper(4, "return stroke",
        "The bright visible flash when current flows up the completed channel at ~1/3 speed of light"));

    kg.upsert_edge({1, 2, RelationType::RELATED_TO, 1.0f});
    kg.upsert_edge({1, 3, RelationType::HAS_PROPERTY, 0.9f});
    kg.upsert_edge({3, 4, RelationType::CAUSES, 1.0f});

    // BFS from lightning node, 2 hops
    auto kg_nodes = kg.bfs(1, 2);
    std::string context = kg.to_context_string(kg_nodes);
    std::cout << "  KG context (" << kg_nodes.size() << " nodes):\n" << context << "\n";

    // 2. Set up dispatcher
    Dispatcher dispatcher;

    // 3. Collect results
    std::mutex result_mutex;
    std::vector<AgentResult> results;

    auto on_result = [&](AgentResult r) {
        std::lock_guard<std::mutex> lock(result_mutex);
        const char* role_names[] = {"Proposer", "Critic", "Retriever", "Synthesizer"};
        std::cout << "  [" << role_names[static_cast<int>(r.role)] << "] done\n" << std::flush;
        results.push_back(std::move(r));
    };

    // 4. Start agent runners (one per role)
    std::cout << "→ Starting 4 agent runners...\n";
    std::vector<std::unique_ptr<AgentRunner>> runners;
    for (uint8_t i = 0; i < static_cast<uint8_t>(AgentRole::COUNT); ++i) {
        auto role = static_cast<AgentRole>(i);
        runners.push_back(std::make_unique<AgentRunner>(
            dispatcher.ring_for(role),
            MODEL_PATH,
            on_result
        ));
        runners.back()->start();
    }

    // 5. Dispatch query
    const std::string query = "What causes lightning?";
    std::cout << "→ Dispatching: \"" << query << "\"\n\n";

    auto t_start = std::chrono::steady_clock::now();
    dispatcher.dispatch(query, context, 0);

    // 6. Wait for all 4 results (timeout 300s)
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(300);
    while (std::chrono::steady_clock::now() < timeout) {
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            if (results.size() >= 4) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto t_end = std::chrono::steady_clock::now();
    double elapsed = std::chrono::duration<double>(t_end - t_start).count();

    // 7. Stop runners
    for (auto& r : runners) r->stop();

    // 8. Print results in role order
    const char* role_names[] = {"Proposer", "Critic", "Retriever", "Synthesizer"};
    std::cout << "\n=== Agent outputs ===\n\n";

    std::sort(results.begin(), results.end(),
        [](const AgentResult& a, const AgentResult& b) {
            return static_cast<int>(a.role) < static_cast<int>(b.role);
        });

    for (const auto& r : results) {
        std::cout << "── " << role_names[static_cast<int>(r.role)] << " ──\n"
                  << (r.success ? r.output : "[FAILED]") << "\n\n";
    }

    std::cout << "=== Total time: " << elapsed << "s ===\n";

    std::filesystem::remove_all(KG_PATH);
    return 0;
}
