#include "../knowledge_graph/kg_types.hpp"
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

using namespace se::agents;
using namespace se::dispatcher;
using namespace se::kg;

int main() {
    const std::string MODEL_PATH = std::string("/home/") + getenv("USER") + "/models/phi3-mini-q4.gguf";
    const std::string KG_PATH    = "/tmp/se_smoke_kg";

    std::cout << "=== Socratic Engine Smoke Test ===\n\n";

    // 1. Load model ONCE — shared across all agents
    std::cout << "→ Loading model (one-time)...\n";
    LlamaEngine engine(MODEL_PATH, 99);
    std::cout << "  Model loaded.\n\n";

    // 2. Build knowledge graph
    std::cout << "→ Building knowledge graph...\n";
    KnowledgeGraph kg(KG_PATH);
    kg.upsert_node(make_node_helper(1, "lightning",
        "Electrostatic discharge between charged regions of a cloud or ground"));
    kg.upsert_node(make_node_helper(2, "charge separation",
        "Ice crystals and water droplets collide inside thunderstorm clouds"));
    kg.upsert_node(make_node_helper(3, "stepped leader",
        "Invisible ionised air channel propagating from cloud toward ground"));
    kg.upsert_node(make_node_helper(4, "return stroke",
        "Bright flash when current flows up the completed channel"));
    kg.upsert_edge({1, 2, RelationType::RELATED_TO, 1.0f});
    kg.upsert_edge({1, 3, RelationType::HAS_PROPERTY, 0.9f});
    kg.upsert_edge({3, 4, RelationType::CAUSES, 1.0f});

    auto kg_nodes = kg.bfs(1, 2);
    std::string context = kg.to_context_string(kg_nodes);
    std::cout << "  KG context (" << kg_nodes.size() << " nodes):\n" << context << "\n";

    // 3. Dispatcher
    Dispatcher dispatcher;

    // 4. Collect results
    std::mutex result_mutex;
    std::vector<AgentResult> results;
    const char* role_names[] = {"Proposer", "Critic", "Retriever", "Synthesizer"};

    auto on_result = [&](AgentResult r) {
        std::lock_guard<std::mutex> lock(result_mutex);
        std::cout << "  [" << role_names[static_cast<int>(r.role)] << "] done\n" << std::flush;
        results.push_back(std::move(r));
    };

    // 5. Start 4 agents — each gets own context, all share engine
    std::cout << "→ Starting 4 agents (shared model, 4 contexts)...\n";
    std::vector<std::unique_ptr<AgentRunner>> runners;
    for (uint8_t i = 0; i < static_cast<uint8_t>(AgentRole::COUNT); ++i) {
        runners.push_back(std::make_unique<AgentRunner>(
            dispatcher.ring_for(static_cast<AgentRole>(i)),
            engine,
            on_result,
            200,
            0.7f
        ));
        runners.back()->start();
    }

    // 6. Dispatch query
    const std::string query = "What causes lightning?";
    std::cout << "→ Dispatching: \"" << query << "\"\n\n";

    auto t_start = std::chrono::steady_clock::now();
    dispatcher.dispatch(query, context, 0);

    // 7. Wait for all 4 results (5 minute timeout)
    auto timeout = std::chrono::steady_clock::now() + std::chrono::seconds(300);
    while (std::chrono::steady_clock::now() < timeout) {
        {
            std::lock_guard<std::mutex> lock(result_mutex);
            if (results.size() >= 4) break;
        }
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
    }

    auto elapsed = std::chrono::duration<double>(
        std::chrono::steady_clock::now() - t_start).count();

    // 8. Stop runners
    for (auto& r : runners) r->stop();

    // 9. Print results in role order
    std::sort(results.begin(), results.end(),
        [](const AgentResult& a, const AgentResult& b) {
            return static_cast<int>(a.role) < static_cast<int>(b.role);
        });

    std::cout << "\n=== Agent outputs ===\n\n";
    for (const auto& r : results) {
        std::cout << "── " << role_names[static_cast<int>(r.role)] << " ──\n"
                  << (r.success ? r.output : "[FAILED — empty output]")
                  << "\n\n";
    }

    std::cout << "=== Total wall time: " << elapsed << "s ===\n";

    std::filesystem::remove_all(KG_PATH);
    return 0;
}
