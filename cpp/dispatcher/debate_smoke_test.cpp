#include "debate_orchestrator.hpp"
#include "../knowledge_graph/kg_types.hpp"
#include <iostream>
#include <filesystem>
#include "ggml.h"

using namespace se::dispatcher;
using namespace se::kg;

// Silence llama.cpp internal logs — only show our own output
static void null_log(ggml_log_level, const char*, void*) {}

int main() {
    llama_log_set(null_log, nullptr);

    const std::string MODEL_PATH = std::string("/home/") +
        getenv("USER") + "/models/phi3-mini-q4.gguf";
    const std::string KG_PATH = "/tmp/se_debate_kg";

    std::cout << "=== Socratic Engine — 4-Round Debate ===\n\n";

    std::cout << "→ Loading model (one-time)...\n";
    LlamaEngine engine(MODEL_PATH, 99);
    std::cout << "  Done.\n\n";

    std::cout << "→ Building knowledge graph...\n";
    KnowledgeGraph kg(KG_PATH);
    kg.upsert_node(make_node_helper(1, "lightning",
        "Electrostatic discharge between charged cloud regions or cloud and ground"));
    kg.upsert_node(make_node_helper(2, "charge separation",
        "Ice crystals and water droplets collide in thunderstorm clouds"));
    kg.upsert_node(make_node_helper(3, "stepped leader",
        "Invisible ionised air channel from cloud toward ground"));
    kg.upsert_node(make_node_helper(4, "return stroke",
        "Bright flash when current flows up the completed channel"));
    kg.upsert_node(make_node_helper(5, "thunder",
        "Sound from rapid air expansion around lightning channel"));
    kg.upsert_edge({1, 2, RelationType::RELATED_TO,   1.0f});
    kg.upsert_edge({1, 3, RelationType::HAS_PROPERTY, 0.9f});
    kg.upsert_edge({3, 4, RelationType::CAUSES,       1.0f});
    kg.upsert_edge({1, 5, RelationType::CAUSES,       0.95f});
    std::cout << "  " << kg.node_count() << " nodes.\n\n";

    std::cout << "→ Initialising debate orchestrator...\n";
    DebateOrchestrator orchestrator(engine, kg, 150, 0.7f);

    std::cout << "→ Running debate: \"What causes lightning?\"\n\n";
    auto result = orchestrator.run("What causes lightning?", 1);

    std::cout << "\n╔══════════════════════════════════════╗\n";
    std::cout <<   "║        DEBATE TRANSCRIPT             ║\n";
    std::cout <<   "╚══════════════════════════════════════╝\n\n";
    std::cout << "Query: " << result.query << "\n\n";
    std::cout << "── Proposer ──\n" << result.proposer_output  << "\n\n";
    std::cout << "── Critic ──\n"   << result.critic_output    << "\n\n";
    std::cout << "── Retriever ──\n"<< result.retriever_output << "\n\n";
    std::cout << "── Final Answer (Synthesizer) ──\n"
              << result.final_answer << "\n\n";
    std::cout << "Time: " << result.elapsed_seconds << "s | "
              << "Success: " << (result.success ? "yes" : "no") << "\n";

    std::filesystem::remove_all(KG_PATH);
    return result.success ? 0 : 1;
}
