#include "../../cpp/knowledge_graph/knowledge_graph.hpp"
#include <cassert>
#include <iostream>
#include <filesystem>
#include <cstring>

using namespace se::kg;
namespace fs = std::filesystem;

// Helper to make a node
KGNode make_node(uint64_t id, const char* label, const char* content, float conf = 0.9f) {
    KGNode n{};
    n.id = id;
    n.confidence = conf;
    n.label_len  = static_cast<uint32_t>(strlen(label));
    n.content_len = static_cast<uint32_t>(strlen(content));
    strncpy(n.label,   label,   MAX_LABEL_LEN - 1);
    strncpy(n.content, content, MAX_CONTENT_LEN - 1);
    return n;
}

KGEdge make_edge(uint64_t src, uint64_t dst, RelationType rel, float w = 1.0f) {
    return KGEdge{src, dst, rel, w};
}

struct TestDB {
    std::string path;
    KnowledgeGraph kg;

    TestDB(const std::string& p) : path(p), kg(p) {}
    ~TestDB() { fs::remove_all(path); }
};

void test_upsert_and_get_node() {
    TestDB db("/tmp/kg_test_1");

    [[maybe_unused]] KGNode n = make_node(1, "lightning", "Electrical discharge in atmosphere");
    assert(db.kg.upsert_node(n));

    [[maybe_unused]] auto result = db.kg.get_node(1);
    assert(result.has_value());
    assert(result->id == 1);
    assert(std::string(result->label, result->label_len) == "lightning");

    std::cout << "✓ test_upsert_and_get_node\n";
}

void test_missing_node_returns_nullopt() {
    TestDB db("/tmp/kg_test_2");
    [[maybe_unused]] auto result = db.kg.get_node(999);
    assert(!result.has_value());
    std::cout << "✓ test_missing_node_returns_nullopt\n";
}

void test_bfs_two_hops() {
    TestDB db("/tmp/kg_test_3");

    // Build: thunderstorm -> lightning -> thunder
    db.kg.upsert_node(make_node(1, "thunderstorm", "A storm with lightning"));
    db.kg.upsert_node(make_node(2, "lightning",    "Electrical discharge"));
    db.kg.upsert_node(make_node(3, "thunder",      "Sound from lightning"));
    db.kg.upsert_edge(make_edge(1, 2, RelationType::CAUSES));
    db.kg.upsert_edge(make_edge(2, 3, RelationType::CAUSES));

    auto nodes = db.kg.bfs(1, 2);
    assert(nodes.size() == 3);

    std::cout << "✓ test_bfs_two_hops (" << nodes.size() << " nodes)\n";
}

void test_bfs_max_nodes_limit() {
    TestDB db("/tmp/kg_test_4");

    // Star graph: node 0 -> nodes 1..20
    db.kg.upsert_node(make_node(0, "root", "root node"));
    for (uint64_t i = 1; i <= 20; i++) {
        db.kg.upsert_node(make_node(i, "child", "child node"));
        db.kg.upsert_edge(make_edge(0, i, RelationType::RELATED_TO));
    }

    auto nodes = db.kg.bfs(0, 1, 5);  // max 5 nodes
    assert(nodes.size() <= 5);
    std::cout << "✓ test_bfs_max_nodes_limit (" << nodes.size() << " nodes)\n";
}

void test_search_by_label() {
    TestDB db("/tmp/kg_test_5");

    db.kg.upsert_node(make_node(1, "lightning bolt",  "Fast discharge"));
    db.kg.upsert_node(make_node(2, "lightning rod",   "Attracts lightning"));
    db.kg.upsert_node(make_node(3, "thunder",         "Sound from lightning"));

    auto results = db.kg.search_by_label("lightning");
    assert(results.size() == 2);
    std::cout << "✓ test_search_by_label (" << results.size() << " results)\n";
}

void test_context_string_generation() {
    TestDB db("/tmp/kg_test_6");

    db.kg.upsert_node(make_node(1, "lightning", "Electrical discharge in thunderstorms"));
    db.kg.upsert_node(make_node(2, "thunder",   "Sound wave from lightning channel"));

    auto nodes = db.kg.bfs(1, 0);  // just seed node
    std::string ctx = db.kg.to_context_string(nodes);

    assert(ctx.find("[Knowledge Graph Context]") != std::string::npos);
    assert(ctx.find("lightning") != std::string::npos);
    std::cout << "✓ test_context_string_generation\n";
    std::cout << "  Sample output:\n" << ctx;
}

void test_node_count() {
    TestDB db("/tmp/kg_test_7");

    for (uint64_t i = 1; i <= 5; i++) {
        db.kg.upsert_node(make_node(i, "node", "content"));
    }
    assert(db.kg.node_count() == 5);
    std::cout << "✓ test_node_count\n";
}

int main() {
    std::cout << "Running knowledge graph tests...\n\n";
    test_upsert_and_get_node();
    test_missing_node_returns_nullopt();
    test_bfs_two_hops();
    test_bfs_max_nodes_limit();
    test_search_by_label();
    test_context_string_generation();
    test_node_count();
    std::cout << "\nAll tests passed.\n";
    return 0;
}
