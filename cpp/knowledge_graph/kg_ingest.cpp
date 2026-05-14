/**
 * kg_ingest.cpp — KG ingestion binary
 *
 * Reads JSON lines from stdin, writes KGNodes/KGEdges to RocksDB.
 *
 * Input formats (one JSON per line):
 *   {"op":"node","id":1,"label":"lightning","content":"...","confidence":0.9}
 *   {"op":"edge","src":1,"dst":2,"relation":"CAUSES","weight":1.0}
 *   {"op":"count"}
 */
#include "knowledge_graph.hpp"
#include "kg_types.hpp"
#include <iostream>
#include <string>
#include <unordered_map>
#include <regex>
#include <cstring>

using namespace se::kg;

/// Extract a string value for a key from a JSON line.
/// Handles both "key":"value" and "key": "value"
static std::string jstr(const std::string& json, const std::string& key) {
    std::regex re("\"" + key + "\"\\s*:\\s*\"((?:[^\"\\\\]|\\\\.)*)\"");
    std::smatch m;
    if (!std::regex_search(json, m, re)) return "";
    std::string val = m[1].str();
    // Unescape
    std::string result;
    for (size_t i = 0; i < val.size(); ++i) {
        if (val[i] == '\\' && i + 1 < val.size()) {
            ++i;
            switch (val[i]) {
                case 'n':  result += '\n'; break;
                case 't':  result += '\t'; break;
                case '"':  result += '"';  break;
                case '\\': result += '\\'; break;
                default:   result += val[i]; break;
            }
        } else {
            result += val[i];
        }
    }
    return result;
}

/// Extract a numeric value for a key from a JSON line.
static double jnum(const std::string& json, const std::string& key, double def = 0.0) {
    std::regex re("\"" + key + "\"\\s*:\\s*([0-9.eE+\\-]+)");
    std::smatch m;
    if (!std::regex_search(json, m, re)) return def;
    try { return std::stod(m[1].str()); }
    catch (...) { return def; }
}

static const std::unordered_map<std::string, RelationType> RELATION_MAP = {
    {"IS_A",         RelationType::IS_A},
    {"HAS_PROPERTY", RelationType::HAS_PROPERTY},
    {"CAUSES",       RelationType::CAUSES},
    {"CONTRADICTS",  RelationType::CONTRADICTS},
    {"SUPPORTS",     RelationType::SUPPORTS},
    {"PART_OF",      RelationType::PART_OF},
    {"RELATED_TO",   RelationType::RELATED_TO},
};

static void emit_ok(const std::string& op, uint64_t id = 0) {
    std::cout << "{\"status\":\"ok\",\"op\":\"" << op << "\"";
    if (id) std::cout << ",\"id\":" << id;
    std::cout << "}\n" << std::flush;
}

static void emit_error(const std::string& msg) {
    std::cout << "{\"status\":\"error\",\"message\":\"" << msg << "\"}\n"
              << std::flush;
}

int main(int argc, char* argv[]) {
    std::string kg_path = "/tmp/se_runtime_kg";
    for (int i = 1; i < argc; i++) {
        std::string arg = argv[i];
        if (arg == "--kg" && i + 1 < argc) kg_path = argv[++i];
    }

    KnowledgeGraph kg(kg_path);
    std::string line;

    while (std::getline(std::cin, line)) {
        if (line.empty()) continue;

        const std::string op = jstr(line, "op");

        if (op == "node") {
            uint64_t id        = static_cast<uint64_t>(jnum(line, "id"));
            std::string label  = jstr(line, "label");
            std::string content= jstr(line, "content");
            float conf         = static_cast<float>(jnum(line, "confidence", 0.9));

            if (id == 0 || label.empty()) {
                emit_error("node requires id and label");
                continue;
            }
            KGNode node = make_node_helper(id, label.c_str(), content.c_str(), conf);
            if (kg.upsert_node(node)) emit_ok("node", id);
            else emit_error("failed to write node");
        }
        else if (op == "edge") {
            uint64_t src       = static_cast<uint64_t>(jnum(line, "src"));
            uint64_t dst       = static_cast<uint64_t>(jnum(line, "dst"));
            std::string rel_str= jstr(line, "relation");
            float weight       = static_cast<float>(jnum(line, "weight", 1.0));

            auto it = RELATION_MAP.find(rel_str);
            if (it == RELATION_MAP.end()) {
                emit_error("unknown relation: " + rel_str);
                continue;
            }
            KGEdge edge{src, dst, it->second, weight};
            if (kg.upsert_edge(edge)) emit_ok("edge");
            else emit_error("failed to write edge");
        }
        else if (op == "count") {
            std::cout << "{\"status\":\"ok\",\"op\":\"count\","
                      << "\"count\":" << kg.node_count() << "}\n" << std::flush;
        }
        else {
            emit_error("unknown op: " + op);
        }
    }

    return 0;
}
