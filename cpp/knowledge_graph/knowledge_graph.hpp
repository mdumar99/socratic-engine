#pragma once
#include "kg_types.hpp"
#include <rocksdb/db.h>
#include <rocksdb/options.h>
#include <string>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <queue>
#include <optional>
#include <memory>
#include <cstring>
#include <sstream>

namespace se::kg {

/**
 * RocksDB-backed knowledge graph with BFS traversal.
 *
 * Storage layout:
 *   node:{id}        -> serialised KGNode
 *   edge:{src}:{dst} -> serialised KGEdge
 *   adj:{id}         -> comma-separated list of neighbour ids
 *
 * All operations are synchronous. Thread safety: external locking
 * required for concurrent writes; concurrent reads are safe.
 */
class KnowledgeGraph {
public:
    explicit KnowledgeGraph(const std::string& db_path) {
        rocksdb::Options opts;
        opts.create_if_missing = true;
        opts.compression = rocksdb::kLZ4Compression;

        rocksdb::DB* raw_db = nullptr;
        auto status = rocksdb::DB::Open(opts, db_path, &raw_db);
        if (!status.ok()) {
            throw std::runtime_error("KG: failed to open RocksDB: " + status.ToString());
        }
        db_.reset(raw_db);
    }

    ~KnowledgeGraph() = default;

    // Non-copyable
    KnowledgeGraph(const KnowledgeGraph&) = delete;
    KnowledgeGraph& operator=(const KnowledgeGraph&) = delete;

    /// Insert or update a node
    bool upsert_node(const KGNode& node) {
        const std::string key = node_key(node.id);
        const std::string val = serialise_node(node);
        auto s = db_->Put(rocksdb::WriteOptions(), key, val);
        return s.ok();
    }

    /// Insert a directed edge and update adjacency list
    bool upsert_edge(const KGEdge& edge) {
        // Store edge
        const std::string key = edge_key(edge.src_id, edge.dst_id);
        const std::string val = serialise_edge(edge);
        auto s = db_->Put(rocksdb::WriteOptions(), key, val);
        if (!s.ok()) return false;

        // Update adjacency list for src
        append_neighbour(edge.src_id, edge.dst_id);
        return true;
    }

    /// Retrieve a node by id
    std::optional<KGNode> get_node(uint64_t id) const {
        std::string val;
        auto s = db_->Get(rocksdb::ReadOptions(), node_key(id), &val);
        if (!s.ok()) return std::nullopt;
        return deserialise_node(val);
    }

    /// BFS from seed node up to max_hops, returns nodes in traversal order
    std::vector<KGNode> bfs(uint64_t seed_id,
                             int max_hops = 2,
                             std::size_t max_nodes = 32) const
    {
        std::vector<KGNode> result;
        std::unordered_set<uint64_t> visited;
        std::queue<std::pair<uint64_t, int>> frontier;  // {id, depth}

        frontier.push({seed_id, 0});
        visited.insert(seed_id);

        while (!frontier.empty() && result.size() < max_nodes) {
            auto [cur_id, depth] = frontier.front();
            frontier.pop();

            auto node = get_node(cur_id);
            if (node) result.push_back(*node);

            if (depth >= max_hops) continue;

            for (uint64_t neighbour : get_neighbours(cur_id)) {
                if (!visited.count(neighbour)) {
                    visited.insert(neighbour);
                    frontier.push({neighbour, depth + 1});
                }
            }
        }

        return result;
    }

    /// Find nodes whose labels contain the query string (simple scan)
    std::vector<KGNode> search_by_label(const std::string& query,
                                         std::size_t max_results = 8) const
    {
        std::vector<KGNode> results;
        auto it = std::unique_ptr<rocksdb::Iterator>(
            db_->NewIterator(rocksdb::ReadOptions()));

        for (it->SeekToFirst(); it->Valid() && results.size() < max_results; it->Next()) {
            const auto key = it->key().ToString();
            if (key.rfind("node:", 0) != 0) continue;

            auto node = deserialise_node(it->value().ToString());
            if (node) {
                std::string label(node->label, node->label_len);
                if (label.find(query) != std::string::npos) {
                    results.push_back(*node);
                }
            }
        }
        return results;
    }

    /// Serialise BFS result into a context string for agent injection
    std::string to_context_string(const std::vector<KGNode>& nodes) const {
        std::ostringstream oss;
        oss << "[Knowledge Graph Context]\n";
        for (const auto& n : nodes) {
            oss << "- " << std::string(n.label, n.label_len)
                << ": " << std::string(n.content, n.content_len)
                << " (confidence=" << n.confidence << ")\n";
        }
        return oss.str();
    }

    std::size_t node_count() const {
        uint64_t count = 0;
        auto it = std::unique_ptr<rocksdb::Iterator>(
            db_->NewIterator(rocksdb::ReadOptions()));
        for (it->SeekToFirst(); it->Valid(); it->Next()) {
            if (it->key().ToString().rfind("node:", 0) == 0) count++;
        }
        return count;
    }

private:
    std::unique_ptr<rocksdb::DB> db_;

    static std::string node_key(uint64_t id) {
        return "node:" + std::to_string(id);
    }
    static std::string edge_key(uint64_t src, uint64_t dst) {
        return "edge:" + std::to_string(src) + ":" + std::to_string(dst);
    }
    static std::string adj_key(uint64_t id) {
        return "adj:" + std::to_string(id);
    }

    void append_neighbour(uint64_t src, uint64_t dst) {
        std::string existing;
        db_->Get(rocksdb::ReadOptions(), adj_key(src), &existing);
        if (!existing.empty()) existing += ",";
        existing += std::to_string(dst);
        db_->Put(rocksdb::WriteOptions(), adj_key(src), existing);
    }

    std::vector<uint64_t> get_neighbours(uint64_t id) const {
        std::string val;
        db_->Get(rocksdb::ReadOptions(), adj_key(id), &val);
        if (val.empty()) return {};

        std::vector<uint64_t> neighbours;
        std::stringstream ss(val);
        std::string token;
        while (std::getline(ss, token, ',')) {
            if (!token.empty()) neighbours.push_back(std::stoull(token));
        }
        return neighbours;
    }

    static std::string serialise_node(const KGNode& n) {
        std::string out(sizeof(KGNode), '\0');
        std::memcpy(out.data(), &n, sizeof(KGNode));
        return out;
    }

    static std::optional<KGNode> deserialise_node(const std::string& s) {
        if (s.size() < sizeof(KGNode)) return std::nullopt;
        KGNode n{};
        std::memcpy(&n, s.data(), sizeof(KGNode));
        return n;
    }

    static std::string serialise_edge(const KGEdge& e) {
        std::string out(sizeof(KGEdge), '\0');
        std::memcpy(out.data(), &e, sizeof(KGEdge));
        return out;
    }
};

} // namespace se::kg
