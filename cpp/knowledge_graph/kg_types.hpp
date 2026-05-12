#pragma once
#include <string>
#include <cstdint>
#include <array>

namespace se::kg {

/// Maximum sizes
static constexpr std::size_t MAX_LABEL_LEN    = 128;
static constexpr std::size_t MAX_CONTENT_LEN  = 1024;
static constexpr std::size_t MAX_EMBEDDING_DIM = 384;

/// Edge relation types
enum class RelationType : uint8_t {
    IS_A          = 0,
    HAS_PROPERTY  = 1,
    CAUSES        = 2,
    CONTRADICTS   = 3,
    SUPPORTS      = 4,
    PART_OF       = 5,
    RELATED_TO    = 6,
    COUNT         = 7
};

inline const char* relation_name(RelationType r) {
    static const char* names[] = {
        "IS_A", "HAS_PROPERTY", "CAUSES",
        "CONTRADICTS", "SUPPORTS", "PART_OF", "RELATED_TO"
    };
    return names[static_cast<uint8_t>(r)];
}

/// A knowledge graph node
struct KGNode {
    uint64_t    id;
    float       confidence;          // 0.0 - 1.0
    uint32_t    label_len;
    uint32_t    content_len;
    char        label[MAX_LABEL_LEN];
    char        content[MAX_CONTENT_LEN];
};

/// A directed edge between two nodes
struct KGEdge {
    uint64_t     src_id;
    uint64_t     dst_id;
    RelationType relation;
    float        weight;             // 0.0 - 1.0
};

} // namespace se::kg
