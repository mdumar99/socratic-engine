#pragma once
#include <cstdint>
#include <cstring>

namespace se::kg {

static constexpr std::size_t MAX_LABEL_LEN    = 128;
static constexpr std::size_t MAX_CONTENT_LEN  = 1024;
static constexpr std::size_t MAX_EMBEDDING_DIM = 384;

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

struct KGNode {
    uint64_t    id;
    float       confidence;
    uint32_t    label_len;
    uint32_t    content_len;
    char        label[MAX_LABEL_LEN];
    char        content[MAX_CONTENT_LEN];
};

struct KGEdge {
    uint64_t     src_id;
    uint64_t     dst_id;
    RelationType relation;
    float        weight;
};

inline KGNode make_node_helper(uint64_t id,
                                const char* label,
                                const char* content,
                                float conf = 0.9f)
{
    KGNode n{};
    n.id          = id;
    n.confidence  = conf;
    n.label_len   = static_cast<uint32_t>(strlen(label));
    n.content_len = static_cast<uint32_t>(strlen(content));
    strncpy(n.label,   label,   MAX_LABEL_LEN - 1);
    strncpy(n.content, content, MAX_CONTENT_LEN - 1);
    return n;
}

} // namespace se::kg
