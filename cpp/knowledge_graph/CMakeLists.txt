# knowledge_graph — header-only for most targets
# kg_ingest: standalone binary for Python KG builder

add_executable(kg_ingest kg_ingest.cpp)
target_link_libraries(kg_ingest PRIVATE rocksdb)
target_compile_options(kg_ingest PRIVATE -Wall -Wextra -O2)
