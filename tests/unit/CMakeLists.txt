add_executable(test_spsc_ring test_spsc_ring.cpp)
target_compile_options(test_spsc_ring PRIVATE -Wall -Wextra -O2)

add_executable(test_dispatcher test_dispatcher.cpp)
target_compile_options(test_dispatcher PRIVATE -Wall -Wextra -O2)

add_executable(test_knowledge_graph test_knowledge_graph.cpp)
target_link_libraries(test_knowledge_graph PRIVATE rocksdb)
target_compile_options(test_knowledge_graph PRIVATE -Wall -Wextra -O2)
