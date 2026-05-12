add_executable(smoke_test smoke_test.cpp)
target_link_libraries(smoke_test PRIVATE rocksdb)
target_compile_options(smoke_test PRIVATE -Wall -Wextra -O2)
