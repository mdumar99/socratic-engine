set(LLAMA_CPP_DIR $ENV{HOME}/llama.cpp)
set(LLAMA_BIN_DIR ${LLAMA_CPP_DIR}/build/bin)

# Only build smoke_test if llama.cpp is present locally
# CI skips this target — it requires GPU + model file
if(EXISTS ${LLAMA_CPP_DIR}/include/llama.h)
    add_executable(smoke_test smoke_test.cpp)
    target_include_directories(smoke_test PRIVATE
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/ggml/include
    )
    target_link_libraries(smoke_test PRIVATE
        rocksdb
        ${LLAMA_BIN_DIR}/libllama.so
        ${LLAMA_BIN_DIR}/libggml.so
        ${LLAMA_BIN_DIR}/libggml-base.so
        ${LLAMA_BIN_DIR}/libggml-cpu.so
        ${LLAMA_BIN_DIR}/libggml-cuda.so
    )
    target_compile_options(smoke_test PRIVATE -Wall -Wextra -O2)
    message(STATUS "llama.cpp found — smoke_test target enabled")
else()
    message(STATUS "llama.cpp not found — smoke_test target skipped")
endif()
