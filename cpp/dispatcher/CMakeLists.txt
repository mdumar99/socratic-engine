set(LLAMA_CPP_DIR $ENV{HOME}/llama.cpp)
set(LLAMA_BIN_DIR ${LLAMA_CPP_DIR}/build/bin)

if(EXISTS ${LLAMA_CPP_DIR}/include/llama.h)
    add_executable(debate_smoke_test debate_smoke_test.cpp)
    target_include_directories(debate_smoke_test PRIVATE
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/ggml/include
    )
    target_link_libraries(debate_smoke_test PRIVATE
        rocksdb
        ${LLAMA_BIN_DIR}/libllama.so
        ${LLAMA_BIN_DIR}/libggml.so
        ${LLAMA_BIN_DIR}/libggml-base.so
        ${LLAMA_BIN_DIR}/libggml-cpu.so
        ${LLAMA_BIN_DIR}/libggml-cuda.so
    )
    target_compile_options(debate_smoke_test PRIVATE -Wall -Wextra -O2)
    message(STATUS "debate_smoke_test target enabled")
endif()
