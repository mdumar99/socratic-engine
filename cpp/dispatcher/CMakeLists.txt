set(LLAMA_CPP_DIR $ENV{HOME}/llama.cpp)
set(LLAMA_BIN_DIR ${LLAMA_CPP_DIR}/build/bin)

if(EXISTS ${LLAMA_CPP_DIR}/include/llama.h)
    set(LLAMA_INCLUDES
        ${LLAMA_CPP_DIR}/include
        ${LLAMA_CPP_DIR}/ggml/include
    )
    set(LLAMA_LIBS
        rocksdb
        ${LLAMA_BIN_DIR}/libllama.so
        ${LLAMA_BIN_DIR}/libggml.so
        ${LLAMA_BIN_DIR}/libggml-base.so
        ${LLAMA_BIN_DIR}/libggml-cpu.so
        ${LLAMA_BIN_DIR}/libggml-cuda.so
    )

    add_executable(debate_smoke_test debate_smoke_test.cpp)
    target_include_directories(debate_smoke_test PRIVATE ${LLAMA_INCLUDES})
    target_link_libraries(debate_smoke_test PRIVATE ${LLAMA_LIBS})
    target_compile_options(debate_smoke_test PRIVATE -Wall -Wextra -O2)

    add_executable(debate_runner debate_runner.cpp)
    target_include_directories(debate_runner PRIVATE ${LLAMA_INCLUDES})
    target_link_libraries(debate_runner PRIVATE ${LLAMA_LIBS})
    target_compile_options(debate_runner PRIVATE -Wall -Wextra -O2)

    message(STATUS "debate_smoke_test + debate_runner targets enabled")
endif()
