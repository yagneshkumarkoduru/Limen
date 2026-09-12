cmake_minimum_required(VERSION 3.16)
project(limen_firmware C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)

add_library(limen_fw STATIC
    limen_features.c
    limen_sqg.c
    limen_inference.c
    limen_supervisor.c
    limen_trace.c
    limen_pipeline.c
)
target_include_directories(limen_fw PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

if(NOT CMAKE_CROSSCOMPILING)
    add_executable(limen_conformance tests/test_conformance.c)
    target_include_directories(limen_conformance PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/tests)
    target_link_libraries(limen_conformance PRIVATE limen_fw)
    enable_testing()
    add_test(NAME conformance COMMAND limen_conformance)
endif()
