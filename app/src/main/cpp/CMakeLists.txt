cmake_minimum_required(VERSION 3.22.1)
project(offlinechat)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED true)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED true)

if(DEFINED ANDROID_ABI)
    message(STATUS "Detected Android ABI: ${ANDROID_ABI}")
    if(ANDROID_ABI STREQUAL "arm64-v8a")
        set(GGML_SYSTEM_ARCH "ARM")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    elseif(ANDROID_ABI STREQUAL "x86_64")
        set(GGML_SYSTEM_ARCH "x86")
        set(GGML_CPU_KLEIDIAI OFF)
        set(GGML_OPENMP OFF)
    else()
        message(WARNING "Unsupported ABI: ${ANDROID_ABI}")
    endif()
endif()

# GPU: OpenCL (Qualcomm Adreno optimized)
set(GGML_OPENCL ON CACHE BOOL "ggml: use OpenCL" FORCE)
set(GGML_OPENCL_USE_ADRENO_KERNELS ON CACHE BOOL "" FORCE)
set(GGML_OPENCL_EMBED_KERNELS ON CACHE BOOL "" FORCE)
set(GGML_VULKAN OFF CACHE BOOL "" FORCE)

# OpenCL headers
set(OPENCL_HEADERS_DIR ${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/OpenCL-Headers)

# OpenCL loader stub (dlopen-based forwarder)
find_library(log-lib log)
add_library(OpenCL_stub SHARED opencl_stub.c)
target_include_directories(OpenCL_stub PRIVATE ${OPENCL_HEADERS_DIR})
target_link_libraries(OpenCL_stub PRIVATE ${log-lib} dl)
set_target_properties(OpenCL_stub PROPERTIES
    OUTPUT_NAME "OpenCL"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_CURRENT_BINARY_DIR}/opencl_lib"
)

# Make find_package(OpenCL) succeed
set(OpenCL_FOUND TRUE CACHE BOOL "" FORCE)
set(OpenCL_INCLUDE_DIRS "${OPENCL_HEADERS_DIR}" CACHE PATH "" FORCE)
set(OpenCL_LIBRARIES "${CMAKE_CURRENT_BINARY_DIR}/opencl_lib/libOpenCL.so" CACHE FILEPATH "" FORCE)
set(OpenCL_INCLUDE_DIR "${OPENCL_HEADERS_DIR}" CACHE PATH "" FORCE)
set(OpenCL_LIBRARY "${CMAKE_CURRENT_BINARY_DIR}/opencl_lib/libOpenCL.so" CACHE FILEPATH "" FORCE)

if(NOT TARGET OpenCL::OpenCL)
    add_library(OpenCL::OpenCL SHARED IMPORTED)
    set_target_properties(OpenCL::OpenCL PROPERTIES
        IMPORTED_LOCATION "${CMAKE_CURRENT_BINARY_DIR}/opencl_lib/libOpenCL.so"
        INTERFACE_INCLUDE_DIRECTORIES "${OPENCL_HEADERS_DIR}"
    )
    add_dependencies(OpenCL::OpenCL OpenCL_stub)
endif()

macro(find_package)
    if("${ARGV0}" STREQUAL "OpenCL")
        message(STATUS "find_package(OpenCL) intercepted — using stub")
    else()
        _find_package(${ARGV})
    endif()
endmacro()

# llama.cpp
set(LLAMA_SRC ${CMAKE_CURRENT_LIST_DIR}/../../../../third_party/llama.cpp)
add_subdirectory(${LLAMA_SRC} ${CMAKE_CURRENT_BINARY_DIR}/build-llama)

add_library(offlinechat SHARED native-lib.cpp)

target_compile_definitions(offlinechat PRIVATE
        GGML_SYSTEM_ARCH=${GGML_SYSTEM_ARCH}
        GGML_OPENCL=1
)

target_include_directories(offlinechat PRIVATE
        ${LLAMA_SRC}
        ${LLAMA_SRC}/common
        ${LLAMA_SRC}/include
        ${LLAMA_SRC}/ggml/include
        ${LLAMA_SRC}/ggml/src
)

find_library(android-lib android)

target_link_libraries(offlinechat
        llama
        ${android-lib}
        ${log-lib}
)
