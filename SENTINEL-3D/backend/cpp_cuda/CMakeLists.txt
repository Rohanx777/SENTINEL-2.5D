cmake_minimum_required(VERSION 3.18)
project(SentinelCUDA LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

# ── Find CUDA ──────────────────────────────────────────────────────────────────
find_package(CUDAToolkit REQUIRED)

# ── Find LibTorch ──────────────────────────────────────────────────────────────
# Set LibTorch path manually if auto-detection fails
# Download from: https://pytorch.org/get-started/locally/
# Windows: https://download.pytorch.org/libtorch/cu121/libtorch-win-shared-with-deps-2.1.0%2Bcu121.zip
set(CMAKE_PREFIX_PATH "${CMAKE_PREFIX_PATH};C:/libtorch")
find_package(Torch REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${TORCH_CXX_FLAGS}")

# ── uWebSockets (header-only + uSockets static lib) ────────────────────────────
# Clone: git clone https://github.com/uNetworking/uWebSockets third_party/uWebSockets
# Clone: git clone https://github.com/uNetworking/uSockets third_party/uSockets
set(USOCKETS_DIR "${CMAKE_CURRENT_SOURCE_DIR}/third_party/uSockets")
if(EXISTS ${USOCKETS_DIR})
    add_library(uSockets STATIC
        ${USOCKETS_DIR}/src/bsd.c
        ${USOCKETS_DIR}/src/context.c
        ${USOCKETS_DIR}/src/loop.c
        ${USOCKETS_DIR}/src/socket.c
        ${USOCKETS_DIR}/src/eventing/epoll_kqueue.c
        ${USOCKETS_DIR}/src/crypto/openssl.c
    )
    target_include_directories(uSockets PUBLIC ${USOCKETS_DIR}/src)
    target_compile_definitions(uSockets PUBLIC LIBUS_USE_OPENSSL)
    if(WIN32)
        target_link_libraries(uSockets ws2_32)
    endif()
endif()

# ── nlohmann/json (header-only) ────────────────────────────────────────────────
# Download single-header: https://github.com/nlohmann/json/releases/download/v3.11.3/json.hpp
# Place in: third_party/json/json.hpp
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/third_party)

# ── CUDA kernels ───────────────────────────────────────────────────────────────
add_library(cuda_kernels STATIC
    src/cuda_voxelizer.cu
    src/cuda_grid_builder.cu
)
target_include_directories(cuda_kernels PUBLIC
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CUDA_INCLUDE_DIRS}
)
target_link_libraries(cuda_kernels PUBLIC CUDA::cudart)
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    CUDA_ARCHITECTURES "75;80;86;89;90"  # Turing, Ampere, Ada, Hopper
    POSITION_INDEPENDENT_CODE ON
)

# ── Main executable ────────────────────────────────────────────────────────────
add_executable(sentinel_bridge
    src/sentinel_bridge.cpp
    src/websocket_server.cpp
    src/inference_engine.cpp
)

target_include_directories(sentinel_bridge PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
    ${CMAKE_CURRENT_SOURCE_DIR}/third_party/uWebSockets/src
    ${CUDA_INCLUDE_DIRS}
)

target_link_libraries(sentinel_bridge PRIVATE
    cuda_kernels
    ${TORCH_LIBRARIES}
    CUDA::cudart
)

if(TARGET uSockets)
    target_link_libraries(sentinel_bridge PRIVATE uSockets)
endif()

if(WIN32)
    target_link_libraries(sentinel_bridge PRIVATE ws2_32)
endif()

# ── Copy torch DLLs on Windows ─────────────────────────────────────────────────
if(WIN32)
    file(GLOB TORCH_DLLS "${TORCH_INSTALL_PREFIX}/lib/*.dll")
    add_custom_command(TARGET sentinel_bridge POST_BUILD
        COMMAND ${CMAKE_COMMAND} -E copy_if_different
        ${TORCH_DLLS}
        $<TARGET_FILE_DIR:sentinel_bridge>
    )
endif()

# ── Python bindings (optional) ─────────────────────────────────────────────────
option(BUILD_PYTHON_BINDINGS "Build pybind11 bindings" OFF)
if(BUILD_PYTHON_BINDINGS)
    find_package(Python COMPONENTS Interpreter Development REQUIRED)
    add_subdirectory(third_party/pybind11)

    pybind11_add_module(sentinel_cuda_py
        src/python_bindings.cpp
    )
    target_link_libraries(sentinel_cuda_py PRIVATE cuda_kernels ${TORCH_LIBRARIES})
endif()
