# Read rings share ring-types.h and the linux_ring interface; read-ring.h picks
# io_uring on Linux, IoRing/overlapped on Windows and the worker pool elsewhere.

# The CPU direct-syscall ring retains its independent scheduling and queue.
if (CMAKE_SYSTEM_NAME STREQUAL "Linux")
    add_library(ggml-ssd-linux-ring STATIC linux-ring.cpp linux-ring.h)
    target_compile_features(ggml-ssd-linux-ring PUBLIC cxx_std_17)
    target_include_directories(ggml-ssd-linux-ring PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set_target_properties(ggml-ssd-linux-ring PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(ggml-ssd INTERFACE ggml-ssd-linux-ring)
endif()
if (WIN32)
    add_library(ggml-ssd-win-ring STATIC win-ring.cpp win-ring.h ring-common.h ring-types.h)
    target_compile_features(ggml-ssd-win-ring PUBLIC cxx_std_17)
    target_include_directories(ggml-ssd-win-ring PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
    set_target_properties(ggml-ssd-win-ring PROPERTIES POSITION_INDEPENDENT_CODE ON)
    target_link_libraries(ggml-ssd-win-ring PUBLIC ggml-ssd-platform)
    target_link_libraries(ggml-ssd INTERFACE ggml-ssd-win-ring)
endif()
# Persistent reader threads: the macOS ring, and a fallback on every OS.
add_library(ggml-ssd-pool-ring STATIC pool-ring.cpp pool-ring.h ring-common.h ring-types.h)
target_compile_features(ggml-ssd-pool-ring PUBLIC cxx_std_17)
target_include_directories(ggml-ssd-pool-ring PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/..")
set_target_properties(ggml-ssd-pool-ring PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_link_libraries(ggml-ssd-pool-ring PUBLIC ggml-ssd-platform)
target_link_libraries(ggml-ssd INTERFACE ggml-ssd-pool-ring)
