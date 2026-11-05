find_package(Threads REQUIRED)

add_library(kimi-vram-tier OBJECT vram-tier.cpp
    generic-compute-adapter.cu group-compute-adapter.cu provider.cpp transport.cpp)
set_target_properties(kimi-vram-tier PROPERTIES POSITION_INDEPENDENT_CODE ON
    CUDA_STANDARD 17 CUDA_STANDARD_REQUIRED ON)
target_include_directories(kimi-vram-tier PRIVATE ../../../include ../..)
target_compile_features(kimi-vram-tier PRIVATE cxx_std_17)
target_compile_options(kimi-vram-tier PRIVATE
    $<$<COMPILE_LANGUAGE:CXX>:-fno-fast-math;-ffp-contract=off>)
target_link_libraries(kimi-vram-tier PRIVATE ggml-base ggml-ssd CUDA::cudart Threads::Threads)
target_link_libraries(ggml-cuda PRIVATE kimi-vram-tier ggml-ssd CUDA::cudart)
target_compile_definitions(ggml-cuda PRIVATE GGML_SSD_CUDA_EXECUTOR)
