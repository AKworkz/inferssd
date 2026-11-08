enable_language(CUDA)
add_executable(inferssd-memprobe ../memprobe.cu)
target_link_libraries(inferssd-memprobe PRIVATE CUDA::cudart)
install(TARGETS inferssd-memprobe RUNTIME)
