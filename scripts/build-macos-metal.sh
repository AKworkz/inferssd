#!/usr/bin/env bash
set -euo pipefail

repo_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PATH="/opt/homebrew/bin:$PATH"
build_dir="${SSD_LLAMA_BUILD_DIR:-$repo_dir/build-mac}"

if [[ "$(uname -s)" != Darwin || "$(uname -m)" != arm64 ]]; then
    echo "This build profile requires an Apple Silicon Mac." >&2
    exit 1
fi

command -v ninja >/dev/null || { echo "Ninja is required (brew install ninja)." >&2; exit 1; }
cmake -S "$repo_dir" -B "$build_dir" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release \
    -DGGML_CUDA=OFF -DGGML_METAL=ON -DGGML_ACCELERATE=ON \
    -DGGML_OPENMP=OFF -DLLAMA_BUILD_TESTS=ON \
    -DLLAMA_BUILD_WEBUI=OFF -DLLAMA_BUILD_EXAMPLES=OFF
cmake --build "$build_dir" \
    --target llama-cli inferssd-server test-chat test-backend-ops test-ssd-cpu-sink test-ssd-cpu-prefill test-llama-mmap-dontneed -j "${SSD_LLAMA_BUILD_JOBS:-8}"
