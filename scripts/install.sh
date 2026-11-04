#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Usage: curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
set -eu

# The release workflow replaces this with the tag it publishes.
DEFAULT_REF=main

die() { printf 'inferssd: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Install $1 first. See docs/quickstart.md."; }

main() (
    if [ "${1:-}" = --help ]; then
        cat <<'EOF'
Install InferSSD from source for Apple Silicon macOS or Linux x86_64/ARM64.
Requires Git, Python >=3.9 with venv and a C/C++ compiler.
macOS requires the Xcode Command Line Tools; the engine runs on the Apple GPU.
Linux requires liburing development headers, and systemd/cgroup v2 at runtime.
The engine is built for the machine's GPUs: with an NVIDIA GPU it uses CUDA and
needs the CUDA toolkit; with an AMD or Intel GPU it uses Vulkan when the Vulkan
SDK (glslc) is installed; otherwise it runs on the CPU.
CMake, Ninja and the Python framework are installed in a private environment.

Environment:
  INFERSSD_HOME         Installation and model root (default: ~/.local/share/inferssd)
  INFERSSD_BIN_DIR      Command directory (default: ~/.local/bin)
  INFERSSD_REF          Branch or tag to build (default: this installer's release)
  INFERSSD_JOBS         Build parallelism (default: 4)
  CUDACXX              CUDA compiler, if nvcc is not on PATH

No sudo is used. A previous installation remains usable if building fails.
EOF
        exit 0
    fi
    [ "$#" -eq 0 ] || die 'Unknown argument; use --help.'
    # Without these tools, git, cc and python3 are macOS stubs that open an install dialog.
    if [ "$(uname -s)" = Darwin ]; then
        xcode-select -p >/dev/null 2>&1 ||
            die 'Install the Xcode Command Line Tools first: xcode-select --install'
    fi
    need git
    need python3
    need cc
    need c++
    python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || die 'Python >=3.9 is required.'
    case "$(uname -s):$(uname -m)" in
        Darwin:arm64) system=darwin ;;
        Linux:x86_64|Linux:aarch64|Linux:arm64)
            system=linux
            printf '#include <liburing.h>\n' | cc -E - >/dev/null 2>&1 ||
                die 'Install liburing development headers (e.g. liburing-dev).'
            ;;
        *) die 'Supported platforms: Apple Silicon macOS and Linux x86_64/ARM64.' ;;
    esac
    install_home=${INFERSSD_HOME:-"$HOME/.local/share/inferssd"}
    bin_dir=${INFERSSD_BIN_DIR:-"$HOME/.local/bin"}
    case "$install_home" in /*) ;; *) die 'INFERSSD_HOME must be an absolute path.' ;; esac
    case "$bin_dir" in /*) ;; *) die 'INFERSSD_BIN_DIR must be an absolute path.' ;; esac
    jobs=${INFERSSD_JOBS:-4}
    case "$jobs" in ''|*[!0-9]*|0) die 'INFERSSD_JOBS must be a positive integer.' ;; esac
    mkdir -p "$install_home/releases" "$bin_dir"
    release=$(mktemp -d "$install_home/releases/source.XXXXXXXX")
    complete=no
    trap 'if [ "$complete" != yes ]; then rm -rf -- "$release"; fi' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    git clone --depth 1 --branch "${INFERSSD_REF:-$DEFAULT_REF}" \
        https://github.com/wici-ai/inferssd.git "$release/source"
    # The GPU backend for this machine's GPUs, by the table the launch reads.
    backend=$(PYTHONPATH="$release/source/framework/src" python3 -m inferssd.machine.backend)
    vulkan=
    case "$system:$backend" in
        darwin:*) preset=macos-metal ;;
        linux:CUDA)
            preset=linux-cuda
            if [ -z "${CUDACXX:-}" ]; then
                if command -v nvcc >/dev/null 2>&1; then
                    CUDACXX=$(command -v nvcc)
                elif [ -x /usr/local/cuda/bin/nvcc ]; then
                    CUDACXX=/usr/local/cuda/bin/nvcc
                else
                    die 'An NVIDIA GPU was found: install the CUDA toolkit or set CUDACXX to nvcc.'
                fi
            fi
            export CUDACXX
            ;;
        linux:Vulkan)
            preset=linux-cpu
            if command -v glslc >/dev/null 2>&1; then
                vulkan=-DGGML_VULKAN=ON
            else
                printf 'inferssd: an AMD or Intel GPU was found; install the Vulkan SDK (glslc) to use it.\n' >&2
            fi
            ;;
        linux:*) preset=linux-cpu ;;
    esac
    printf 'Building InferSSD (%s%s). This may take several minutes.\n' "$preset" "${vulkan:+, Vulkan}"
    python3 -m venv "$release/venv"
    py="$release/venv/bin/python"
    "$py" -m pip install --upgrade pip
    "$py" -m pip install 'cmake>=3.24,<5' ninja "$release/source/framework[hf]"
    PATH="$release/venv/bin:$PATH"
    export PATH
    cd "$release/source"
    cmake --preset "$preset" $vulkan
    if [ "$preset" = linux-cuda ]; then
        cmake --build "build-$preset" --parallel "$jobs" \
            --target inferssd-server inferssd-cpuprobe inferssd-ssdprobe inferssd-memprobe
    else
        cmake --build "build-$preset" --parallel "$jobs" \
            --target inferssd-server inferssd-cpuprobe inferssd-ssdprobe
    fi
    "$py" scripts/engine-resources.py --build "build-$preset" \
        --output "$release/engine-resources.json"
    "$release/venv/bin/inferssd" cli --help >/dev/null
    "$release/venv/bin/inferssd" serve --help >/dev/null
    # Everything below may publish a pointer to this fully verified release.
    # Retain it even if publishing the command fails halfway through.
    complete=yes
    # Quote paths as shell words, and replace only after the build/schema checks pass.
    "$py" - "$release" "$install_home" "$bin_dir" <<'PY'
import os
import shlex
import sys
import tempfile
from pathlib import Path

release, home, bindir = map(Path, sys.argv[1:])
wrapper = "#!/bin/sh\n"
wrapper += "export INFERSSD_HOME=${INFERSSD_HOME:-" + shlex.quote(str(home)) + "}\n"
wrapper += "export INFERSSD_RESOURCES=${INFERSSD_RESOURCES:-" + shlex.quote(str(release / "engine-resources.json")) + "}\n"
wrapper += "exec " + shlex.quote(str(release / "venv/bin/inferssd")) + ' "$@"\n'
with tempfile.NamedTemporaryFile(mode="w", dir=bindir, prefix=".inferssd-", delete=False) as stream:
    stream.write(wrapper)
    temporary = Path(stream.name)
try:
    temporary.chmod(0o755)
    link = release / "current-link"
    link.symlink_to(release, target_is_directory=True)
    os.replace(link, home / "current")
    os.replace(temporary, bindir / "inferssd")
finally:
    temporary.unlink(missing_ok=True)
PY
    printf '\nInstalled: %s/inferssd\n' "$bin_dir"
    case ":$PATH:" in
        *":$bin_dir:"*) ;;
        *) "$py" - "$bin_dir" <<'PY'
import shlex, sys
print('Add this directory to your shell PATH (run now, and add to your shell profile):')
print('  export PATH=' + shlex.quote(sys.argv[1]) + ':"$PATH"')
PY
        ;;
    esac
    cat <<'EOF'

Download and chat (first launch downloads hundreds of GB):
  inferssd cli -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

Start the OpenAI-compatible API:
  inferssd serve -hf mrfakename/DeepSeek-V4.1-Flash-1017-GGUF

Muse:
  inferssd serve -hf unsloth/Muse-Spark-Open-816B-A42B-GGUF
EOF
)

main "$@"
