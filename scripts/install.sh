#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Usage: curl -fsSL https://github.com/wici-ai/inferssd/releases/latest/download/install.sh | sh
set -eu

# The release workflow replaces this with the tag it publishes.
DEFAULT_REF=main
# The prebuilt package of main for Apple silicon, as .github/workflows/package-macos.yml
# publishes it (docs/macos-package.md).
PACKAGES=https://appapi.wici.ai/versioning/inferssd/config/latest-macos.json

die() { printf 'inferssd: %s\n' "$*" >&2; exit 1; }
need() { command -v "$1" >/dev/null 2>&1 || die "Install $1 first. See docs/quickstart.md."; }

# The package list into $listing, or false when none is published yet (HTTP 404), as before the
# first package: the installer then builds from source.
package_list() {
    listing=$(mktemp "$install_home/releases/.latest.XXXXXXXX")
    packages=${INFERSSD_PACKAGES:-$PACKAGES}
    if status=$(curl -fsSL --retry 3 -o "$listing" -w '%{http_code}' "$packages"); then return 0; fi
    rm -f "$listing"
    [ "$status" != 404 ] || return 1
    die "Cannot read the package list $packages. Set INFERSSD_FROM_SOURCE=1 to build from source."
}

# The prebuilt package the package list names, with nothing but what macOS has.
install_package() {
    release=$(mktemp -d "$install_home/releases/package.XXXXXXXX")
    mv "$listing" "$release/latest.json"
    schema=$(field schema)
    [ "$schema" = 'inferssd-package 1' ] ||
        die 'This installer does not read the package list; download the installer again.'
    platform=$(field platform)
    [ "$platform" = darwin-arm64 ] || die "The package list is for $platform."
    name=$(field name)
    label=$(field label)
    url=$(field url)
    sha256=$(field sha256)
    size=$(field size)
    minimum=$(field minimum_macos)
    os_version=$(sw_vers -productVersion)
    [ "${os_version%%.*}" -ge "${minimum%%.*}" ] ||
        die "InferSSD $label needs macOS $minimum or later and this Mac has $os_version. Update macOS, or set INFERSSD_FROM_SOURCE=1 to build from source."
    printf 'Downloading InferSSD %s (%s MB).\n' "$label" "$(((size + 999999) / 1000000))"
    curl -fL --retry 3 -o "$release/package.tar.gz" "$url"
    actual=$(shasum -a 256 "$release/package.tar.gz")
    [ "${actual%% *}" = "$sha256" ] || die 'The download does not match the package list.'
    printf 'Unpacking InferSSD %s.\n' "$label"
    tar -xzf "$release/package.tar.gz" -C "$release"
    rm -f "$release/package.tar.gz" "$release/latest.json"
    package="$release/$name"
    "$package/bin/inferssd" cli --help >/dev/null
    "$package/bin/inferssd" serve --help >/dev/null
    complete=yes
    INFERSSD_HOME="$install_home" INFERSSD_BIN_DIR="$bin_dir" "$package/activate" >/dev/null
    py="$package/python/bin/python3"
}

# One value from the package list.
field() {
    plutil -extract "$1" raw -o - "$release/latest.json" 2>/dev/null ||
        die "The package list has no $1."
}

# A build of INFERSSD_REF for this machine's GPUs, in a private Python environment.
build_source() {
    # Without these tools, git, cc and python3 are macOS stubs that open an install dialog.
    if [ "$system" = darwin ]; then
        xcode-select -p >/dev/null 2>&1 ||
            die 'Install the Xcode Command Line Tools first: xcode-select --install'
    fi
    need git
    need python3
    need cc
    need c++
    python3 -c 'import sys; sys.exit(sys.version_info < (3, 9))' || die 'Python >=3.9 is required.'
    jobs=${INFERSSD_JOBS:-4}
    case "$jobs" in ''|*[!0-9]*|0) die 'INFERSSD_JOBS must be a positive integer.' ;; esac
    release=$(mktemp -d "$install_home/releases/source.XXXXXXXX")
    # Every commit but no file contents, a few megabytes more than one commit: the engine's
    # build number is the branch's commit count and its day the commit's, read from the clone.
    git clone --filter=blob:none --single-branch --branch "${INFERSSD_REF:-$DEFAULT_REF}" \
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
}

# Keep the release now installed and the one it replaced; remove the older ones. Paths are
# compared resolved, since a link may name a release through another path to it.
prune() {
    now=$(cd -P "$install_home/current" 2>/dev/null && pwd) || return 0
    before=
    if [ -n "$previous" ]; then before=$(cd -P "$previous" 2>/dev/null && pwd) || before=; fi
    for old in "$install_home"/releases/*; do
        [ -d "$old" ] || continue
        real=$(cd -P "$old" && pwd)
        case "$now/" in "$real"/*) continue ;; esac
        case "$before/" in "$real"/*) continue ;; esac
        rm -rf -- "$old"
    done
}

main() (
    if [ "${1:-}" = --help ]; then
        cat <<'EOF'
Install InferSSD for Apple silicon macOS or Linux x86_64/ARM64.
This installer from main puts the prebuilt package of main, once one is published, on a Mac
with macOS 15 or later, which then needs no developer tools. Otherwise it builds from source,
which requires Git, Python >=3.9 with venv and a C/C++ compiler; on macOS, the Xcode Command
Line Tools.
On macOS the engine runs on the Apple GPU. Linux requires systemd and cgroup v2 at runtime.
A Linux build is made for the machine's GPUs: with an NVIDIA GPU it uses CUDA and needs the
CUDA toolkit; with an AMD or Intel GPU it uses Vulkan when the Vulkan SDK (glslc) is
installed; otherwise it runs on the CPU.
CMake, Ninja and the Python framework are installed in a private environment.

Environment:
  INFERSSD_HOME         Installation and model root (default: ~/.local/share/inferssd)
  INFERSSD_BIN_DIR      Command directory (default: ~/.local/bin)
  INFERSSD_FROM_SOURCE  Build from source on a Mac as well (any value)
  INFERSSD_PACKAGES     Package list a Mac installs from (default: main's)
  INFERSSD_REF          Branch or tag to build from source (default: this installer's release)
  INFERSSD_JOBS         Build parallelism (default: 4)
  CUDACXX              CUDA compiler, if nvcc is not on PATH

No sudo is used. A previous installation remains usable if installing fails; the installer
keeps the installation it replaces and removes older ones.
EOF
        exit 0
    fi
    [ "$#" -eq 0 ] || die 'Unknown argument; use --help.'
    case "$(uname -s):$(uname -m)" in
        Darwin:arm64) system=darwin ;;
        Linux:x86_64|Linux:aarch64|Linux:arm64) system=linux ;;
        *) die 'Supported platforms: Apple Silicon macOS and Linux x86_64/ARM64.' ;;
    esac
    install_home=${INFERSSD_HOME:-"$HOME/.local/share/inferssd"}
    bin_dir=${INFERSSD_BIN_DIR:-"$HOME/.local/bin"}
    case "$install_home" in /*) ;; *) die 'INFERSSD_HOME must be an absolute path.' ;; esac
    case "$bin_dir" in /*) ;; *) die 'INFERSSD_BIN_DIR must be an absolute path.' ;; esac
    mkdir -p "$install_home/releases" "$bin_dir"
    previous=$(readlink "$install_home/current" 2>/dev/null || true)
    release=
    complete=no
    trap 'if [ "$complete" != yes ] && [ -n "$release" ]; then rm -rf -- "$release"; fi' EXIT
    trap 'exit 130' INT
    trap 'exit 143' TERM
    # A release's installer builds its tag; a branch or tag asked for is built from source too.
    if [ "$system" = darwin ] && [ "$DEFAULT_REF" = main ] && [ -z "${INFERSSD_REF:-}" ] &&
        [ -z "${INFERSSD_FROM_SOURCE:-}" ]; then
        if package_list; then
            install_package
        else
            printf 'inferssd: no package is published yet; building from source.\n' >&2
            build_source
        fi
    else
        build_source
    fi
    prune
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
