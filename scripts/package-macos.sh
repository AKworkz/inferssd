#!/bin/sh
# SPDX-License-Identifier: Apache-2.0
# Build InferSSD into one archive for Apple silicon: the engine, its probes and a Python of its
# own with the framework, in a directory that runs wherever it is unpacked, on any Apple silicon
# Mac with macOS 15 or later. Making it needs what a source build needs (Xcode or the Command Line
# Tools, CMake, Ninja, and the branch's Git history for the build number); the Mac that unpacks it
# needs none of them. See docs/macos-package.md.
#
# Usage: scripts/package-macos.sh [OUTPUT_DIR]      (default: dist)
#
# OUTPUT_DIR receives inferssd-<label>-macos-arm64.tar.gz, named by the build's label, and beside it
# a .json that describes it for scripts/publish-package.py. INFERSSD_JOBS sets the build parallelism (default: every CPU).
set -eu

# A relocatable CPython (python-build-standalone), pinned by its checksum.
PYTHON_VERSION=3.12.16
PYTHON_RELEASE=20261026
PYTHON_SHA256=353d5f1b4fd2715e7053f12f068435d5bbad16b91e057006b5d01e0925709a7e
PYTHON_URL="https://github.com/astral-sh/python-build-standalone/releases/download/$PYTHON_RELEASE/cpython-$PYTHON_VERSION%2B$PYTHON_RELEASE-aarch64-apple-darwin-install_only_stripped.tar.gz"
# The oldest macOS the package runs on.
MACOS_MINIMUM=15.0

die() { printf 'package-macos: %s\n' "$*" >&2; exit 1; }

[ "$(uname -s):$(uname -m)" = Darwin:arm64 ] || die 'run this on an Apple silicon Mac'
root=$(cd -P "$(dirname "$0")/.." && pwd)
out=${1:-"$root/dist"}
case "$out" in /*) ;; *) out="$PWD/$out" ;; esac
jobs=${INFERSSD_JOBS:-$(sysctl -n hw.ncpu)}
[ "$(git -C "$root" rev-parse --is-shallow-repository)" = false ] ||
    die 'the build number is the commit count: clone with the history (not --depth)'

build="$root/build-package-macos"
# The preset a source build uses, but for every Apple silicon CPU: without -mcpu or -march,
# Apple's clang targets the M1. Each binary finds its libraries beside it (@loader_path), not in
# the build directory; the server is built without OpenSSL, since the framework downloads models;
# and the oldest macOS is fixed.
cmake --preset macos-metal -B "$build" \
    -DGGML_NATIVE=OFF \
    -DCMAKE_OSX_DEPLOYMENT_TARGET="$MACOS_MINIMUM" \
    -DCMAKE_BUILD_WITH_INSTALL_RPATH=ON \
    -DCMAKE_INSTALL_RPATH=@loader_path \
    -DLLAMA_OPENSSL=OFF \
    -DLLAMA_BUILD_TESTS=OFF
cmake --build "$build" --parallel "$jobs" --target inferssd-server inferssd-cpuprobe inferssd-ssdprobe

stage=$(mktemp -d "${TMPDIR:-/tmp}/inferssd-package.XXXXXXXX")
trap 'rm -rf -- "$stage"' EXIT
package="$stage/package"
mkdir -p "$package/bin" "$package/templates" "$package/licenses"
# The libraries keep their version links; each binary names @rpath/lib<name>.0.dylib.
cp -P "$build"/bin/*.dylib "$package/bin/"

curl -fsSL --retry 3 -o "$stage/python.tar.gz" "$PYTHON_URL"
printf '%s  %s\n' "$PYTHON_SHA256" "$stage/python.tar.gz" | shasum -a 256 -c - >/dev/null ||
    die 'the Python download does not match its checksum'
tar -xzf "$stage/python.tar.gz" -C "$package"
python="$package/python/bin/python3"
pip() { "$python" -m pip --disable-pip-version-check --no-cache-dir "$@"; }
ls "$package/python/bin" >"$stage/python-bin"
# Only the pinned wheels, then the framework built offline with the pinned setuptools.
pip install --no-warn-script-location --only-binary=:all: --require-hashes \
    -r "$root/scripts/package-macos-requirements.txt"
pip install --no-warn-script-location --no-deps --no-build-isolation --no-index "$root/framework"
pip uninstall -y -q setuptools pip
# What the framework never uses: Tcl/Tk and IDLE, what only builds against or teaches Python, and
# libpython, which bin/python3 does not load. pip's commands go too: their first line names this
# staging directory, and bin/inferssd replaces them.
for command in "$package"/python/bin/*; do
    grep -qxF "$(basename "$command")" "$stage/python-bin" || rm -f "$command"
done
lib="$package/python/lib"
stdlib="$lib/python${PYTHON_VERSION%.*}"
rm -rf "$package/python/include" "$package/python/share" "$lib/pkgconfig" "$lib"/libpython* \
    "$lib"/libtcl* "$lib"/libtk* "$lib"/tcl* "$lib"/tk* "$lib"/itcl* "$lib"/thread* \
    "$stdlib"/config-* "$stdlib/ensurepip" "$stdlib/idlelib" "$stdlib/tkinter" \
    "$stdlib/turtledemo" "$stdlib/turtle.py" "$stdlib/lib2to3" "$stdlib/pydoc_data" \
    "$stdlib"/lib-dynload/_tkinter.* \
    "$package"/python/bin/*-config "$package"/python/bin/idle* "$package"/python/bin/2to3* \
    "$package"/python/bin/pydoc* "$package"/python/bin/pip*

"$python" "$root/scripts/engine-resources.py" --build "$build" --output "$stage/engine-resources.json"
# The descriptor with every path relative to it and the files it names copied beside it, and the
# package's name.
"$python" -I - "$stage/engine-resources.json" "$package" "$stage/name" <<'PY'
import json
import shutil
import sys
from pathlib import Path

from inferssd.version import label

source, package, named = map(Path, sys.argv[1:])
data = json.loads(source.read_text(encoding="utf-8"))


def place(path, folder):
    path = Path(path)
    target = package / folder / path.name
    if not target.exists():
        shutil.copy2(path, target)
    return folder + "/" + path.name


data["binary"] = place(data["binary"], "bin")
data["templates"] = {name: place(path, "templates") for name, path in data["templates"].items()}
data["probes"] = {name: place(path, "bin") for name, path in data["probes"].items()}
(package / "engine-resources.json").write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
built = label(data.get("build"), data.get("date"), data.get("revision"))
if not built:
    sys.exit("package-macos: the build has no build number, day and commit")
named.write_text(f"inferssd-{built}-macos-arm64\n", encoding="utf-8")
PY
name=$(cat "$stage/name")
cp "$root/LICENSE" "$root/NOTICE" "$package/licenses/"
cp "$root/engine/LICENSE" "$package/licenses/LICENSE-llama.cpp"
cp "$root"/engine/licenses/* "$package/licenses/"

cat >"$package/bin/inferssd" <<'EOF'
#!/bin/sh
# The inferssd command, on this package's own Python, wherever the package is.
root=$(cd -P "$(dirname "$0")/.." && pwd)
export INFERSSD_RESOURCES="${INFERSSD_RESOURCES:-$root/engine-resources.json}"
exec "$root/python/bin/python3" -I -c '
import sys
sys.argv[0] = "inferssd"
from inferssd.cli import main
sys.exit(main())
' "$@"
EOF
cat >"$package/activate" <<'EOF'
#!/bin/sh
# Make this unpacked package the installed engine, as scripts/install.sh does for a source build:
# point INFERSSD_HOME/current at it and write the inferssd command into INFERSSD_BIN_DIR.
set -eu
root=$(cd -P "$(dirname "$0")" && pwd)
home=${INFERSSD_HOME:-"$HOME/.local/share/inferssd"}
bin_dir=${INFERSSD_BIN_DIR:-"$HOME/.local/bin"}
case "$home:$bin_dir" in
    /*:/*) ;;
    *) echo 'inferssd: INFERSSD_HOME and INFERSSD_BIN_DIR must be absolute paths.' >&2; exit 1 ;;
esac
"$root/bin/inferssd" cli --help >/dev/null
"$root/bin/inferssd" serve --help >/dev/null
mkdir -p "$home" "$bin_dir"
exec "$root/python/bin/python3" -I - "$root" "$home" "$bin_dir" <<'PY'
import os
import shlex
import sys
import tempfile
from pathlib import Path

release, home, bindir = map(Path, sys.argv[1:])
wrapper = "#!/bin/sh\n"
wrapper += "export INFERSSD_HOME=${INFERSSD_HOME:-" + shlex.quote(str(home)) + "}\n"
wrapper += "export INFERSSD_RESOURCES=${INFERSSD_RESOURCES:-" + shlex.quote(str(release / "engine-resources.json")) + "}\n"
wrapper += "exec " + shlex.quote(str(release / "bin/inferssd")) + ' "$@"\n'
with tempfile.NamedTemporaryFile(mode="w", dir=bindir, prefix=".inferssd-", delete=False) as stream:
    stream.write(wrapper)
    temporary = Path(stream.name)
try:
    temporary.chmod(0o755)
    link = home / (".current-" + os.urandom(4).hex())
    link.symlink_to(release, target_is_directory=True)
    os.replace(link, home / "current")
    os.replace(temporary, bindir / "inferssd")
finally:
    temporary.unlink(missing_ok=True)
print("Installed: " + str(bindir / "inferssd"))
PY
EOF
chmod 755 "$package/bin/inferssd" "$package/activate"

# The package must answer before it is archived.
"$package/bin/inferssd" --version --json >"$stage/version.json"
"$package/bin/inferssd" serve --help >/dev/null

mv "$package" "$stage/$name"
python="$stage/$name/python/bin/python3"
mkdir -p "$out"
COPYFILE_DISABLE=1 tar -C "$stage" -czf "$out/$name.tar.gz" "$name"
# What scripts/publish-package.py publishes as the package list, less the URL it is given.
"$python" -I - "$stage/version.json" "$out/$name.tar.gz" "$MACOS_MINIMUM" >"$out/$name.json" <<'PY'
import hashlib
import json
import sys
from pathlib import Path

version, archive, minimum = json.loads(Path(sys.argv[1]).read_text()), Path(sys.argv[2]), sys.argv[3]
digest = hashlib.sha256(archive.read_bytes()).hexdigest()
name = archive.name[: -len(".tar.gz")]
fields = ("version", "build", "revision", "date", "branch", "label")
print(json.dumps({
    "schema": "inferssd-package 1",
    "name": name,
    **{key: version[key] for key in fields},
    "platform": "darwin-arm64",
    "minimum_macos": minimum,
    "file": archive.name,
    "size": archive.stat().st_size,
    "sha256": digest,
}, indent=2))
PY
printf 'Packaged %s (%s)\n' "$out/$name.tar.gz" "$(du -h "$out/$name.tar.gz" | cut -f1)"
