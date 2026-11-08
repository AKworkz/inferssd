#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Check a package from scripts/package-macos.sh as a Mac without developer tools meets it: every
# binary depends only on macOS and the package and is signed, nothing calls a developer tool, and
# scripts/install.sh installs it from its package list, after which the engine answers and
# measures this machine.
#
# Usage: scripts/check-package-macos.sh PACKAGE.tar.gz   (its .json beside it)
set -euo pipefail

archive=$(cd "$(dirname "$1")" && pwd -P)/$(basename "$1")
listing=${archive%.tar.gz}.json
root=$(cd -P "$(dirname "$0")/.." && pwd)
fail() { printf 'check-package-macos: %s\n' "$*" >&2; exit 1; }
[ -f "$archive" ] && [ -f "$listing" ] || fail "need $archive and $listing"
field() { /usr/bin/plutil -extract "$1" raw -o - "$listing"; }
name=$(field name)
minimum=$(field minimum_macos)

work=$(mktemp -d "${TMPDIR:-/tmp}/inferssd-check.XXXXXXXX")
trap 'rm -rf -- "$work"' EXIT
# A path with a space in it, unlike the build's.
place="$work/unpacked here"
mkdir -p "$place"
tar -xzf "$archive" -C "$place"
package="$place/$name"
[ -d "$package" ] || fail "the archive does not hold $name/"
echo "$name: $(du -h "$archive" | cut -f1) archive, $(du -sh "$package" | cut -f1) unpacked"

echo "== Binaries"
count=0
while IFS= read -r -d '' file; do
    file -b "$file" | grep -q Mach-O || continue
    count=$((count + 1))
    # The first line of otool -L names the file, and a library's next line its own install name.
    outside=$(otool -L "$file" | tail -n +2 | awk '{print $1}' | grep -v -x -F "$(otool -D "$file" | tail -n +2)" |
        grep -v -E '^(@rpath/|@loader_path/|@executable_path/|/System/Library/|/usr/lib/)' || true)
    [ -z "$outside" ] || fail "$file loads $outside"
    rpaths=$(otool -l "$file" | awk '/cmd LC_RPATH/ {getline; getline; print $2}' |
        grep -v -E '^@(loader|executable)_path' || true)
    [ -z "$rpaths" ] || fail "$file searches $rpaths"
    codesign --verify "$file" 2>/dev/null || fail "$file is not signed"
    oldest=$(otool -l "$file" | awk '/ minos / {print $2; exit}')
    [ -z "$oldest" ] || [ "$(printf '%s\n%s\n' "$oldest" "$minimum" | sort -V | tail -1)" = "$minimum" ] ||
        fail "$file needs macOS $oldest, past $minimum"
done < <(find "$package" -type f \( -perm -u+x -o -name '*.dylib' -o -name '*.so' \) -print0)
echo "$count Mach-O files load only macOS and the package, find it beside themselves, are signed"
echo "and run on macOS $minimum"

# Every developer tool records its call and fails, as the stubs macOS has without the tools.
mkdir -p "$work/tools"
for tool in git cc c++ clang clang++ gcc python python3 pip pip3 xcrun xcode-select xcodebuild \
    cmake ninja make brew; do
    printf '#!/bin/sh\necho "%s $*" >>"%s/called"\nexit 127\n' "$tool" "$work" >"$work/tools/$tool"
    chmod +x "$work/tools/$tool"
done
export PATH="$work/tools:/usr/bin:/bin:/usr/sbin:/sbin"
export HOME="$work/home" CFFIXED_USER_HOME="$work/home"
export INFERSSD_HOME="$HOME/.local/share/inferssd" INFERSSD_BIN_DIR="$HOME/.local/bin"
unset PYTHONPATH PYTHONHOME VIRTUAL_ENV INFERSSD_RESOURCES INFERSSD_REF INFERSSD_FROM_SOURCE
mkdir -p "$HOME"

echo "== Unpacked"
"$package/bin/inferssd" --version
"$package/bin/inferssd" serve --help >/dev/null
"$package/bin/inferssd-server" --list-devices 2>&1 | grep -A3 '^Available devices'

echo "== Installed by scripts/install.sh from a package list"
/usr/bin/plutil -replace url -string "file://$archive" -o "$work/latest.json" "$listing"
INFERSSD_PACKAGES="file://$work/latest.json" sh "$root/scripts/install.sh" >"$work/install.log" 2>&1 ||
    { cat "$work/install.log"; fail 'scripts/install.sh failed'; }
grep -E '^(Downloading|Installed)' "$work/install.log"
installed=$("$INFERSSD_BIN_DIR/inferssd" --version --json)
echo "$installed"
[ "$(printf '%s' "$installed" | /usr/bin/plutil -extract label raw -o - -)" = "$(field label)" ] ||
    fail 'the installed engine does not report the package label'

echo "== Profile"
"$INFERSSD_BIN_DIR/inferssd" profile --json >"$work/profile.json" 2>"$work/profile.log" ||
    { tail -20 "$work/profile.log"; fail 'profile failed'; }
/usr/bin/plutil -extract machine.cpu raw -o - "$work/profile.json"
/usr/bin/plutil -extract machine.gpus.0.name raw -o - "$work/profile.json" 2>/dev/null || true

if [ -s "$work/called" ]; then
    sort "$work/called" | uniq -c
    fail 'a developer tool was called'
fi
echo "No developer tool was called."
