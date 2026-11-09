# The macOS package

On an Apple silicon Mac with macOS 15 or later, the installer on `main` installs a prebuilt
package of `main` rather than building from source, so the Mac needs no developer tools: no
Xcode or Command Line Tools, Git, Python, CMake or compiler.

```sh
curl -fsSL https://raw.githubusercontent.com/wici-ai/inferssd/main/scripts/install.sh | sh
```

`INFERSSD_FROM_SOURCE=1` or a branch or tag in `INFERSSD_REF` builds from source instead, and
an installer from a release builds that release from source, as before. So does the installer on
`main` while no package is published, when the package list answers 404.

## What it is

One archive, `inferssd-<label>-macos-arm64.tar.gz`, about 30 MB, 80 MB unpacked, named by the
[build label](../README.md#versions) of the commit it was built from:

```text
inferssd-<label>-macos-arm64/
  bin/inferssd               the command, on the package's own Python
  bin/inferssd-server        the engine, its libraries beside it
  bin/inferssd-cpuprobe, bin/inferssd-ssdprobe
  python/                    CPython 3.12 with the framework and Hugging Face support
  templates/                 the models' chat templates
  engine-resources.json      the engine descriptor, with paths relative to it
  activate                   makes this directory the installed engine
  licenses/
```

It runs wherever it is unpacked: every binary loads only macOS's libraries and the package's
own, found beside it. The engine is built for the M1, which every Apple silicon CPU runs, rather
than for the building Mac's CPU as a source build is. On an M5, the machine profile's CPU
measurements of the two builds match at one token, which is what decode runs; in batches they
differ by format, one way or the other. The Python packages are pinned by hash in
`scripts/package-macos-requirements.txt`.

`activate` does what the installer's last step does for a source build: it points
`$INFERSSD_HOME/current` at the package and writes the `inferssd` command into
`$INFERSSD_BIN_DIR`. The installer keeps the installation it replaces and removes older ones.

## How it is made

```sh
scripts/package-macos.sh dist          # on an Apple silicon Mac with Xcode, CMake and Ninja
scripts/check-package-macos.sh dist/inferssd-*-macos-arm64.tar.gz
```

The first needs a clone with `main`'s history, since the build number is its commit count. It
writes the archive and a `.json` that describes it. The second unpacks the archive under a path
with a space, checks every binary's libraries, search paths, signature and oldest macOS, then
installs it with `scripts/install.sh` from a package list of its own, with every developer tool
replaced by a stub that fails, and profiles the machine.

Every numbered `vX.Y.Z` release calls **Package macOS** and **Package Windows** at the same
tag commit. macOS still builds on macOS 15 and checks on macOS 15 and 26; Windows also checks
installation without developer tools. Only after both platforms pass does the common **Release**
job attach both archives, manifests and installers to one numbered GitHub release and update both
normal package feeds with `scripts/publish-package.py`. `APP_VERSIONING_TOKEN` is required.

Each numbered installer pins its package manifest to that same GitHub release. The installer
from `main` keeps following the normal latest package list; explicit `INFERSSD_REF` or
`INFERSSD_FROM_SOURCE` continues to request a source build.

**Package macOS** remains manually runnable on `main` for validation and its existing explicit
**publish** option. The publisher uploads to `https://cdn.wici.ai/apps/inferssd/`, checks the
served bytes before updating the list, and refuses a list older than the one already published.

The CDN keeps a file for a year, so a name never gets other bytes: packing a commit already
uploaded again is refused. If publishing fails, re-run the failed job: it publishes the same
files, and the service skips the ones it holds.

## The package list

The installer reads `https://appapi.wici.ai/versioning/inferssd/config/latest-macos.json`
(`INFERSSD_PACKAGES` names another), with `plutil`, as macOS has it:

```json
{
  "schema": "inferssd-package 1",
  "name": "inferssd-261107.207.84bcc56-macos-arm64",
  "version": "0.1.3",
  "build": 207,
  "revision": "84bcc569ac5f41690ad38cbd820b6113c9accc45",
  "date": "2026-11-07",
  "branch": "main",
  "label": "261107.207.84bcc56",
  "platform": "darwin-arm64",
  "minimum_macos": "15.0",
  "file": "inferssd-261107.207.84bcc56-macos-arm64.tar.gz",
  "size": 27751636,
  "sha256": "…",
  "url": "https://cdn.wici.ai/apps/inferssd/inferssd-261107.207.84bcc56-macos-arm64.tar.gz"
}
```

It downloads `url`, refuses it unless its SHA-256 is `sha256`, unpacks it into a new directory
under `releases` and runs `activate`. A failed download or check leaves the installed engine as
it was. Whatever runs the engine can read the same list to learn of a newer `build`.
