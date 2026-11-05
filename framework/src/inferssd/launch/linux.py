"""Pure Linux launch policy; every external observation is explicit."""

from __future__ import annotations

import posixpath
import re
from collections.abc import Mapping
from typing import cast

from inferssd.launch.types import AwaitFact, Fact, Finished, LaunchInput, Reply, Resolution, State

LAUNCHER_SETTINGS = frozenset(
    """CONTEXT UBATCH CTX_CHECKPOINTS THREADS PROMPT_CACHE_DIR
PROMPT_CACHE_MIB MEMORY_HEADROOM_MIB MEMORY_RESERVE_MIB MEMORY_MAX VRAM_RESERVE_MIB
VRAM_ELASTIC_SAFETY_MIB CPUSET SERVER_BIN PRELOAD_LIB LLAMA_SERVER_HOST LLAMA_SERVER_PORT
EXPERT_ALIAS EXPERT_PACK EXPERT_INDEX DENSE_HOMES ARGS""".split()
)
EMPTY_IS_NONE = frozenset(
    "PROMPT_CACHE_DIR MEMORY_MAX EXPERT_ALIAS EXPERT_PACK DENSE_HOMES CPUSET".split()
)


def read_values(text: str, path: str, state: State) -> dict[str, str]:
    values: dict[str, str] = {}
    for number, line in enumerate(text.split("\n"), 1):
        if re.fullmatch(r"\s*(#.*)?", line):
            continue
        match = re.fullmatch(r"([A-Za-z_][A-Za-z0-9_]*)=(.*)", line)
        if not match:
            state.error(
                (
                    f"{path}:{number}: not a KEY=VALUE line (no 'export', no blanks "
                    f"around '='): {line}"
                ),
                2,
                append_status=True,
            )
        assert match is not None
        name, value = match.groups()
        value = re.split(r"\s#", value, maxsplit=1)[0].rstrip()
        if len(value) >= 2 and value[0] == value[-1] == "'":
            value = value[1:-1]
        values[name] = value
    return values


def engine_name(name: str) -> bool:
    return name.startswith(("GGML_", "LLAMA_"))


def cpu_count(value: str) -> str:
    """Preserve the launcher's awk range/duplicate counting for CPU lists."""
    seen: set[int] = set()

    def number(text: str) -> int:
        match = re.match(r"\s*([+-]?\d+)", text)
        return int(match.group(1)) if match else 0

    for part in value.split(","):
        pieces = part.split("-")
        start = number(pieces[0])
        stop = number(pieces[1]) if len(pieces) == 2 else start
        seen.update(range(start, stop + 1))
    return str(len(seen)) if seen else ""


class LinuxState(State):
    def __init__(self, inputs: LaunchInput, facts: Mapping[Fact, Reply], *, execute: bool = False):
        super().__init__(inputs, facts)
        self.execute = execute
        self.settings: dict[str, str] = {}
        self.variables = dict(self.env)
        self.set_names: set[str] = set()

    def given_by(self, name: str) -> str:
        none = name in EMPTY_IS_NONE
        if self.variables.get(name) or (none and name in self.variables):
            return "environment"
        if self.settings.get(name) or (none and name in self.settings):
            return "settings"
        return "default"

    def setting(self, name: str, default: str) -> str:
        source = self.given_by(name)
        if source == "settings":
            self.variables[name] = self.settings[name]
        elif source == "default":
            self.variables[name] = default
        return self.variables[name]

    def typed_number(self, name: str, value: str, *, leading_zero: bool = False) -> int:
        """Accept decimal data, not Bash arithmetic or signed overflow.

        Leading zeros are retained only for pass-through CTX_CHECKPOINTS;
        the separately validated prefill threshold also uses decimal grammar.
        """
        syntax = r"[0-9]+" if leading_zero else r"(?:0|[1-9][0-9]*)"
        significant = value.lstrip("0") or "0"
        if (
            not re.fullmatch(syntax, value)
            or len(significant) > 19
            or int(significant) > 9223372036854775807
        ):
            original = dict(self.inputs.environment)
            if original.get(name):
                source = "environment"
            elif self.settings.get(name):
                source = "SETTINGS " + self.variables.get("SETTINGS", "")
            else:
                source = "profile default"
            self.error(
                f"{name}={value!r} from {source}: expected unsigned decimal integer data "
                "in 0..9223372036854775807" + ("" if leading_zero else " without leading zeros"),
                2,
            )
        return int(significant)

    def engine(self, name: str, default: str) -> str:
        source = self.given_by(name)
        return (
            self.variables[name]
            if source == "environment"
            else self.settings[name]
            if source == "settings"
            else default
        )

    def add(self, name: str, value: str) -> None:
        if name in self.set_names:
            self.error(f"{name} is set twice", 3, append_status=True)
        self.set_names.add(name)
        self.assigned.append((name, value))

    def receipt(self, name: str, value: str, source: str, input_: str = "") -> None:
        self.emit(
            f"[ssd_setting] component=launcher name={name} value={value} source={source}"
            + (f" input={input_}" if input_ else "")
            + "\n"
        )

    def put(self, name: str, value: str, source: str, input_: str = "") -> None:
        where = self.given_by(name)
        if where == "environment":
            value, source, input_ = self.variables[name], "override", "environment"
        elif where == "settings":
            value, source, input_ = self.settings[name], "settings", self.variables["SETTINGS"]
        self.add(name, value)
        self.receipt(name, value, source, input_)

    def derived(self, name: str, value: str, input_: str) -> None:
        if engine_name(name) and self.given_by(name) != "default":
            given = self.engine(name, "")
            if given != value:
                self.error(
                    (
                        f"{name}={given} contradicts {name}={value}, which the launcher "
                        f"derives from {input_}: set {input_} instead"
                    ),
                    2,
                    append_status=True,
                )
        self.add(name, value)
        self.receipt(name, value, "derived", input_)

    def refuse(self, text: str) -> None:
        if self.execute:
            self.error(text)
        self.emit("dry run: the launch would stop here: " + text + "\n", "stderr")

    def profile(self, pairs: tuple[tuple[str, str], ...]) -> None:
        for name, value in pairs:
            self.put(name, value, "profile")


def _resolve(s: LinuxState) -> None:
    env = s.variables
    repo = s.inputs.repo
    conf_path = env.get("MODEL_CONF", "")
    if not conf_path:
        s.error("Set MODEL_CONF to a model conf (tools/adaptive/models/*.conf)")
    if not s.value("is_file", conf_path):
        s.error(f"Missing model conf: {conf_path}")
    conf = read_values(s.text("read_text", conf_path), conf_path, s)
    settings_path = env.get("SETTINGS", "")
    if settings_path:
        if not s.value("is_file", settings_path):
            s.error(f"Missing SETTINGS file: {settings_path}")
        s.settings = read_values(s.text("read_text", settings_path), settings_path, s)
    for name in s.settings:
        if not engine_name(name) and name not in LAUNCHER_SETTINGS:
            s.error(
                (
                    f"{settings_path}: {name} is neither a launcher setting nor an "
                    f"engine variable (GGML_*, LLAMA_*)"
                ),
                2,
                append_status=True,
            )
    model_name = conf.get("NAME") or posixpath.basename(conf_path).removesuffix(".conf")
    model = env.get("MODEL", "")
    if not model:
        directory = env.get("MODEL_DIR", "")
        pattern = conf.get("MODEL_GLOB", "")
        if not directory:
            s.error(
                f"Set MODEL (the first GGUF shard) or MODEL_DIR (where to look for "
                f"{pattern or 'MODEL_GLOB'})"
            )
        if not pattern:
            s.error(f"{conf_path} has no MODEL_GLOB; set MODEL")
        found = cast(tuple[str, ...], s.value("model_matches", directory, pattern))
        if len(found) != 1:
            s.error(
                f"MODEL_GLOB {pattern} matches {len(found)} files under {directory}: "
                f"{' '.join(found) or 'none'}; set MODEL"
            )
        model = found[0]
    if env.get("LAUNCH_RESOLVE", "0") == "1":
        # Structured early result avoids introducing a shell-eval public protocol.
        s.argv = [model]
        s.assigned = [("CONF_" + key, value) for key, value in conf.items()]
        return
    server = s.setting("SERVER_BIN", repo + "/build-cuda130/bin/llama-server")
    if not s.value("is_executable", server):
        s.error(f"Missing llama-server executable: {server} (set SERVER_BIN)")
    binary_dir = s.text("absolute_directory", posixpath.dirname(server))
    preload_lib = s.setting("PRELOAD_LIB", binary_dir + "/libkimi-no-mmap-prefetch.so")
    host = s.setting("LLAMA_SERVER_HOST", "127.0.0.1")
    port = s.setting("LLAMA_SERVER_PORT", "18181")
    s.typed_number("LLAMA_SERVER_PORT", port)
    if not re.fullmatch(r"[0-9]+", port) or not 1 <= int(port) <= 65535:
        s.error("LLAMA_SERVER_PORT must be in 1..65535", 2, append_status=True)
    cpuset = s.setting("CPUSET", "")
    cache_root = env.get("XDG_CACHE_HOME") or env.get("HOME", "") + "/.cache"
    prompt_cache = s.setting("PROMPT_CACHE_DIR", cache_root + "/ssd-llama/" + model_name)
    if prompt_cache:
        fs = s.text("filesystem", prompt_cache)
        if fs in ("overlayfs", "tmpfs"):
            s.emit(
                (
                    f"warning: PROMPT_CACHE_DIR {prompt_cache} is on {fs}; put it on the "
                    f"drive that holds the model\n"
                ),
                "stderr",
            )
    reserve = s.setting("VRAM_RESERVE_MIB", "0")
    s.typed_number("VRAM_RESERVE_MIB", reserve)
    if not re.fullmatch(r"[0-9]+", reserve) or int(reserve) > 8192:
        s.error("VRAM_RESERVE_MIB must be in 0..8192", 2, append_status=True)
    memory_reserve = s.setting("MEMORY_RESERVE_MIB", "0")
    memory_max = s.setting("MEMORY_MAX", "")
    alias = s.setting("EXPERT_ALIAS", "")
    pack = s.setting("EXPERT_PACK", conf.get("EXPERT_PACK", ""))
    if alias:
        pack = ""
    if not alias and not pack:
        index = s.setting(
            "EXPERT_INDEX",
            cache_root + "/ssd-llama/experts/" + posixpath.basename(model).removesuffix(".gguf"),
        )
        reply = s.metadata("index", model, index)
        if reply.code:
            s.error(f"Cannot index the experts of {model} into {index}")
        alias = index + "/experts.tsv"
    for asset in (model, alias or pack, preload_lib):
        if not s.value("is_file", asset):
            s.error(f"Missing required asset: {asset}")
    reply = s.metadata("source_facts", model, alias)
    if reply.code:
        s.error(f"Cannot read the GGUF header of {model}")
    facts = cast(Mapping[str, str], reply.value)
    expert_types, alias_unaligned = facts["expert_types"], int(facts["alias_unaligned"])
    homes = s.setting("DENSE_HOMES", "")
    dense = dict.fromkeys(
        ("dense_drive", "dense_drive_bytes", "dense_ram", "dense_ram_bytes", "dense_stage_bytes"),
        "0",
    )
    dense_alias = ""
    dense_args: list[str] = []
    if homes:
        if not s.value("is_file", homes):
            s.error(f"Missing DENSE_HOMES file: {homes}")
        reply = s.metadata("dense", model, homes, homes + ".alias.tsv")
        if reply.code:
            s.error(f"DENSE_HOMES {homes} cannot be served (above)", 2, append_status=True)
        dense.update(cast(Mapping[str, str], reply.value))
        if int(dense["dense_drive"]) > 0:
            dense_alias = homes + ".alias.tsv"
        groups: dict[str, list[str]] = {}
        for line in s.text("read_text", homes).splitlines():
            parts = line.split("#", 1)[0].rstrip().split("\t")
            if len(parts) >= 2 and parts[0] and parts[1]:
                groups.setdefault(parts[1], []).append(parts[0].replace(".", "\\."))
        for where, backend in (("drive", "CPU"), ("ram", "CUDA_Host")):
            if groups.get(where):
                dense_args += ["-ot", "^(" + "|".join(groups[where]) + ")$=" + backend]
    mxfp4 = "MXFP4" in expert_types.split()
    libraries = [preload_lib]
    if mxfp4:
        libraries.append(binary_dir + "/libkimi-mxfp4-parity.so")
    libraries.append(binary_dir + "/libkimi-vram-tier.so")
    for library in libraries[1:]:
        if not s.value("is_file", library):
            s.error(f"Missing required asset: {library}")
        base = binary_dir + "/libggml-base.so"
        if s.value("is_file", base) and s.value("older", library, base):
            s.emit(
                (
                    f"warning: {library} is older than the libggml-base.so it was linked "
                    f"against: rebuild it (tools/adaptive/build.sh)\n"
                ),
                "stderr",
            )
    gpu_available = bool(s.value("gpu_available"))
    gpu_mib = s.text("gpu_memory") if gpu_available else ""
    unified = bool(gpu_mib and not re.fullmatch(r"\s*[0-9]+\s*", gpu_mib))
    threads = s.setting("THREADS", cpu_count(cpuset) if cpuset else s.text("default_threads"))
    context = s.setting("CONTEXT", "131072")
    ubatch = s.setting("UBATCH", "8192")
    checkpoints = s.setting("CTX_CHECKPOINTS", "8")
    prompt_mib = s.setting("PROMPT_CACHE_MIB", "16384")
    headroom = s.setting("MEMORY_HEADROOM_MIB", "512")
    low = max(int(reserve), 16)
    high = int(reserve) + 64
    margin = s.setting("VRAM_ELASTIC_SAFETY_MIB", str(high))
    s.put("GGML_KIMI_CPU_SINK", "1", "profile")
    threshold = s.engine("GGML_KIMI_CPU_PREFILL_MAX_TOKENS", "")
    if threshold:
        if s.given_by("GGML_KIMI_CPU_PREFILL") != "default":
            s.error(
                (
                    "GGML_KIMI_CPU_PREFILL and GGML_KIMI_CPU_PREFILL_MAX_TOKENS are "
                    "both given: the threshold is one input, set MAX_TOKENS"
                ),
                2,
                append_status=True,
            )
        if not re.fullmatch(r"[0-9]{1,10}", threshold) or int(threshold) > 4294967295:
            s.error(
                (
                    "GGML_KIMI_CPU_PREFILL_MAX_TOKENS must be a number of tokens (at "
                    "most 4294967295, every ubatch)"
                ),
                2,
                append_status=True,
            )
        s.put("GGML_KIMI_CPU_PREFILL_MAX_TOKENS", threshold, "settings")
        max_tokens = int(threshold)
    else:
        prefill = s.engine("GGML_KIMI_CPU_PREFILL", "1")
        if prefill not in ("0", "1"):
            s.error(
                "GGML_KIMI_CPU_PREFILL must be 0 (the GPU graph) or 1 (the CPU sink)",
                2,
                append_status=True,
            )
        s.put("GGML_KIMI_CPU_PREFILL", "1", "profile")
        max_tokens = 4294967295 if prefill == "1" else 0
    # Ordered, unchanged launch profile. Engine-side folding is a separate contract change.
    s.profile(
        (
            ("GGML_MOE_IO_REQUIRE_DIRECT", "1"),
            ("GGML_MOE_IO_BACKEND", "iouring"),
            ("GGML_MOE_IO_DEPTH", "64"),
            ("GGML_KIMI_CPU_SINK_RING_EXPERTS", "8"),
            ("GGML_KIMI_CPU_SINK_READY_FIRST", "1"),
            ("GGML_KIMI_CPU_SINK_SPLIT_CV", "1"),
            ("GGML_KIMI_CPU_SINK_DECODE_RESERVE", "1"),
            ("GGML_KIMI_CPU_SINK_CACHE_RESERVE_BYTES", "536870912"),
            ("GGML_KIMI_CPU_SINK_CACHE_PARTIAL_RECLAIM", "1"),
            ("GGML_KIMI_CPU_SINK_CACHE_POLICY", "count_decay2"),
            ("GGML_KIMI_CPU_SINK_CACHE_ALLOCATION", "uniform"),
            ("GGML_KIMI_CPU_SINK_CACHE_KEEP_ACROSS_REQUESTS", "1"),
            ("GGML_KIMI_CPU_SINK_CACHE_RECLAIMABLE_FILE", "1"),
            ("GGML_KIMI_CPU_SINK_CACHE_PREFILL_READ", "1"),
        )
    )
    if not unified:
        s.put("GGML_KIMI_CPU_SINK_VRAM_BYTES", "21000000000", "profile")
    s.profile(
        (
            ("GGML_KIMI_CPU_SINK_VRAM_POLICY", "count_decay2"),
            ("GGML_KIMI_CPU_SINK_VRAM_MIN_COUNT", "2.5"),
            ("GGML_KIMI_CPU_SINK_VRAM_DECAY_HALF_LIFE", "16"),
            ("GGML_KIMI_CPU_SINK_VRAM_GLOBAL_POOL", "1"),
            ("GGML_KIMI_CPU_SINK_VRAM_EXCLUSIVE", "1"),
            ("GGML_KIMI_CPU_SINK_VRAM_ASYNC_UPLOAD", "1"),
            ("GGML_KIMI_CPU_SINK_VRAM_GPU_HIDDEN", "0"),
            ("GGML_KIMI_CPU_SINK_VRAM_KEEP_ACROSS_REQUESTS", "1"),
            ("GGML_KIMI_CPU_SINK_VRAM_PREFILL", "1"),
        )
    )
    if not unified:
        s.put(
            "GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB", headroom, "profile", "MEMORY_HEADROOM_MIB"
        )
    elastic_margin = s.engine("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB", margin)
    if not unified and all(
        s.engine("GGML_KIMI_CPU_SINK_VRAM_" + name, default) == expected
        for name, default, expected in (
            ("ELASTIC", "1", "1"),
            ("GLOBAL_POOL", "1", "1"),
            ("KEEP_ACROSS_REQUESTS", "1", "1"),
            ("ELASTIC_SAFETY_BLOCKS", "0", "0"),
        )
    ):
        s.put(
            "GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB",
            elastic_margin,
            "derived",
            (
                "the elastic margin (GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB) of "
                "a tier that grows on a discrete GPU"
            ),
        )
    else:
        s.put("GGML_KIMI_CPU_SINK_VRAM_SAFETY_MIB", "1024", "profile")
    s.profile(
        (
            ("GGML_KIMI_CPU_SINK_VRAM_ELASTIC", "1"),
            ("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_DEMAND_RECLAIM", "1"),
            ("GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_BLOCKS", "0"),
        )
    )
    s.put(
        "GGML_KIMI_CPU_SINK_VRAM_ELASTIC_SAFETY_MIB",
        margin,
        "profile",
        "VRAM_RESERVE_MIB, VRAM_ELASTIC_SAFETY_MIB",
    )
    s.put("GGML_KIMI_CPU_SINK_VRAM_WATERMARK_LOW_MIB", str(low), "profile", "VRAM_RESERVE_MIB")
    s.put(
        "GGML_KIMI_CPU_SINK_VRAM_WATERMARK_HIGH_MIB",
        margin,
        "profile",
        "VRAM_RESERVE_MIB, VRAM_ELASTIC_SAFETY_MIB",
    )
    if mxfp4:
        s.put("GGML_KIMI_CPU_SINK_VRAM_MXFP4_GPU_HIDDEN", "1", "profile")
    s.profile(
        (
            ("GGML_KIMI_CPU_SINK_VRAM_DEFERRED_UPLOAD", "1"),
            ("GGML_KIMI_CPU_SINK_EARLY_UP_GATE", "1"),
            ("GGML_KIMI_CPU_SINK_ROLE_MAJOR", "1"),
            ("GGML_KIMI_CPU_SINK_NONBLOCKING_DOWN_SCHED", "1"),
            ("GGML_KIMI_CPU_SINK_PREFETCH", "2"),
            ("GGML_KIMI_CPU_SINK_DYNAMIC_ROWS", "1"),
            ("GGML_KIMI_CPU_SINK_DYNAMIC_CHUNK_ROWS", "64"),
            ("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE", "1"),
            ("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_PROMPT_ONLY", "1"),
            ("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_BATCH", "1"),
            ("GGML_SCHED_MOE_COPY_EXPERT_PACK_SOURCE_CACHE_BYPASS", "1"),
            ("LLAMA_DROP_EXPERT_MMAP_AFTER_PROMPT", "1"),
            ("LLAMA_DROP_MMAP_EVERY_PROMPT_BATCH", "1"),
            ("LLAMA_DROP_MMAP_FADVISE_DONTNEED", "1"),
            ("LLAMA_DROP_DENSE_MMAP_CACHE", "1"),
        )
    )
    s.typed_number("THREADS", threads)
    if not re.fullmatch(r"[1-9][0-9]*", threads):
        s.error("THREADS must be a positive integer", 2, append_status=True)
    s.typed_number("CONTEXT", context)
    if not re.fullmatch(r"[1-9][0-9]*", context) or int(context) > 1048576:
        s.error("CONTEXT must be a positive number of tokens up to 1048576", 2, append_status=True)
    s.typed_number("UBATCH", ubatch)
    s.typed_number("CTX_CHECKPOINTS", checkpoints, leading_zero=True)
    s.typed_number("PROMPT_CACHE_MIB", prompt_mib)
    if (
        not re.fullmatch(r"[1-9][0-9]*", ubatch)
        or not re.fullmatch(r"[0-9]+", checkpoints)
        or not re.fullmatch(r"[1-9][0-9]*", prompt_mib)
    ):
        s.error(
            (
                "UBATCH and PROMPT_CACHE_MIB must be positive integers, "
                "CTX_CHECKPOINTS a nonnegative one"
            ),
            2,
            append_status=True,
        )
    s.typed_number("VRAM_ELASTIC_SAFETY_MIB", margin)
    if not re.fullmatch(r"[1-9][0-9]*", margin) or not high <= int(margin) <= 8192:
        s.error(
            (
                f"VRAM_ELASTIC_SAFETY_MIB must be in {high}..8192 (the decode "
                f"watermark above VRAM_RESERVE_MIB)"
            ),
            2,
            append_status=True,
        )
    s.typed_number("MEMORY_HEADROOM_MIB", headroom)
    s.typed_number("MEMORY_RESERVE_MIB", memory_reserve)
    if not re.fullmatch(r"[0-9]+", headroom) or not re.fullmatch(r"[0-9]+", memory_reserve):
        s.error(
            "MEMORY_HEADROOM_MIB and MEMORY_RESERVE_MIB must be nonnegative integers",
            2,
            append_status=True,
        )
    graph = max_tokens < int(ubatch)
    s.derived(
        "LD_LIBRARY_PATH",
        binary_dir + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else ""),
        "SERVER_BIN",
    )
    s.derived(
        "LD_PRELOAD", ":".join(libraries), "SERVER_BIN, PRELOAD_LIB, expert types " + expert_types
    )
    s.derived("GGML_CUDA_MEMORY_RESERVE_MIB", reserve, "VRAM_RESERVE_MIB")
    s.derived("GGML_MOE_EXPERT_PACK", pack, "EXPERT_PACK")
    s.derived(
        "GGML_MOE_EXPERT_GGUF_ALIAS_TSV",
        alias + ((":" if alias else "") + dense_alias if dense_alias else ""),
        "EXPERT_ALIAS" + (", DENSE_HOMES" if dense_alias else ""),
    )
    s.derived("GGML_KIMI_CPU_SINK_THREADS", threads, "THREADS")
    if unified:
        s.put(
            "GGML_KIMI_CPU_SINK_VRAM_BYTES",
            "auto",
            "derived",
            f"unified memory (memory.total {gpu_mib})",
        )
        s.put(
            "GGML_KIMI_CPU_SINK_VRAM_HOST_HEADROOM_MIB",
            str(max(int(headroom), 4096)),
            "derived",
            f"unified memory (memory.total {gpu_mib}), MEMORY_HEADROOM_MIB",
        )
    s.put(
        "GGML_KIMI_CPU_PREFILL_ACTUAL_SHAPE",
        "0" if s.engine("GGML_KIMI_CPU_SINK", "1") == "0" else "1",
        "derived",
        "GGML_KIMI_CPU_SINK",
    )
    if alias_unaligned > 0 and graph:
        s.put(
            "GGML_MOE_IO_ALIGNED_ALIAS_BATCH",
            "1",
            "derived",
            f"{alias_unaligned} unaligned entries of EXPERT_ALIAS",
        )
    for pair in conf.get("ENGINE_ENV", "").split():
        name, _, value = pair.partition("=")
        s.put(name, value if "=" in pair else pair, "model", conf_path)
    for name in sorted(s.settings):
        if engine_name(name) and name not in s.set_names and name not in LAUNCHER_SETTINGS:
            s.put(name, s.settings[name], "settings", settings_path)
    for name in sorted(s.env):
        if engine_name(name) and name not in s.set_names and name not in LAUNCHER_SETTINGS:
            s.put(name, s.env[name], "override", "environment")
    if s.value("ss_available") and s.text("port_busy", port):
        s.refuse(f"Port {port} is already in use")
    if homes:
        s.emit(
            f"[ssd_setting] component=launcher name=DENSE_HOMES value={homes} "
            f"source={s.given_by('DENSE_HOMES')} "
            f"input=drive={dense['dense_drive']} "
            f"drive_bytes={dense['dense_drive_bytes']} ram={dense['dense_ram']} "
            f"ram_bytes={dense['dense_ram_bytes']} "
            f"stage_bytes={dense['dense_stage_bytes']}\n"
        )
    if gpu_available:
        busy = s.text("gpu_processes")
        if busy and int(reserve) == 0:
            s.refuse(
                "The GPU already runs a compute process; stop it first or set VRAM_RESERVE_MIB:\n"
                + busy
            )
        elif busy:
            s.emit(f"Sharing the GPU with (VRAM_RESERVE_MIB={reserve}):\n{busy}\n", "stderr")
    else:
        s.emit("warning: nvidia-smi not found; skipping the idle-GPU check\n", "stderr")
    if not s.value("cgroup_v2"):
        s.refuse("cgroup v2 is required (the RAM expert cache reads memory.max)")
    total, available = cast(tuple[int, int], s.value("memory_kib"))
    memory_source = "MEMORY_MAX"
    if not memory_max:
        memory_max = str((available - (int(headroom) + int(memory_reserve)) * 1024) * 1024)
        memory_source = (
            f"total {total / 1048576:.2f} GiB, in use by the system "
            f"{(total - available) / 1048576:.2f} GiB, headroom "
            f"{int(headroom) / 1024:.2f} GiB, reserved for others "
            f"{int(memory_reserve) / 1024:.2f} GiB"
        )
        if int(memory_max) < 4 * 1073741824:
            s.refuse(
                f"Only {int(memory_max) >> 20} MiB would be left for llama-server ({memory_source})"
            )
    uid = cast(int, s.value("euid"))
    s.scope = ["systemd-run", "--scope"] if uid == 0 else ["systemd-run", "--user", "--scope"]
    if uid != 0:
        if not s.value("user_manager"):
            s.refuse(
                f"No systemd user manager; run from a login session (ssh) or "
                f"'loginctl enable-linger {env.get('USER', '')}'"
            )
        if not s.value("memory_controller", str(uid)):
            s.refuse(
                f"The memory controller is not delegated to "
                f"/sys/fs/cgroup/user.slice/user-{uid}.slice/user@{uid}.service/cgroup.controllers"
            )
    cache_args = (
        ["--cache-disk", prompt_cache, "--cache-disk-mib", prompt_mib]
        if prompt_cache
        else ["--cache-ram", "0"]
    )
    args = [
        server,
        "-m",
        model,
        "-c",
        context,
        "-np",
        "1",
        "-t",
        threads,
        "-tb",
        threads,
        "-b",
        ubatch,
        "-ub",
        ubatch,
        "--ctx-checkpoints",
        checkpoints,
        "--fit",
        "off",
        "-ngl",
        "99",
        "-fa",
        "on",
    ]
    if conf.get("TEMPLATE"):
        template = conf["TEMPLATE"]
        args += [
            "--chat-template-file",
            template if template.startswith("/") else repo + "/" + template,
        ]
    if conf.get("TEMPLATE_KWARGS"):
        args += ["--chat-template-kwargs", conf["TEMPLATE_KWARGS"]]
    args += conf.get("SAMPLING", "").split()
    args += [
        "--defer-experts",
        "--no-repack",
        *dense_args,
        *cache_args,
        "--no-warmup",
        "--host",
        host,
        "--port",
        port,
    ]
    args += s.settings.get("ARGS", "").split()
    args += s.inputs.extra
    scope_name = env.get("LAUNCH_SCOPE") or model_name + "-" + s.text("scope_suffix")
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_.-]*", scope_name) or len(scope_name) > 200:
        s.error("Invalid LAUNCH_SCOPE")
    s.scope += [
        "--quiet",
        "--unit=" + scope_name,
        "-p",
        "MemoryMax=" + memory_max,
        "-p",
        "MemorySwapMax=0",
    ]
    if cpuset:
        s.scope += ["taskset", "-c", cpuset]
    s.argv = args
    route = "prefill on the CPU sink"
    if graph:
        route += (
            f" up to {max_tokens} tokens and where the GPU does not offload, on the GPU graph above"
        )
    s.emit(
        f"{model_name} on http://{host}:{port} (threads={threads}, context={context}, {route})\n"
    )
    s.emit(f"memory.max={memory_max} ({memory_source})\n")
    for value in s.scope:
        s.emit("SCOPE " + value + "\n")
    for name, value in s.assigned:
        s.emit("ENV " + name + "=" + value + "\n")
    for value in args:
        s.emit("ARG " + value + "\n")


def resolve_linux(
    inputs: LaunchInput, facts: Mapping[Fact, Reply], *, execute: bool = False
) -> Resolution:
    state = LinuxState(inputs, facts, execute=execute)
    try:
        _resolve(state)
    except AwaitFact as missing:
        return state.finish(None, missing.fact)
    except Finished as stopped:
        return state.finish(stopped.code)
    return state.finish(0)
