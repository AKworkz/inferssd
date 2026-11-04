"""Typed preparation/ownership contracts; synthetic metadata is not GPU evidence."""

import json
from dataclasses import asdict, replace
from functools import partial
from pathlib import Path

import pytest
from test_engine_runtime import Effects, Owner, policy
from test_gguf_alias import fixture as gguf_fixture
from test_strategy_inputs import Q8_0, deepseek_like, moe, muse_like, write_gguf

from inferssd.configuration.settings import parse
from inferssd.engine import runtime
from inferssd.launch.prepare import prepare
from inferssd.launch.recorded import RecordedArtifacts
from inferssd.launch.typed import LaunchSpec, reject_legacy_environment, resolve_typed
from inferssd.launch.typed_command import main
from inferssd.launch.types import LaunchInput, Threads
from inferssd.source import gguf_alias


def setup(tmp_path, platform="linux", automatic=False):
    root = tmp_path / "repo"
    template = root / "engine/models/templates/muse-spark.jinja"
    template.parent.mkdir(parents=True)
    template.write_text("template")
    binary = root / "bin/server"
    binary.parent.mkdir()
    binary.write_text("must never execute")
    binary.chmod(0o755)
    model = root / "model.gguf"
    tensors = [
        (f"blk.0.ffn_{role}_exps.weight", (32, 32, 256), 8, i * 278528)
        for i, role in enumerate(("up", "gate", "down"))
    ]
    gguf_fixture(model, tensors=tensors, payload=3 * 278528)
    settings = root / "input.ssd"
    source = root / "provided.tsv"
    source.write_text("explicit source fixture; resolver does not validate payloads")
    text = "inferssd-settings 1\nprefetch.depth=4 # measured predictor depth\n"
    if not automatic:
        text += "expert.source=" + json.dumps([str(source)]) + " # provided index\n"
    settings.write_text(text)
    spec = LaunchSpec(str(model), 8080, str(settings), (("prefetch.depth", "2"),), str(binary))
    host = {
        "platform_system": "Linux" if platform == "linux" else "Darwin",
        "platform_machine": "arm64",
        "euid": 0,
        "cgroup_v2": True,
        "gpu_available": True,
        "gpu_processes": "",
        "engine_gpu_backends": ["CUDA"] if platform == "linux" else ["Metal"],
    }
    inputs = LaunchInput(str(root), (("PATH", "/bin"), ("PRIVATE_TOKEN", "never-persist")))
    out = tmp_path / "run"
    out.mkdir()
    seen = []

    def indexer(command):
        assert command[:4] == ["metadata-python", "-I", "-m", "inferssd.source.gguf_alias"]
        seen.append(command)
        assert gguf_alias.main(command[4:]) == 0

    adapter = RecordedArtifacts(
        inputs, host, run=indexer, replace=lambda a, b: Path(a).replace(b), python="metadata-python"
    )
    return inputs, spec, host, adapter, out, seen


def resolve(inputs, spec, adapter, out, platform="linux", configured=None):
    return prepare(
        inputs,
        resolver=partial(
            resolve_typed,
            spec=spec,
            platform=platform,
            directory=str(out),
            memory=4 << 30,
            configured=configured,
        ),
        observe=adapter,
        emit=lambda _: None,
    )


@pytest.mark.parametrize("platform", ["linux", "darwin"])
def test_settings_file_and_cli_provenance_reach_native_argv(tmp_path, platform):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    result = resolve(inputs, spec, adapter, out, platform)
    assert result.code == 0 and not result.assigned_environment
    assert "--ssd-settings" in result.argv and "--ssd-telemetry" in result.argv
    assert result.argv[result.argv.index("--ssd-prefetch-depth") + 1] == "2"
    assert not {"-t", "-tb", "-C", "-Cb"} & set(result.argv)  # unset counts keep server defaults
    body = (out / "settings.ssd").read_text()
    assert "expert.source" in body and "provided index" in body
    assert "prefetch.depth" not in body  # actual CLI wins, registry defaults stay defaults
    native = parse(body, str(out / "settings.ssd"), overrides=spec.overrides, platform=platform)
    assert native.at("prefetch.depth").origin == "cli"
    assert native.at("vram.budget").origin == "default"
    assert native.at("expert.source").origin == "file"
    assert not any("GGML_KIMI" in x for x in result.argv)
    assert ("MemoryMax=4294967296" in result.scope) == (platform == "linux")
    # Every platform's SSD executor reads the routed experts; none stay with the model.
    assert "--defer-experts" in result.argv and "--cpu-moe" not in result.argv
    assert not (out / "native-events.jsonl").exists()  # only the native writer owns creation
    with pytest.raises(FileExistsError):
        resolve(inputs, spec, adapter, out, platform)


@pytest.mark.parametrize(
    "platform,present,chosen",
    [
        ("linux", ["build", "build-linux-cpu", "build-linux-cuda"], "build-linux-cuda"),
        ("linux", ["build", "build-linux-cpu"], "build-linux-cpu"),
        ("linux", ["build", "build-mac"], "build"),
        ("darwin", ["build", "build-mac", "build-macos-metal"], "build-macos-metal"),
        ("darwin", ["build", "build-mac"], "build-mac"),
        ("darwin", ["build", "build-linux-cpu"], "build"),
    ],
)
def test_repo_adapter_prefers_preset_build_directories(tmp_path, platform, present, chosen):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    for name in present:
        binary = Path(inputs.repo) / name / "bin/inferssd-server"
        binary.parent.mkdir(parents=True)
        binary.write_text("must never execute")
        binary.chmod(0o755)
    result = resolve(inputs, replace(spec, binary=None), adapter, out, platform)
    assert result.code == 0
    assert str((Path(inputs.repo) / chosen / "bin/inferssd-server").resolve()) in result.argv


@pytest.mark.parametrize(
    "platform,first", [("linux", "build-linux-cuda"), ("darwin", "build-macos-metal")]
)
def test_repo_adapter_reports_the_preset_build_when_none_exists(tmp_path, platform, first):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    missing = "missing server executable: .*/" + first + "/bin/inferssd-server"
    with pytest.raises(ValueError, match=missing):
        resolve(inputs, replace(spec, binary=None), adapter, out, platform)


def test_auto_alias_uses_real_strict_header_indexer(tmp_path):
    inputs, spec, _, adapter, out, calls = setup(tmp_path, automatic=True)
    result = resolve(inputs, spec, adapter, out)
    assert result.code == 0 and len(calls) == 1
    value = parse((out / "settings.ssd").read_text(), str(out / "settings.ssd"))
    alias = Path(value.at("expert.source").value[0])
    rows = alias.read_text().splitlines()
    assert "--n-experts" not in calls[0]  # the expert count comes from the model
    assert len(rows) == 1 + 3 * 256 and "source_path" in rows[0]
    assert str(Path(spec.model)) in rows[-1]
    assert "source=generated" in value.at("expert.source").receipt


@pytest.mark.parametrize(
    "name", ["GGML_KIMI_CPU_SINK_THREADS", "TASK3_PREFETCH", "SETTINGS", "TRACE_ENV"]
)
def test_retired_inputs_fail_by_name_without_value(name):
    with pytest.raises(ValueError) as error:
        reject_legacy_environment({name: "PRIVATE-OLD-CONTROL"})
    assert name in str(error.value) and "PRIVATE-OLD-CONTROL" not in str(error.value)


@pytest.mark.parametrize(
    "extra",
    [
        ("-mwrong",),
        ("--model=wrong",),
        ("--port=8",),
        ("--ssd-prefetch-depth", "8"),
        ("--ssd_prefetch_depth", "8"),
        ("-np2",),
        ("--threads=9",),
        ("--threads_batch", "9"),
        ("--ubatch-size", "64"),
    ],
)
def test_cannot_override_index_model_or_owned_listener(tmp_path, extra):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="explicit typed"):
        resolve(replace(inputs, extra=extra), spec, adapter, out)
    assert not (out / "settings.ssd").exists()


@pytest.mark.parametrize(
    "fact,value,message",
    [
        ("systemd", False, "systemd is required"),
        ("cgroup_v2", False, "cgroup v2"),
        ("gpu_available", False, "nvidia-smi"),
    ],
)
def test_resource_preparation_refuses_before_materialized_settings(tmp_path, fact, value, message):
    inputs, spec, host, _, out, _ = setup(tmp_path)
    host[fact] = value
    adapter = RecordedArtifacts(
        inputs,
        host,
        run=lambda _: pytest.fail("no index"),
        replace=lambda *_: None,
        python="unused",
    )
    with pytest.raises(ValueError, match=message):
        resolve(inputs, spec, adapter, out)
    assert not (out / "settings.ssd").exists()


def test_runtime_uses_typed_resolution_and_keeps_owned_process_port(tmp_path, monkeypatch):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    effects = Effects()
    owner = Owner(effects)
    captured = {}
    monkeypatch.setattr(runtime, "LiveArtifacts", lambda *a, **kw: adapter)
    monkeypatch.setattr(runtime, "NativeEffects", lambda: effects)

    def owned(result, directory, execution_policy, *, reservation):
        assert execution_policy.platform == "linux"
        assert reservation is effects.reservation
        captured.update(result=result, directory=directory)
        return owner

    # Keep the real Runtime→Engine wiring, replacing only OS ownership effects.
    monkeypatch.setattr(runtime, "OwnedProcess", owned)
    monkeypatch.setattr(effects, "reservation", object(), raising=False)
    value = runtime.Runtime(
        inputs, policy(tmp_path, "linux"), host=object(), emit=lambda _: None, configuration=spec
    )
    engine = value.engine(
        str(out),
        8080,
        4,
        "all",
        "",
        [],
        configured=parse(Path(spec.settings_file).read_text(), spec.settings_file),
    )
    assert engine.ok and captured["result"].argv[1:3] == ("-m", spec.model)
    env = dict(captured["result"].effective_environment)
    assert "INFERSSD_DEBUG" not in env  # controller observes JSONL without debug logging
    assert "GGML_KIMI_CPU_SINK_TRACE" not in env and "LLAMA_SERVER_PORT" not in env
    engine.stop()
    assert effects.events[-1] == ("busy", 8080)


def test_busy_port_refuses_before_claim_or_owner(tmp_path, monkeypatch):
    effects = Effects(busy=True)
    monkeypatch.setattr(runtime, "NativeEffects", lambda: effects)
    value = runtime.Runtime(
        LaunchInput("/unused", ()),
        policy(tmp_path),
        host=object(),
        emit=lambda _: None,
        configuration=LaunchSpec("/unused", 8080),
    )
    with pytest.raises(ValueError, match="already in use"):
        value.engine(str(tmp_path / "run"), 8080, 0, "static", "", [])
    assert effects.events == [("busy", 8080)]


def test_typed_command_prepares_native_file_without_executing_or_recording_secrets(
    tmp_path, monkeypatch
):
    inputs, spec, host, _, out, _ = setup(tmp_path)
    contract = tmp_path / "policy.json"
    contract.write_text(json.dumps(asdict(policy(tmp_path, "linux"))))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(host))
    receipt = tmp_path / "resolution.json"
    monkeypatch.setattr("os.environ", dict(inputs.environment))
    assert (
        main(
            [
                "--dry-run",
                "--platform",
                "linux",
                "--repo",
                inputs.repo,
                "--model",
                spec.model,
                "--profile",
                "muse",
                "--binary",
                spec.binary,
                "--settings",
                spec.settings_file,
                "--set",
                "prefetch.depth=2",
                "--cpus",
                "0-2,4",
                "--batch-threads",
                "3",
                "--runtime-policy",
                str(contract),
                "--host-facts",
                str(facts),
                "--output-dir",
                str(out),
                "--result",
                str(receipt),
            ]
        )
        == 0
    )
    assert (out / "settings.ssd").exists() and (out / ".inferssd-owner").exists()
    assert not (out / "runtime.json").exists()
    assert "never-persist" not in receipt.read_text()
    argv = json.loads(receipt.read_text())["argv"]
    assert "--ssd-prefetch-depth" in argv
    pools = argv[argv.index("-t") : argv.index("--chat-template-file")]
    strict = ["--cpu-strict", "1", "--cpu-strict-batch", "1"]
    assert pools == ["-t", "4", "-tb", "3", "-C", "0x17", "-Cb", "0x17"] + strict


def test_omitted_output_directory_creates_a_new_run_directory(tmp_path, monkeypatch, capsys):
    inputs, spec, host, _, _, _ = setup(tmp_path)
    contract = tmp_path / "policy.json"
    contract.write_text(json.dumps(asdict(policy(tmp_path, "linux"))))
    facts = tmp_path / "facts.json"
    facts.write_text(json.dumps(host))
    work = tmp_path / "work"
    work.mkdir()
    monkeypatch.chdir(work)
    monkeypatch.setattr("os.environ", dict(inputs.environment))
    command = [
        "--dry-run",
        "--platform",
        "linux",
        "--model",
        spec.model,
        "--profile",
        "muse",
    ]
    command += ["--binary", spec.binary, "--repo", inputs.repo, "--settings", spec.settings_file]
    command += ["--runtime-policy", str(contract), "--host-facts", str(facts)]
    assert main(command) == 0 and main(command) == 0
    runs = sorted((work / "runs").iterdir())
    assert len(runs) == 2 and all((run / "settings.ssd").exists() for run in runs)
    assert str(runs[0]) in capsys.readouterr().err


def test_owned_snapshot_does_not_reread_mutated_settings_file(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    configured = parse(Path(spec.settings_file).read_text(), spec.settings_file)
    Path(spec.settings_file).unlink()
    assert resolve(inputs, spec, adapter, out, configured=configured).code == 0


@pytest.mark.parametrize("threads", [4, 8])
def test_mac_measured_prefill_threads_are_explicit_and_preserved(tmp_path, threads):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, "darwin")
    spec = replace(spec, threads=Threads(threads, 12))
    result = resolve(inputs, spec, adapter, out, "darwin")
    assert result.argv[result.argv.index("-t") + 1] == str(threads)
    assert result.argv[result.argv.index("-tb") + 1] == "12"
    assert "-C" not in result.argv and "--cpu-strict" not in result.argv
    launch = json.loads(result.outputs[-1].text)["launch"]
    assert launch["decode_threads"] == threads and launch["batch_threads"] == 12


@pytest.mark.parametrize(
    "threads,decode,batch,mask",
    [
        (Threads(6), "6", "6", None),
        (Threads(batch=5), None, "5", None),
        (Threads(cpus=(5, 0, 9)), "3", "3", "0x221"),
        (Threads(2, 3, (64, 1, 2)), "2", "3", hex((1 << 64) | 6)),
    ],
)
def test_threadpool_counts_and_strict_cpu_masks_reach_native_argv(
    tmp_path, threads, decode, batch, mask
):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    # Decode participants bound only the decode pool, the one that runs decode.
    spec = replace(spec, overrides=spec.overrides + (("cpu.decode_participants", "2"),))
    result = resolve(inputs, replace(spec, threads=threads), adapter, out)
    argv = list(result.argv)
    assert (argv[argv.index("-t") + 1] if "-t" in argv else None) == decode
    assert argv[argv.index("-tb") + 1] == batch
    if mask is None:
        assert not {"-C", "-Cb", "--cpu-strict", "--cpu-strict-batch"} & set(argv)
    else:
        assert argv[argv.index("-C") + 1] == argv[argv.index("-Cb") + 1] == mask
        assert argv[argv.index("--cpu-strict") + 1] == "1"
        assert argv[argv.index("--cpu-strict-batch") + 1] == "1"
    launch = json.loads(result.outputs[-1].text)["launch"]
    assert launch["cpus"] == sorted(threads.cpus)


@pytest.mark.parametrize(
    "platform,threads,overrides,message",
    [
        ("linux", Threads(batch=0), (), "thread counts"),
        ("linux", Threads(batch=-1), (), "thread counts"),
        ("linux", Threads(batch=257), (), "thread counts"),
        ("linux", Threads(batch=True), (), "thread counts"),
        ("linux", Threads(decode=0), (), "thread counts"),
        ("linux", Threads(cpus=(512,)), (), "strict CPUs"),
        ("linux", Threads(cpus=(1, 1)), (), "strict CPUs"),
        ("linux", Threads(3, cpus=(0, 1)), (), "exceed the strict CPUs"),
        ("linux", Threads(batch=3, cpus=(0, 1)), (), "exceed the strict CPUs"),
        ("darwin", Threads(cpus=(0, 1)), (), "cannot pin"),
        ("linux", Threads(cpus=(0, 1)), (("cpu.io_cpus", "1"),), "overlaps"),
        ("linux", Threads(2, 4), (("cpu.decode_participants", "3"),), "decode_participants"),
        ("linux", Threads(cpus=(0, 1), decode_cpus=(0, 1)), (), "excludes a CPU set"),
        ("linux", Threads(3, 8, (), (0, 1)), (), "exceeds the decode CPUs"),
        ("darwin", Threads(decode_cpus=(0, 1)), (), "cannot pin"),
        ("linux", Threads(2, 8, (), (0, 1)), (("cpu.io_cpus", "1"),), "overlaps"),
    ],
)
def test_bad_explicit_threads_refused_before_native_file(
    tmp_path, platform, threads, overrides, message
):
    inputs, spec, _, adapter, out, _ = setup(tmp_path, platform)
    spec = replace(spec, threads=threads, overrides=spec.overrides + overrides)
    with pytest.raises(ValueError, match=message):
        resolve(inputs, spec, adapter, out, platform)
    assert not (out / "settings.ssd").exists()


def test_decode_cpus_pin_only_the_decode_pool(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    threads = Threads(None, 19, (), (5, 6, 7, 8, 9, 15, 16, 17, 18, 19))
    result = resolve(inputs, replace(spec, threads=threads), adapter, out)
    argv = list(result.argv)
    assert argv[argv.index("-t") : argv.index("-t") + 4] == ["-t", "10", "-tb", "19"]
    assert argv[argv.index("-C") + 1] == hex(0b11111000001111100000)
    assert argv[argv.index("--cpu-strict") + 1] == "1"
    # The batch pool has its own count and no mask: it never inherits the decode CPUs.
    assert not {"-Cb", "--cpu-strict-batch"} & set(argv)
    launch = json.loads(result.outputs[-1].text)["launch"]
    assert (launch["cpus"], launch["decode_cpus"]) == ([], list(threads.decode_cpus))


def test_prompt_cache_on_disk_per_model(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    result = resolve(inputs, spec, adapter, out)
    assert result.argv[result.argv.index("--cache-ram") + 1] == "0"
    assert "--cache-disk" not in result.argv
    second = tmp_path / "second"
    second.mkdir()
    cache = str(tmp_path / "prompts")
    result = resolve(inputs, replace(spec, prompt_cache=cache), adapter, second)
    argv = list(result.argv)
    assert argv[argv.index("--cache-disk") + 1] == str(Path(cache) / "model")
    assert argv[argv.index("--cache-disk-mib") + 1] == "16384" and "--cache-ram" not in argv


def test_sizes_default_to_the_context_and_batch():
    assert (LaunchSpec("m", 1).context, LaunchSpec("m", 1).batch) == (None, None)


@pytest.mark.parametrize(
    "extra",
    [
        ("--fit=on",),
        ("--chat-template-file", "other"),
        ("--chat-template-kwargs", "{}"),
        ("--cache-disk", "/tmp/x"),
        ("--cache-ram", "8192"),
        ("-ngl0",),
        ("--cpu-moe",),
        ("--no-host",),
        ("--tensor-split=1,0",),
        ("-otbad",),
        ("--cpu-mask=ff",),
        ("-Cr", "0-3"),
        ("-Cb0xff",),
        ("--cpu_strict", "0"),
        ("--cpu-strict-batch", "0"),
    ],
)
def test_extra_cannot_replace_selected_native_placement(tmp_path, extra):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="explicit typed"):
        resolve(replace(inputs, extra=extra), spec, adapter, out)


@pytest.mark.parametrize("explicit,expected", [(False, 1), (True, 0)])
def test_deepseek_profile_materializes_model_policy_and_preserves_explicit_values(
    tmp_path, explicit, expected
):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    inputs = replace(inputs, profile="dsv")
    (Path(inputs.repo) / "engine/models/templates/deepseek-ai-DeepSeek-V4.1.jinja").write_text(
        "template"
    )
    if explicit:
        spec = replace(
            spec,
            overrides=spec.overrides + (("model.engram_prefetch", "0"), ("model.hc_fused", "0")),
        )
    result = resolve(inputs, spec, adapter, out)
    assert result.code == 0
    native = parse((out / "settings.ssd").read_text(), overrides=spec.overrides)
    assert native.at("model.engram_prefetch").value == expected
    assert native.at("model.hc_fused").value == expected
    # Earlier turns keep their thinking, so a history rendered again matches the cache.
    argv = list(result.argv)
    assert argv[argv.index("--chat-template-kwargs") + 1] == '{"drop_thinking": false}'
    assert argv.index("--chat-template-kwargs") == argv.index("--chat-template-file") + 2


@pytest.mark.parametrize("profile", ["qwen4exp", "qwen35moe"])
def test_qwen_profiles_launch_with_the_template_in_their_gguf(tmp_path, profile):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    inputs = replace(inputs, profile=profile)
    result = resolve(inputs, spec, adapter, out)
    assert result.code == 0 and "--chat-template-file" not in result.argv
    explicit = tmp_path / "chosen.jinja"
    explicit.write_text("chosen")
    second = tmp_path / "second"
    second.mkdir()
    result = resolve(inputs, replace(spec, template=str(explicit)), adapter, second)
    assert result.argv[result.argv.index("--chat-template-file") + 1] == str(explicit)


def test_unknown_profile_refused_before_native_file(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    with pytest.raises(ValueError, match="dsv, muse, qwen4exp, qwen35moe"):
        resolve(replace(inputs, profile="qwen"), spec, adapter, out)
    assert not (out / "settings.ssd").exists()


def qwen_like(arch="qwen4exp"):
    return moe(
        arch,
        layers=2,
        experts=8,
        used=2,
        latent=256,
        hidden=256,
        gate=Q8_0,
        up=Q8_0,
        down=Q8_0,
    )


@pytest.mark.parametrize(
    "layout, profile",
    [
        (deepseek_like, "dsv"),
        (muse_like, "muse"),
        (qwen_like, "qwen4exp"),
        (partial(qwen_like, "qwen35moe"), "qwen35moe"),
    ],
)
def test_the_profile_follows_the_gguf_architecture(tmp_path, layout, profile):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    model = str(write_gguf(tmp_path / "model.gguf", *layout()))
    (Path(inputs.repo) / "engine/models/templates/deepseek-ai-DeepSeek-V4.1.jinja").write_text("t")
    result = resolve(replace(inputs, profile=""), replace(spec, model=model), adapter, out)
    assert json.loads(result.outputs[-1].text)["profile"] == profile
    argv = list(result.argv)
    templates = {"dsv": "deepseek-ai-DeepSeek-V4.1.jinja", "muse": "muse-spark.jinja"}
    if profile in templates:
        assert argv[argv.index("--chat-template-file") + 1].endswith(templates[profile])
    else:
        assert "--chat-template-file" not in argv
    # DeepSeek-V4.1's own generation config; the GGUF carries none.
    sampling = ["--temp", "1.0", "--top-p", "0.95", "--top-k", "0", "--min-p", "0"]
    if profile == "dsv":
        assert argv[argv.index("--temp") : argv.index("--temp") + 8] == sampling
    else:
        assert "--temp" not in argv


def test_an_explicit_profile_overrides_the_architecture_and_an_unknown_one_is_refused(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    model = str(write_gguf(tmp_path / "model.gguf", *deepseek_like()))
    result = resolve(replace(inputs, profile="muse"), replace(spec, model=model), adapter, out)
    assert json.loads(result.outputs[-1].text)["profile"] == "muse" and "--temp" not in result.argv
    tensors, kv = moe(
        "mystery",
        layers=1,
        experts=4,
        used=2,
        latent=256,
        hidden=256,
        gate=Q8_0,
        up=Q8_0,
        down=Q8_0,
    )
    other = str(write_gguf(tmp_path / "other.gguf", tensors, kv))
    second = tmp_path / "second"
    second.mkdir()
    with pytest.raises(ValueError, match="no model profile for the GGUF architecture mystery"):
        resolve(replace(inputs, profile=""), replace(spec, model=other), adapter, second)


def test_the_users_sampling_wins(tmp_path):
    inputs, spec, _, adapter, out, _ = setup(tmp_path)
    model = str(write_gguf(tmp_path / "model.gguf", *deepseek_like()))
    (Path(inputs.repo) / "engine/models/templates/deepseek-ai-DeepSeek-V4.1.jinja").write_text("t")
    inputs = replace(inputs, profile="", extra=("--temp", "0.6", "--top-k=40"))
    argv = list(resolve(inputs, replace(spec, model=model), adapter, out).argv)
    assert argv[-3:] == ["--temp", "0.6", "--top-k=40"] and argv.count("--temp") == 1
    assert "--top-k" not in argv and argv[argv.index("--top-p") + 1] == "0.95"
    assert argv[argv.index("--min-p") + 1] == "0"
