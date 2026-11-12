import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

ROOT = Path(__file__).resolve().parents[2]


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


launcher = load("mac_launcher", ROOT / "framework/tests/fixtures/macos/run-macos-ssd.py")
indexer = load("alias_indexer", ROOT / "framework/tests/fixtures/macos/build-gguf-expert-alias-tsv.py")


def test_split_model_requires_first_and_every_shard(tmp_path):
    paths = [tmp_path / f"model-{i:05d}-of-00003.gguf" for i in range(1, 4)]
    for path in paths:
        path.touch()
    assert launcher.model_shards(paths[0]) == paths
    with pytest.raises(ValueError, match="first GGUF shard"):
        launcher.model_shards(paths[1])
    paths[1].unlink()
    with pytest.raises(ValueError, match="Missing GGUF shard"):
        launcher.model_shards(paths[0])


def test_alias_rebuild_is_atomic_on_failure_and_accepts_literal_glob_characters(tmp_path, monkeypatch):
    model = tmp_path / "model[1].gguf"
    model.touch()
    cache = tmp_path / "cache"
    failed = False

    def generate(command, check):
        assert check and command.count("--model-glob") == 1
        assert command[command.index("--model-glob") + 1].endswith("model[[]1].gguf")
        output = Path(command[command.index("--out") + 1])
        summary = Path(command[command.index("--summary-json") + 1])
        output.write_text("partial" if failed else "complete index")
        if failed:
            raise subprocess.CalledProcessError(1, command)
        summary.write_text(json.dumps({"model_shards": 1, "entries": 256}))

    monkeypatch.setattr(launcher.subprocess, "run", generate)
    alias = launcher.generate_alias([model], "muse", cache)
    assert alias.read_text() == "complete index"
    failed = True
    with pytest.raises(subprocess.CalledProcessError):
        launcher.generate_alias([model], "muse", cache)
    assert alias.read_text() == "complete index"
    assert not list(cache.glob(".index-*"))


@pytest.fixture
def launch(tmp_path, monkeypatch):
    monkeypatch.setattr(launcher, "REPO", ROOT / "engine")
    model = tmp_path / "model.gguf"
    model.touch()
    build = tmp_path / "build"
    (build / "bin").mkdir(parents=True)
    for name in ("llama-server", "llama-cli"):
        binary = build / "bin" / name
        binary.write_text("#!/bin/sh\nexit 0\n")
        binary.chmod(0o755)
    for key in list(launcher.os.environ):
        if key.startswith(("SSD_LLAMA_", "GGML_MOE_EXPERT_", "LLAMA_")):
            monkeypatch.delenv(key)
    monkeypatch.setenv("SSD_LLAMA_MODEL", str(model))
    monkeypatch.setenv("SSD_LLAMA_BUILD_DIR", str(build))
    monkeypatch.setenv("XDG_CACHE_HOME", str(tmp_path / "cache"))
    monkeypatch.setattr(launcher.platform, "system", lambda: "Darwin")
    monkeypatch.setattr(launcher.platform, "machine", lambda: "arm64")
    monkeypatch.setattr(launcher.subprocess, "check_output", lambda *args, **kwargs: str(32 * 1024**3))
    alias = tmp_path / "generated.tsv"
    calls = []
    monkeypatch.setattr(launcher, "generate_alias", lambda paths, profile, directory: alias)
    monkeypatch.setattr(launcher.os, "execvpe", lambda binary, argv, env: calls.append((binary, argv, env)))
    return calls, alias, build


@pytest.mark.parametrize("profile,mode,template", [
    ("dsv", "server", "deepseek-ai-DeepSeek-V4.1.jinja"),
    ("muse", "chat", "muse-spark.jinja"),
])
def test_launcher_selects_matching_template_and_preserves_explicit_controls(launch, monkeypatch, profile, mode, template):
    calls, alias, build = launch
    monkeypatch.setenv("SSD_LLAMA_MAC_EXPERT_CACHE_MIB", "24576")
    monkeypatch.setenv("SSD_LLAMA_MAC_EXPERT_READ_AHEAD", "2")
    monkeypatch.setenv("SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT", "4096")
    monkeypatch.setenv("SSD_LLAMA_MAC_EXPERT_PREFILL", "1")
    launcher.main([profile, mode, "--seed", "7"])
    binary, argv, env = calls[0]
    assert Path(binary).parent == build / "bin"
    assert argv[argv.index("--chat-template-file") + 1].endswith("/" + template)
    assert argv[-2:] == ["--seed", "7"]
    assert env["GGML_MOE_EXPERT_GGUF_ALIAS_TSV"] == str(alias)
    assert env["SSD_LLAMA_MAC_EXPERT_CACHE_MIB"] == "24576"
    assert env["SSD_LLAMA_MAC_EXPERT_READ_AHEAD"] == "2"
    assert env["SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT"] == "4096"
    assert env["SSD_LLAMA_MAC_EXPERT_PREFILL"] == "1"


@pytest.mark.parametrize("invalid", ["missing-model", "pack", "alignment", "model-override", "alias-error"])
def test_launcher_starts_no_server_on_invalid_input(launch, monkeypatch, invalid):
    calls, _, _ = launch
    extra = []
    if invalid == "missing-model":
        monkeypatch.delenv("SSD_LLAMA_MODEL")
    elif invalid == "pack":
        monkeypatch.setenv("SSD_LLAMA_MAC_EXPERT_PACK", "pack.bin")
    elif invalid == "alignment":
        monkeypatch.setenv("SSD_LLAMA_MAC_EXPERT_READ_ALIGNMENT", "3")
    elif invalid == "model-override":
        extra = ["--model", "wrong.gguf"]
    else:
        def failed_alias(*args):
            raise subprocess.CalledProcessError(1, "generator")
        monkeypatch.setattr(launcher, "generate_alias", failed_alias)
    with pytest.raises(SystemExit) as error:
        launcher.main(["muse", "server", *extra])
    assert error.value.code == 2 and not calls


@pytest.mark.parametrize("option", ["--model=wrong.gguf", "-hf", "--model-url", "--docker-repo"])
def test_launcher_rejects_alternate_model_sources(launch, option):
    calls, _, _ = launch
    with pytest.raises(SystemExit):
        launcher.main(["dsv", "server", option, "other-model"])
    assert not calls


def test_launcher_rejects_inherited_remote_model_source(launch, monkeypatch):
    calls, _, _ = launch
    monkeypatch.setenv("LLAMA_ARG_HF_REPO", "example/model")
    with pytest.raises(SystemExit):
        launcher.main(["dsv", "server"])
    assert not calls


def test_alias_generator_checks_shape_duplicates_and_resolved_paths(tmp_path, monkeypatch):
    tensor = SimpleNamespace(name="blk.0.ffn_up_exps.weight", shape=(32, 16, 256),
                             n_bytes=256 * 128, data_offset=64, tensor_type="Q4_K")
    package = ModuleType("gguf")
    reader = ModuleType("gguf.gguf_reader")
    reader.GGUFReader = lambda path: SimpleNamespace(tensors=[tensor])
    monkeypatch.setitem(sys.modules, "gguf", package)
    monkeypatch.setitem(sys.modules, "gguf.gguf_reader", reader)
    source = tmp_path / "model.gguf"
    rows, summary = indexer.build_rows([source], {"up"}, 256, False)
    assert len(rows) == 256 and rows[-1]["offset"] == 64 + 255 * 128
    assert summary["model_shards"] == 1 and rows[0]["source_path"] == str(source.resolve())
    with pytest.raises(RuntimeError, match="duplicate expert tensor"):
        indexer.build_rows([source, tmp_path / "second.gguf"], {"up"}, 256, False)
    tensor.shape = (256, 16, 32)
    with pytest.raises(RuntimeError, match="expected 3D tensor"):
        indexer.build_rows([source], {"up"}, 256, False)
    bad_dir = tmp_path / "bad\tpath"
    bad_dir.mkdir()
    monkeypatch.chdir(bad_dir)
    with pytest.raises(RuntimeError, match="tabs or newlines"):
        indexer.build_rows([Path("model.gguf")], {"up"}, 256, False)


def test_alias_generator_refuses_empty_index(tmp_path, monkeypatch):
    package = ModuleType("gguf")
    reader = ModuleType("gguf.gguf_reader")
    reader.GGUFReader = lambda path: SimpleNamespace(tensors=[])
    monkeypatch.setitem(sys.modules, "gguf", package)
    monkeypatch.setitem(sys.modules, "gguf.gguf_reader", reader)
    with pytest.raises(RuntimeError, match="no expert tensors"):
        indexer.build_rows([tmp_path / "empty.gguf"], {"up"}, 256, False)
