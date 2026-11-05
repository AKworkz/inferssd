"""Actual typed artifact effects against the already frozen reference transcripts."""

import hashlib
import os
from pathlib import Path

import pytest
from observe_launch_reference import header, normalize
from test_launch_linux import GOLDENS

from inferssd.launch.linux import resolve_linux
from inferssd.launch.prepare import prepare
from inferssd.launch.recorded import RecordedArtifacts, generate_alias
from inferssd.launch.types import LaunchInput, Output


def forbidden(*args, **kwargs):
    raise AssertionError("No subprocess or engine is allowed in this test")


@pytest.mark.parametrize("record", GOLDENS, ids=lambda r: r["case"]["name"])
def test_typed_artifact_chain(record, tmp_path):
    base = tmp_path.resolve()
    repo, d = base / "repo", base / "case"
    repo.mkdir()
    case = record["case"]
    for name in ("models", "bin", "home", "stub"):
        (d / name).mkdir(parents=True)
    model = d / "models/Synthetic-Q4.gguf"
    header(model, case.get("type", 12))
    if case.get("ambiguous"):
        header(d / "models/Synthetic-other.gguf")
    for name in ("libkimi-no-mmap-prefetch.so", "libkimi-mxfp4-parity.so", "libkimi-vram-tier.so"):
        (d / "bin" / name).touch()
    binary = d / "bin/llama-server"
    binary.write_text("DO NOT EXECUTE\n")
    binary.chmod(0o755)
    (d / "model.conf").write_text(record["conf"])
    (d / "selected.idx").write_text(
        "tensor\texpert_idx\tnbytes\tsource_path\toffset\nx\t0\t128\t" + str(model) + "\t96\n"
    )
    (d / "model.pack").write_bytes(b"fixture pack selected without parsing")
    if "settings" in case:
        (d / "settings.env").write_text(case["settings"])
    if "dense" in case:
        (d / "dense.tsv").write_text(case["dense"])
    env = {
        k: v.replace("@case", str(d)).replace("@repo", str(repo)) for k, v in record["env"].items()
    }
    inputs = LaunchInput(str(repo), tuple(env.items()), tuple(case.get("extra", [])))
    old_host = record["host"]
    host = dict(
        filesystem=old_host["filesystem"],
        gpu_available=True,
        gpu_memory=old_host["gpu_mib"],
        gpu_processes=old_host["gpu_apps"],
        default_threads=old_host["cpu_count"],
        ss_available=True,
        port_busy=old_host["port_busy"],
        cgroup_v2=record["cgroup_v2"],
        memory_kib=[old_host["mem_total_kib"], old_host["mem_available_kib"]],
        euid=record["uid"],
        user_manager=True,
        memory_controller=True,
    )
    adapter = RecordedArtifacts(inputs, host, run=forbidden, replace=os.replace, python="never")
    for index, expected in enumerate(record["runs"]):
        if index == 2:
            os.utime(model, (4102444800, 4102444800))
        output = []
        metadata = []

        def observe(fact):
            command = "facts" if fact.name == "source_facts" else fact.name
            if command == old_host.get("fail_metadata"):
                from inferssd.launch.types import Reply

                return Reply(stderr="fixture metadata failure: " + command + "\n", code=9)
            reply = adapter(fact)
            if command in ("index", "facts", "dense"):
                metadata.append((command, reply))
            return reply

        actual = prepare(inputs, resolver=resolve_linux, observe=observe, emit=output.append)
        assert actual.code == expected["code"]
        normalized = normalize(
            {
                "stdout": "".join(e.text for e in output if e.channel == "stdout"),
                "stderr": "".join(e.text for e in output if e.channel == "stderr"),
            },
            [(repo, "@repo"), (d, "@case")],
        )
        if case["name"] == "early-resolve":
            assert actual.argv == (str(model),)
            assert output == [] and metadata == []
        else:
            assert normalized["stdout"] == expected["stdout"]
            if case["name"] == "missing-model":
                # Actual module filenames/lines are deliberately not forged.
                old_last = expected["metadata"][0]["stderr"].splitlines()[-1]
                new_last = normalize(
                    metadata[0][1].stderr, [(repo, "@repo"), (d, "@case")]
                ).splitlines()[-1]
                assert new_last == old_last
                assert normalized["stderr"].endswith(
                    "Cannot index" + expected["stderr"].split("Cannot index", 1)[1]
                )
            else:
                assert normalized["stderr"] == expected["stderr"]
        artifacts = {
            str(p.relative_to(d)): p.read_text()
            for p in sorted(d.rglob("*"))
            if p.is_file()
            and (p.name in ("experts.tsv", "experts.json") or p.name.endswith(".alias.tsv"))
        }
        assert normalize(artifacts, [(repo, "@repo"), (d, "@case")]) == expected["artifacts"]
        assert [name for name, _ in metadata] == [m["command"] for m in expected["metadata"]]
        for (_, reply), old in zip(metadata, expected["metadata"]):
            assert reply.code == old["code"]
            assert normalize(reply.stdout, [(repo, "@repo"), (d, "@case")]) == old["stdout"]


def test_mac_alias_hash_and_order(tmp_path):
    calls = []
    model = "/models/synthetic.gguf"

    def run(argv):
        calls.append("generate")
        Path(argv[argv.index("--out") + 1]).write_text("rows\n")
        Path(argv[argv.index("--summary-json") + 1]).write_text('{"model_shards":1,"entries":1}')

    def replace(source, target):
        calls.append(Path(target).suffix)
        os.replace(source, target)

    path = generate_alias(
        (model,), "muse", tmp_path, repo="/repo", python="/python", run=run, replace=replace
    )
    assert Path(path).name == hashlib.sha256(("muse\0" + model).encode()).hexdigest()[:20] + ".tsv"
    assert calls == ["generate", ".tsv", ".json"]


def test_output_failure_prevents_next_effect():
    from inferssd.launch.types import Fact, Reply, Resolution

    events = []

    def resolver(inputs, facts):
        return Resolution((), (), (), (), (Output("stdout", "before\n"),), (), Fact("write"), None)

    def emit(event):
        events.append(event.text)
        raise OSError("output failed")

    with pytest.raises(OSError, match="output failed"):
        prepare(
            LaunchInput("/repo", ()),
            resolver=resolver,
            observe=lambda f: events.append("WRONG") or Reply(),
            emit=emit,
        )
    assert events == ["before\n"]


def test_missing_fact_is_lazy_and_repeated_observation_is_distinct(tmp_path):
    from inferssd.launch.types import Fact, Reply, State

    inputs = LaunchInput("/repo", ())
    adapter = RecordedArtifacts(inputs, {}, run=forbidden, replace=forbidden, python="never")
    assert adapter(Fact("is_file", (str(tmp_path / "missing"),))).value is False
    with pytest.raises(ValueError, match="Missing recorded host fact: gpu_available"):
        adapter(Fact("gpu_available"))
    state = State(inputs, {Fact("x"): Reply(1)})
    assert state.value("x") == 1
    from inferssd.launch.types import AwaitFact

    with pytest.raises(AwaitFact) as missing:
        state.value("x")
    assert missing.value.fact.occurrence == 1
