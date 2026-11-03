"""Offline native-entry contracts; no server, host probe or resource scope is started."""

from __future__ import annotations

import io
import json
from dataclasses import asdict, replace

import pytest
from observe_engine_reference import CASES, observe
from observe_engine_reference import Effects as FEffects
from observe_engine_reference import api as reference_api

from inferssd.engine import core, owned
from inferssd.launch.types import Resolution
from inferssd.platform.policy import Policy
from inferssd.platform.run_directory import reserve


def policy(tmp_path, platform="darwin"):
    return Policy(
        platform,
        str(tmp_path / "device.lock"),
        str(tmp_path / "STOP"),
        1 << 30,
        1 << 20,
        0,
        0,
        0,
        90,
        2,
    )


def resolution(scope=(), environment=()):
    return Resolution(
        ("/fixture/server", "--host", "127.0.0.1", "--port", "8080"),
        (),
        environment,
        scope,
        (),
        (),
        None,
        0,
    )


@pytest.mark.parametrize(
    "case",
    [
        c
        for c in CASES
        if c["name"] in ("ready", "duplicate-knobs", "extra-env", "empty-trace", "bad-knob")
    ],
    ids=lambda c: c["name"],
)
def test_environment_original_call_order(case):
    frozen = observe(case)
    try:
        actual = core.environment_for(FEffects(case).env, 8080, case.get("envs", "K=7"))
    except ValueError as error:
        assert [type(error).__name__, str(error)] == frozen["error"]
    else:
        event = next(e for e in frozen["events"] if e[0] == "spawn")
        assert list(map(list, actual.items())) == event[1][-1]


@pytest.mark.parametrize(
    "case", [c for c in CASES if c.get("action") == "post"], ids=lambda c: c["name"]
)
def test_post_exact_reference_events(case, monkeypatch):
    f, candidate = FEffects(case), FEffects(case)
    results = []
    monkeypatch.setattr(core.urllib.request, "urlopen", candidate.urlopen)
    for function in (reference_api(f).post, core.post):
        try:
            results.append(
                (
                    function(
                        8080,
                        "/completion",
                        {"prompt": "你", "n_predict": 4},
                        **({"timeout": case["timeout"]} if "timeout" in case else {}),
                    ),
                    None,
                )
            )
        except Exception as error:
            results.append((None, (type(error), str(error))))
    assert results[0] == results[1]
    assert f.events == candidate.events


class Effects:
    def __init__(self, health=(b'"ok"',), busy=False):
        self.events, self.log, self.healths, self.busy = [], io.StringIO(), list(health), busy
        self.now = 100

    def makedirs(self, path):
        self.events.append(("mkdir", path))

    def open_log(self, path):
        self.events.append(("open", path))
        return self.log

    def clock(self):
        self.now += 0.25
        return self.now

    def sleep(self, n):
        self.events.append(("sleep", n))
        self.now += n

    def health(self, port):
        value = self.healths[0]
        if len(self.healths) > 1:
            self.healths.pop(0)
        if isinstance(value, Exception):
            raise value
        return value

    def archive(self, path):
        self.events.append(("archive", path))

    def port_busy(self, port):
        self.events.append(("busy", port))
        return self.busy


class Owner:
    def __init__(self, effects):
        self.effects, self.ended = effects, None

    def poll(self):
        return self.ended

    def confirm_listener(self, port):
        self.effects.events.append(("listener", port))

    def stop(self):
        self.effects.events.append(("stop",))
        self.ended = 0


def engine(effects, owner=None, resolver=None):
    owner = owner or Owner(effects)
    return core.Engine(
        "/fixture/instance",
        8080,
        16,
        "2:2",
        "K=7",
        [],
        environment={},
        effects=effects,
        resolve=resolver or (lambda *_: resolution()),
        own=lambda *_: owner,
    )


@pytest.mark.parametrize("health", [(b'"ok"',), (OSError("loading"), b"wait", b'"ok"')])
def test_engine_health_owned_listener_and_graceful_stop(health):
    effects = Effects(health)
    value = engine(effects)
    assert value.ok and value.alive()
    assert ("listener", 8080) in effects.events
    value.stop()
    assert effects.log.closed and not value.alive()
    assert effects.events.index(("stop",)) < effects.events.index(("archive", "/fixture/instance"))


def test_startup_exception_cleans_only_owner_and_closes_log():
    effects = Effects((ValueError("bad health"),))
    with pytest.raises(ValueError, match="bad health"):
        engine(effects)
    assert effects.log.closed and ("stop",) in effects.events
    assert not any(e[0] == "archive" for e in effects.events)


def test_foreign_busy_port_never_killed():
    effects = Effects(busy=True)
    value = engine(effects)
    with pytest.raises(RuntimeError, match="no unrelated process"):
        value.stop()
    assert effects.events.count(("sleep", 1)) == 60
    assert effects.events.count(("stop",)) == 1


def test_resolve_failure_does_not_spawn_and_closes_log():
    effects = Effects()

    def fail(*_):
        raise ValueError("bad resolve")

    with pytest.raises(ValueError, match="bad resolve"):
        engine(effects, resolver=fail)
    assert effects.log.closed and not any(e[0] == "stop" for e in effects.events)


@pytest.mark.parametrize(
    "key,value",
    [
        ("platform", "win32"),
        ("memory_budget_bytes", True),
        ("runtime_seconds", 0),
        ("host_reserve_bytes", -1),
        ("minimum_memlock_bytes", "0"),
        ("lock_path", ""),
    ],
)
def test_policy_rejects_invalid_contract(tmp_path, key, value):
    data = asdict(policy(tmp_path))
    data[key] = value
    with pytest.raises(ValueError, match=key if key != "platform" else "platform"):
        Policy.read(data)


def test_policy_exact_keys_and_roundtrip(tmp_path):
    value = policy(tmp_path)
    assert Policy.read(asdict(value)) == value
    with pytest.raises(ValueError, match="fields"):
        Policy.read(dict(asdict(value), extra=0))


def test_scope_fresh_exact_owned_unit(tmp_path):
    p = policy(tmp_path, "linux")
    scope = (
        "systemd-run",
        "--user",
        "--scope",
        "--unit=old.scope",
        "-p",
        "MemoryMax=" + str(p.memory_budget_bytes),
        "-p",
        "MemorySwapMax=0",
        "taskset",
        "-c",
        "0-1",
    )
    a, unit = owned.scope_command(resolution(scope), p)
    b, other = owned.scope_command(resolution(scope), p)
    assert unit != other and unit.startswith("inferssd-") and "--unit=" + unit in a
    assert a.index("Delegate=yes") < a.index("taskset")
    assert scope[3] == "--unit=old.scope" and "--unit=" + other in b
    with pytest.raises(ValueError, match="byte budget"):
        owned.scope_command(resolution(scope), replace(p, memory_budget_bytes=3))
    with pytest.raises(ValueError, match="Darwin"):
        owned.scope_command(resolution(scope), replace(p, platform="darwin"))


def test_loader_environment_only_restored_for_server():
    env = {
        "PATH": "/fixture/bin",
        "LD_PRELOAD": "private-loader",
        "DYLD_LIBRARY_PATH": "libs",
        "GGML_KIMI_CPU_SINK": "1",
        "PYTHONPATH": "old-checkout",
        "TOKEN": "opaque",
    }
    transported = owned.worker_environment(env)
    assert not any(k in transported for k in env if k not in ("PATH", "TOKEN"))
    restored = dict(transported)
    restored.update(json.loads(restored.pop("INFERSSD_INTERNAL_SERVER_ENV")))
    assert restored == env


def test_directory_existing_empty_allowed_but_completed_claim_not_reused(tmp_path):
    run = tmp_path / "run"
    run.mkdir()
    claim = reserve(str(run))
    claim.verify()
    (run / "server.log").write_text("live log")
    (run / "runtime.json").write_text("live spec")
    before = {p.name: p.read_bytes() for p in run.iterdir()}
    with pytest.raises(FileExistsError):
        reserve(str(run))
    assert before == {p.name: p.read_bytes() for p in run.iterdir()}


def test_directory_two_competing_claims(tmp_path):
    from concurrent.futures import ThreadPoolExecutor

    run = str(tmp_path / "race")

    def attempt(_):
        try:
            return reserve(run)
        except FileExistsError:
            return None

    with ThreadPoolExecutor(max_workers=2) as workers:
        claims = list(workers.map(attempt, range(2)))
    assert sum(c is not None for c in claims) == 1
    next(c for c in claims if c is not None).verify()


def test_directory_changed_token_rejected(tmp_path):
    claim = reserve(str(tmp_path / "run"))
    (claim.directory / ".inferssd-owner").write_text('{"token":"other"}')
    with pytest.raises(RuntimeError, match="ownership changed"):
        claim.verify()
