"""Read native JSONL through a bounded broker stage and use the production consumer."""

from pathlib import Path
import time

from ci.contract import digest, save
from ci.staged_smoke import collect_log


def observe(session, config, model, output, *, platform):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    path = Path(config["path"])
    if not path.is_absolute() or ".." in path.parts or path.suffix != ".jsonl":
        raise ValueError("explicit reviewed native observation path required")
    if platform == "darwin":
        # The Mac broker maps this to its fixed Xcode Python. No extra sandbox
        # executable permission is needed to read the worker's own telemetry.
        argv = ["/usr/bin/python3", "-I", "-c",
                "import pathlib,sys;sys.stdout.buffer.write(pathlib.Path(sys.argv[1]).read_bytes())", str(path)]
    elif platform == "linux":
        argv = ["/usr/bin/cat", str(path)]
    else:
        raise ValueError("explicit supported observation platform required")
    stage = session.start("candidate-observations-" + model, "test",
                          argv, {}, config["cwd"], 30)
    completed = session.wait(stage, deadline=time.monotonic() + 30)
    save(output / "stage.json", completed)
    collect_log(session, stage, output)
    if completed.get("returncode") != 0 or completed.get("cleanup_confirmed") is not True:
        raise RuntimeError("native observation extraction failed")
    events = output / "native-events.jsonl"
    events.write_bytes((output / "stdout.log").read_bytes())
    # These imports belong to the pinned installed controller wheel, never the
    # candidate source tree. A server.log is intentionally absent here.
    from inferssd.telemetry.observations import LocalObservations, NativeObservationSession
    arm = NativeObservationSession(files=LocalObservations()).read_arm_cached(str(output), 0)
    if not arm["groups"] or not arm["trace"] or not arm["prof"]:
        raise ValueError("production observer did not consume native route and profile records")
    receipt = dict(status="passed", source="native-events.jsonl", sha256=digest(events),
                   legacy_log_present=False, groups=len(arm["groups"]), trace=len(arm["trace"]),
                   prof=len(arm["prof"]))
    save(output / "observation.json", receipt)
    return receipt
