"""Typed production launch and Linux controller factories.

Darwin executes static M1 launches. Online adaptive Mac M2 remains deferred.
"""

from __future__ import annotations

import os
import sys
from collections.abc import Callable, Mapping, Sequence
from functools import partial
from typing import Optional

from inferssd.configuration.settings import Snapshot
from inferssd.engine.core import Engine, NativeEffects
from inferssd.engine.owned import OwnedProcess
from inferssd.launch.live import Host, LiveArtifacts
from inferssd.launch.machine import MachineSource
from inferssd.launch.machine import characteristics as machine_characteristics
from inferssd.launch.prepare import prepare
from inferssd.launch.recorded import run_indexer
from inferssd.launch.resources import probe_paths, server_binary
from inferssd.launch.typed import LaunchSpec, reject_legacy_environment, resolve_typed
from inferssd.launch.types import LaunchInput, Output, Resolution
from inferssd.platform.policy import Policy


class Runtime:
    def __init__(
        self,
        inputs: LaunchInput,
        policy: Policy,
        *,
        host: Host,
        emit: Callable[[Output], None],
        configuration: Optional[LaunchSpec] = None,
        machine: Optional[MachineSource] = None,
    ) -> None:
        """MACHINE is where a launch with a strategy takes the characteristics from."""
        self.inputs, self.policy, self.host, self.emit = inputs, policy, host, emit
        self.configuration = configuration
        self.machine = machine
        self.characteristics_text: Optional[tuple[Optional[str]]] = None

    def characteristics(self, specification: Optional[LaunchSpec]) -> Optional[str]:
        """The machine characteristics for SPECIFICATION's strategy, measured first if needed.

        Measured once per runtime, before the first server is prepared.
        """
        if self.machine is None or specification is None or not specification.strategy:
            return None
        if self.characteristics_text is None:
            platform, repo = self.policy.platform, self.inputs.repo
            resources = specification.resources
            binary = server_binary(platform, specification.binary, resources, repo, os.path.isfile)
            binary = os.path.realpath(binary) if binary else None
            probes = probe_paths(platform, binary, resources, repo, os.path.isfile)
            text = machine_characteristics(self.machine, platform, specification.model, probes)
            self.characteristics_text = (text,)
        return self.characteristics_text[0]

    def resolve(
        self,
        environment: Mapping[str, str],
        args: Sequence[str],
        *,
        directory: str,
        configured: Optional[Snapshot] = None,
        specification: Optional[LaunchSpec] = None,
    ) -> Resolution:
        specification = specification or self.configuration
        if specification is None:
            raise ValueError("production Runtime requires explicit typed LaunchSpec")
        inputs = LaunchInput(
            self.inputs.repo,
            tuple(environment.items()),
            tuple(args),
            self.inputs.profile,
            self.inputs.mode,
        )
        adapter = LiveArtifacts(
            inputs,
            host=self.host,
            run=run_indexer,
            replace=os.replace,
            python=sys.executable,
            machine=self.characteristics(specification),
        )
        value = prepare(
            inputs,
            resolver=partial(
                resolve_typed,
                spec=specification,
                platform=self.policy.platform,
                directory=directory,
                memory=self.policy.memory_budget_bytes,
                configured=configured,
            ),
            observe=adapter,
            emit=self.emit,
            darwin_errors=self.policy.platform == "darwin",
        )
        if value.code != 0:
            raise RuntimeError("native launch preparation failed")
        return value

    def engine(
        self,
        directory: str,
        port: int,
        tier: int,
        cpus: str,
        envs: str,
        args: list[str],
        *,
        configured: Optional[Snapshot] = None,
        specification: Optional[LaunchSpec] = None,
    ) -> Engine:
        if envs:
            raise ValueError("production Engine does not accept legacy environment controls")
        reject_legacy_environment(dict(self.inputs.environment))

        def environment_for(
            environment: Mapping[str, str], _port: int, _envs: str
        ) -> dict[str, str]:
            reject_legacy_environment(environment)
            output = dict(environment)
            return output

        def checked(environment: Mapping[str, str], extra: Sequence[str]) -> Resolution:
            value = self.resolve(
                environment,
                extra,
                directory=directory,
                configured=configured,
                specification=specification,
            )
            ports = [
                value.argv[i + 1] for i, item in enumerate(value.argv[:-1]) if item == "--port"
            ]
            hosts = [
                value.argv[i + 1] for i, item in enumerate(value.argv[:-1]) if item == "--host"
            ]
            if not ports or ports[-1] != str(port) or not hosts or hosts[-1] != "127.0.0.1":
                raise ValueError("owned Engine requires its exact loopback health host and port")
            return value

        # Measure the machine, if it must be, before the run directory exists.
        self.characteristics(specification or self.configuration)
        effects = NativeEffects()

        def own(result: Resolution, path: str) -> OwnedProcess:
            assert effects.reservation is not None
            return OwnedProcess(result, path, self.policy, reservation=effects.reservation)

        return Engine(
            directory,
            port,
            tier,
            cpus,
            envs,
            args,
            environment=dict(self.inputs.environment),
            effects=effects,
            resolve=checked,
            own=own,
            prepare_environment=environment_for,
            check_port=True,
        )
