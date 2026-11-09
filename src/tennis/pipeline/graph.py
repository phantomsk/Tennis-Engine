"""Declarative stage graph (PLAN.md §7).

Each stage declares the artifacts it reads (`deps`), the file it writes
(`output`, with an explicit extension) and a `version`. The runner re-runs a
stage only when its fingerprint -- a hash of its deps' output hashes, version,
model version and params -- differs from the one its stored output was built
from. Recalibrating therefore re-runs PLAYER_TRACKING and its descendants but
never BALL_TRACKING.

Every stage except PROBE_VIDEO is still a stub; real CV/ML replaces each `run`
in later phases (bump `version` when it does).
"""

import hashlib
import json
from contextlib import AbstractContextManager
from dataclasses import dataclass, field
from graphlib import TopologicalSorter
from pathlib import Path
from typing import Callable, Protocol

from tennis.cv.video_loader import probe_video


class StageContext(Protocol):
    job_id: str
    params: dict

    def read(self, name: str) -> bytes: ...
    def path(self, name: str) -> AbstractContextManager[Path]: ...
    def report(self, fraction: float) -> None: ...


@dataclass(frozen=True)
class StageSpec:
    name: str
    deps: tuple[str, ...]
    output: str
    version: str
    run: Callable[[StageContext], bytes]
    model_version: str = ""
    params: dict = field(default_factory=dict)


# Inputs written by the UI rather than computed by a stage.
SOURCES: dict[str, str] = {
    "UPLOAD": "upload.mp4",
    "CLIP_SETTINGS": "clip.json",  # {"time_scale": 1.0}; >1 for slow-motion clips
    "CALIBRATION": "calibration.json",
}


def _run_probe(ctx: StageContext) -> bytes:
    time_scale = float(json.loads(ctx.read("CLIP_SETTINGS")).get("time_scale", 1.0))
    with ctx.path("UPLOAD") as path:
        return json.dumps(probe_video(path, time_scale=time_scale)).encode()


def _stub(stage: str, extra: dict | None = None) -> Callable[[StageContext], bytes]:
    def run(ctx: StageContext) -> bytes:
        return json.dumps({"stage": stage, "stub": True, **(extra or {})}).encode()

    return run


_SPECS = [
    StageSpec("PROBE_VIDEO", ("UPLOAD", "CLIP_SETTINGS"), "probe.json", "1", _run_probe),
    StageSpec("PLAYER_TRACKING", ("UPLOAD", "CALIBRATION", "PROBE_VIDEO"), "players.json", "0-stub",
              _stub("PLAYER_TRACKING")),
    StageSpec("BALL_TRACKING", ("UPLOAD", "PROBE_VIDEO"), "ball.json", "0-stub", _stub("BALL_TRACKING")),
    StageSpec("DETECTING_EVENTS", ("UPLOAD", "PLAYER_TRACKING", "BALL_TRACKING", "CALIBRATION"), "events.json",
              "0-stub", _stub("DETECTING_EVENTS")),
    StageSpec("DETECTING_BOUNCES", ("BALL_TRACKING", "DETECTING_EVENTS", "CALIBRATION"), "bounces.json", "0-stub",
              _stub("DETECTING_BOUNCES")),
    # Becomes states.parquet when the real state builder lands.
    StageSpec("BUILDING_STATES", ("PLAYER_TRACKING", "DETECTING_EVENTS", "DETECTING_BOUNCES"), "states.json",
              "0-stub", _stub("BUILDING_STATES")),
    StageSpec("RUNNING_INFERENCE", ("BUILDING_STATES",), "recommendations.json", "0-stub",
              _stub("RUNNING_INFERENCE", {"recommendation": {"shot": "CROSS_DEEP", "confidence": 0.42}})),
]

SPECS: dict[str, StageSpec] = {s.name: s for s in _SPECS}

# Raises graphlib.CycleError on import if the graph is ever made cyclic.
ORDER: list[str] = [
    n for n in TopologicalSorter({s.name: s.deps for s in _SPECS}).static_order() if n in SPECS
]


def fingerprint(spec: StageSpec, output_hashes: dict[str, str]) -> str:
    blob = json.dumps(
        {
            "stage": spec.name,
            "version": spec.version,
            "model_version": spec.model_version,
            "params": spec.params,
            "inputs": {dep: output_hashes[dep] for dep in spec.deps},
        },
        sort_keys=True,
    )
    return hashlib.sha256(blob.encode()).hexdigest()


def artifact_name(name: str) -> str:
    return SOURCES[name] if name in SOURCES else SPECS[name].output
