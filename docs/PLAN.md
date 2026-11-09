# Tennis Strategy Engine — Implementation Plan (All-Python)

> **Purpose:** a build plan for Claude Code. MVP is a **single-machine, all-Python app you can run on a laptop**. It uses:
> - a person detector plus top-down pose via onnxruntime for players;
> - PyTorch for ball tracking and the deep models;
> - OpenCV for video;
> - Streamlit for the UI.
>
> It has no TypeScript and no REST layer. The MVP queue is a SQLite table drained by one local worker subprocess.
>
> **Core principle:** *Keep the interfaces, collapse the implementations.* Every network boundary the full system would use (S3, SQS, separate worker, FastAPI + web frontend) is hidden behind an interface. Scaling up means swapping the implementation, not rewriting the code.

> **Revision 2 (2026-11-08).** The plan was reviewed for architecture and ML/CV flaws, and seven critical issues were fixed:
> 1. The pipeline blocked the Streamlit UI for minutes. It now runs in a worker subprocess (§7).
> 2. The training labels were assumed to be "free". There is now a staged data strategy (§13).
> 3. The 9 actions couldn't be labeled from the CV output. They are replaced by landing zones derived from detected bounces (§11b, §12).
> 4. The depth-2 search had no transition model and mixed up Q and V. The decision engine is now staged, with explicit definitions (§15).
> 5. Airborne ball and hip positions were projected through the ground homography. Now only ground contacts are projected (§8–§10).
> 6. MediaPipe can't detect the far player. It is replaced by person detection plus top-down pose (§9).
> 7. The linear stage list hid the real dependencies between stages. Stages now form a dependency graph with input fingerprints (§7).
>
> All runtime and accuracy figures are **estimates until measured**. Items marked *verify* need checking before you commit to them.
>
> Revision 2 also takes these lessons from a similar project (Aretes forehand phase detection; see §20):
> - an active-learning labeling loop with a validated auto-accept threshold (§13);
> - a pose-sequence TCN with grammar-constrained Viterbi decoding as the MVP 2 upgrade for event detection (§11c);
> - slow-motion normalization at probe time (§7);
> - timing-error evaluation in milliseconds, where the gap between mean and median signals outliers (§16).

---

## 0. Guiding Principles

1. **Separate video processing from shot recommendation.** The pipeline produces structured tennis states; the recommendation engine consumes them. Each side can be tested on its own.
2. **Every stage reads named artifacts and writes one artifact**, kept as local files in the MVP. Stage dependencies are declared, and a stage only re-runs when its inputs change.
3. **Constrain the input.** MVP assumes single-angle baseline footage from a fixed camera, a planar court, **short clips (10–30 s of rally)**, 30 or 60 fps, and no ultra-wide lens.
4. **One deep-learning framework: PyTorch.** Do **not** mix PyTorch and TensorFlow. Pretrained person and pose models run through **onnxruntime**, which is an inference runtime, not a second training framework. Gradient-boosted tabular models (LightGBM) are fine where the data is small and tabular.
5. **Two seams make it reversible:** a `Storage` interface (local FS ⇄ S3) and a `JobQueue` interface (SQLite table ⇄ SQS). Callers only touch these plus job and stage status.
6. **The UI never blocks on compute.** All pipeline work runs in a worker process; the UI only enqueues jobs and polls their status.
7. **Never project anything above the ground through the court homography.** Only feet and ball bounces get court coordinates.
8. **A heuristic baseline comes first.** Every learned model must beat a rule-based recommender on held-out data before it ships.

---

## 1. The Reversibility Map (read first)

Build the left column. Each row can move right independently when a real bottleneck forces it.

| Concern | MVP (build now) | Scale-up (swap in later) |
|---|---|---|
| UI | **Streamlit** (all-Python) | Next.js + REST frontend |
| API / orchestration | UI enqueues to a SQLite queue table and polls the job row | FastAPI + SQS / Step Functions |
| Worker | One `tennis.worker` subprocess, auto-started by Streamlit | Separate ECS/Fargate worker(s) |
| Artifact storage | Local `./artifacts/` | S3 (same keys) |
| Database | SQLite (WAL mode) | RDS PostgreSQL |
| Video upload | Streamlit file uploader → local FS | Presigned S3 PUT |
| Deployment | One box / `streamlit run` | ALB + ECS + RDS |
| GPU/CV work | Inside the worker process (MPS/CUDA if available) | Dedicated GPU worker pool |

Because callers depend only on `Storage`, `JobQueue`, and job/stage status, they don't care which column you're in.

---

## 2. Tech Stack (all-Python)

| Job | Library | Notes |
|---|---|---|
| Video I/O, metadata probe | **OpenCV** | `cv2.VideoCapture`; use `CAP_PROP_POS_MSEC` for timestamps, because phone video is often variable frame rate |
| Court calibration | **OpenCV** | `cv2.findHomography` least squares over 8 clicked points |
| Overlays / annotation | **OpenCV** | Draw court, player boxes, ball, bounces, recommendation |
| Player detection + tracking | **rtmlib (RTMDet/YOLOX) via onnxruntime** | Person boxes on court-half crops. Foot point = box bottom-centre → homography. Players are matched across frames by court region and a speed limit |
| Keypoints (contact windows only) | **RTMPose (17 COCO keypoints) via onnxruntime** | Top-down on the tracked box, run only around contacts. *Verify weight licenses and opencv dependency pins.* |
| Two-player ID assignment | **Plain Python** | Court-half region + max-speed gating; near/far fixed per rally |
| Shot wing (FH / BH) | **Plain Python** | Which side of the player the ball is on (court coordinates, mirrored for the far player, using per-player handedness). The wrist keypoint confirms it when present |
| Ball tracking | **PyTorch** (TrackNet-style heatmap) | 3-frame stack in, 3 heatmaps out; MPS/CUDA when available |
| Trajectory gap-fill | **NumPy** | Short gaps only, never across a hit or bounce. A constant-velocity Kalman/RTS smoother is about 30 lines (filterpy is unmaintained) |
| Bounce detection | **NumPy** | Piecewise parabola fit between consecutive contacts |
| Decision engine | **Rules → LightGBM → PyTorch** | Staged; see §15 |
| State vectors / numerics | **NumPy + pandas + PyArrow** | Persist as Parquet |
| Schemas | **Pydantic** | One model per artifact, each with a `schema_version` |
| UI (MVP) | **Streamlit** | Upload, calibrate, run, progress, review, recommendation, metrics |
| Config / packaging | **Pydantic-settings**, **uv** | Reproducible env; pin model and library versions |
| Testing / CI | **pytest** + **GitHub Actions** | Unit tests on synthetic data, plus a slow test that runs the pipeline on a short sample clip |
| **Rejected** | MediaPipe Pose | Finds people by their face, so it fails on a 60–100 px far player. It also pulls in `opencv-contrib-python`, which clashes with `opencv-python-headless` |
| **Rejected** | Ultralytics YOLO | AGPL-3.0 |
| **Optional only** | TensorFlow / Keras | ONLY to load the original TrackNet weights, and kept behind the ball-tracker interface |
| **Scale-up additions** | FastAPI, Next.js, S3, SQS/Step Functions, RDS, ECS/Fargate, ALB, Terraform, LocalStack, GPU spot / SageMaker | Added per the reversibility map, not now |

---

## 3. Architecture (MVP)

```text
Streamlit (UI process): upload · calibrate (8 clicks + handedness + "which player am I") ·
                        run · live progress · bounce review · recommendation · metrics
        │  writes sources: upload.mp4, calibration.json   │ polls jobs / job_stages
        ▼                                                  │
pipeline.schedule(job_id) → QUEUE.enqueue(job_id, stage) for each ready + stale stage
        │
        ▼  tennis.worker subprocess (receive → handle_stage → ack)
  UPLOAD ─► PROBE_VIDEO ─┬─► PLAYER_TRACKING ──┐
  CALIBRATION ───────────┘                     ├─► DETECTING_EVENTS ─► DETECTING_BOUNCES ─► BUILDING_STATES ─► RUNNING_INFERENCE
  UPLOAD, PROBE_VIDEO ───► BALL_TRACKING ──────┘   (+ contact-window      (+ BOUNCES_MANUAL
                                                      pose)                 overrides)

Storage → ./artifacts/jobs/{job_id}/      DB → SQLite (jobs, job_stages, queue_messages)
```

Two processes run on one machine. The worker is shaped like the scaled-up worker will be, so splitting it onto ECS later means swapping `SqliteQueue` for `SqsQueue`.

---

## 4. Repository Structure

```text
tennis-strategy-engine/
├── app.py                         # Streamlit entrypoint (MVP UI); enqueues + polls only
├── src/tennis/
│   ├── worker.py                  # receive → handle_stage → ack loop (python -m tennis.worker)
│   ├── core/
│   │   ├── storage.py             # Storage interface + LocalStorage (S3Storage later)
│   │   ├── queue.py               # JobQueue interface + SqliteQueue (SqsQueue later)
│   │   └── config.py              # flags select implementations
│   ├── pipeline/
│   │   ├── graph.py               # StageSpec declarations + graphlib validation
│   │   └── runner.py              # schedule(), handle_stage(), fingerprints
│   ├── cv/
│   │   ├── video_loader.py        # probe + on-the-fly frame iterator
│   │   ├── court_calibration.py   # 8-point homography + validation
│   │   ├── player_tracking.py     # PlayerTracker: detector + court-gated 2-player tracker
│   │   ├── pose.py                # PoseEstimator: RTMPose on a box, used around contacts
│   │   ├── ball_tracking.py       # PyTorch TrackNet + gap-fill
│   │   ├── event_detection.py     # contacts from ball ∩ player box (+ pose)
│   │   ├── bounce_detection.py    # piecewise parabola fit between contacts
│   │   └── state_builder.py       # state vectors → Parquet
│   ├── data/
│   │   ├── mcp_ingest.py          # Match Charting Project → context tables
│   │   └── labeling.py            # gold-set labeling helpers (UI lives in app)
│   ├── ml/
│   │   ├── heuristic.py           # Stage-0 rule-based recommender
│   │   ├── policy_model.py        # behavior policy π(a|s)
│   │   ├── value_model.py         # V(s), p_err, p_win heads
│   │   ├── transition.py          # hand-built T(s,a)
│   │   ├── search.py              # depth-2 expectimax
│   │   ├── train.py
│   │   ├── evaluate.py            # calibration, off-policy value, CV accuracy
│   │   └── inference.py
│   ├── db/{models,database}.py    # SQLAlchemy + SQLite (WAL)
│   └── schemas/                   # Pydantic, one model per artifact
├── tests/
├── pyproject.toml
└── README.md
```

Nothing outside `core/` imports boto3 or a queue library. `player_tracking.py`, `pose.py` and `ball_tracking.py` each sit behind a small interface so the models can be swapped.

---

## 5. The Two Seams (build early — they make scaling free)

```python
# core/storage.py — keys are (job_id, filename); the extension is explicit
class Storage(Protocol):
    def write_artifact(self, job_id: str, name: str, data: bytes) -> str: ...   # atomic: temp + rename
    def read_artifact(self, job_id: str, name: str) -> bytes: ...
    def local_path(self, job_id: str, name: str) -> ContextManager[Path]: ...  # for cv2 (S3: download to temp)

class LocalStorage(Storage):
    """./artifacts/jobs/{job_id}/{name}. MVP default."""

# class S3Storage(Storage): ...   # scale-up, same interface
# Model weights live in a separate ModelStore: models/{kind}/{version}/model.{pt,onnx}

# core/queue.py — SQS semantics: at-least-once delivery, lease = visibility timeout
@dataclass(frozen=True)
class Message: id: str; job_id: str; stage: str; attempts: int

class JobQueue(Protocol):
    def enqueue(self, job_id: str, stage: str) -> None: ...
    def receive(self, visibility_timeout_s: int) -> Message | None: ...
    def ack(self, msg: Message) -> None: ...
    def nack(self, msg: Message, error: str) -> None: ...

class SqliteQueue(JobQueue):
    """queue_messages table, WAL mode, atomic UPDATE…RETURNING lease. MVP default."""

# class SqsQueue(JobQueue): ...   # same interface; extend() added when needed

# core/config.py
STORAGE = LocalStorage()          # ← swap to S3Storage() later
QUEUE   = SqliteQueue()           # ← swap to SqsQueue() later
```

Producers (the UI and `schedule`) only `enqueue`. Only `tennis.worker` calls `receive`. Any message can be delivered twice, so every stage handler must be idempotent (see §7).

---

## 6. Phase 1 — Skeleton + Streamlit shell ✅

Prove the loop with stub processing before any CV: upload a clip → create a job → run stub stages → show a fake recommendation. Job status is a real state machine in SQLite.

---

## 7. Phase 2 — Non-blocking DAG Pipeline

**Execution model.** The pipeline never runs in Streamlit's script thread.
- `ensure_worker()` (wrapped in `st.cache_resource`) spawns one `python -m tennis.worker` subprocess per server. Setting `TENNIS_WORKER=external` turns this off.
- The worker loop: `receive` → `handle_stage` → `ack`.
- If a stage raises, the worker calls `nack`. After 2 attempts it marks the job `FAILED` and stores the error.
- The lease is 30 minutes. If the worker crashes, the message is redelivered when the lease expires.
- A subprocess rather than a thread gives crash isolation from the CV libraries and no GIL shared with the UI.

**Stage graph.** Stages are declared in `pipeline/graph.py`:
```python
@dataclass(frozen=True)
class StageSpec:
    name: str
    deps: tuple[str, ...]
    output: str                 # "players.json" — explicit extension
    version: str                # bump when stage code changes
    run: Callable[[dict[str, bytes], dict], bytes]
    model_version: str = ""
    params: dict = field(default_factory=dict)

SOURCES = {"UPLOAD": "upload.mp4", "CALIBRATION": "calibration.json",
           "BOUNCES_MANUAL": "bounces_manual.json"}            # written by the UI
SPECS = [
  StageSpec("PROBE_VIDEO",       ("UPLOAD",),                                     "probe.json", ...),
  StageSpec("PLAYER_TRACKING",   ("UPLOAD", "CALIBRATION", "PROBE_VIDEO"),        "players.json", ...),
  StageSpec("BALL_TRACKING",     ("UPLOAD", "PROBE_VIDEO"),                       "ball.json", ...),
  StageSpec("DETECTING_EVENTS",  ("UPLOAD", "PLAYER_TRACKING", "BALL_TRACKING", "CALIBRATION"), "events.json", ...),
  StageSpec("DETECTING_BOUNCES", ("BALL_TRACKING", "DETECTING_EVENTS", "CALIBRATION"), "bounces.json", ...),
  StageSpec("BUILDING_STATES",   ("PLAYER_TRACKING", "DETECTING_EVENTS", "DETECTING_BOUNCES"), "states.parquet", ...),
  StageSpec("RUNNING_INFERENCE", ("BUILDING_STATES",),                            "recommendations.json", ...),
]
```
`graphlib.TopologicalSorter` (stdlib) validates the graph and catches cycles. `BOUNCES_MANUAL` is an optional source: when it exists it overrides automatic bounces and joins the `DETECTING_BOUNCES` fingerprint.

**How stages are scheduled and cached:**
- Each stage has an `input_hash = sha256(dep output hashes + version + model_version + params)`.
- After any stage or source artifact changes, `schedule(job_id)` enqueues every stage whose deps are all fresh and whose stored `input_hash` differs from the newly computed one.
- A worker claims a `(job_id, stage)` with a conditional `UPDATE` on `job_stages`. Under duplicate delivery, `rowcount == 0` means another worker already has it, so the duplicate is a no-op.
- A fan-in stage is enqueued by whichever parent finishes last.
- **Recalibrating** re-runs `PLAYER_TRACKING` and everything after it. `BALL_TRACKING` stays cached.
- Bumping a stage's `version` re-runs that stage and its descendants.
- Re-uploading the same video (same content hash) is an instant cache hit.

**DB.**
- `job_stages(job_id, stage, status, input_hash, output_hash, attempts, lease_until, error, progress)` replaces `Job.completed_stages`.
- `queue_messages(id, job_id, stage, visible_at, attempts, acked_at, last_error, created_at)` backs `SqliteQueue`.
- SQLite runs with `PRAGMA journal_mode=WAL` and `busy_timeout=5000`.
- Job status is derived from the stage rows: `FAILED` if any row failed, `COMPLETED` when `RUNNING_INFERENCE` is fresh.

**`PROBE_VIDEO`** outputs metadata only: fps, frame count, resolution, rotation, a variable-frame-rate flag, a **slow-motion flag** and time-scale factor, and a per-frame timestamp table. Frames are always decoded on the fly, and nothing per-frame is stored. Upload rejects clips longer than 30 s of *real* time or above 1080p, and 60 fps input is decimated to 30 fps for ball tracking.

**Slow motion.** Phone slow-mo plays one real second over 3–8 s of video. That breaks every assumption tied to time: TrackNet's 3-frame motion cue, the player speed limits, flight-time thresholds, and any temporal receptive field. Aretes found that a model does *not* learn to generalize across recording speeds on its own.
- Detect slow-mo from the capture frame rate in container metadata (ffprobe), or ask the user at upload when that metadata is missing.
- Downstream stages then work in *real* time. Timestamps are rescaled by the time-scale factor, and frames are subsampled back to an effective 30–60 fps of real time.
- Normalization happens once, in `PROBE_VIDEO`. Stages never handle slow-mo themselves.

**UI.**
- "Run analysis" enqueues and returns immediately.
- `job_id` lives in `st.query_params`, so a refresh, closing the tab, or another device all return to the same job.
- `st.fragment(run_every=2)` renders one status chip per stage (parallel stages side by side), the in-stage `progress` (0–1, written every N frames), and elapsed time.
- A `FAILED` job shows its error and a Retry button. A `COMPLETED` job reads its recommendations from the artifact, so they persist.
- The sidebar lists recent jobs.

**Acceptance:**
- Killing the worker mid-`BALL_TRACKING` resumes the job after the lease expires.
- A stage that raises ends in `FAILED` with the error visible.
- Refreshing mid-run keeps showing progress.
- Recalibrating re-runs `PLAYER_TRACKING` but not `BALL_TRACKING`.

---

## 7b. Performance Budget (estimates — measure with a per-stage benchmark script)

A 30 s clip at 30 fps is 900 frames. Before any mitigations, the estimate on an M-series Mac is **4–16 min CPU-only, or 1.5–4 min with TrackNet on MPS**. Ball tracking dominates. Once ball tracking is on a GPU, player detection becomes the bottleneck.

Mitigations, in priority order:
1. **Worker subprocess + polling UI** (§7). This doesn't shorten compute, but the user is never blocked and can leave and return.
2. **TrackNet on MPS/CUDA**, models loaded once per worker process, plus one warm-up inference at startup (5–10×).
3. **TrackNet 3-in/3-out**, batched 8–16 at a time, fp16 on CUDA (about 3×; needs weights trained that way).
4. **Input-fingerprint caching** (§7): repeat runs and re-calibrations skip unchanged stages.
5. **Normalize to 30 fps** for ball tracking (2× on 60 fps input; also matches TrackNet's training distribution).
6. **Crop TrackNet's input to the court area plus a margin** (1.3–2×). Do not lower the resolution below 288×512: the ball becomes about 1 px and recall collapses.
7. **Player detection at stride 2**, with interpolation (2×). **Pose only within ±0.25 s of contacts** (about 90% fewer pose calls).
8. Later: CoreML/ONNX export of TrackNet (1.5–3×), and pose and ball tracking in parallel once there are multiple workers.

**Target:** a 30 s clip takes **about 30–60 s on an M-series Mac**, running in the background without blocking the UI.

---

## 8. Phase 3 — Manual Court Calibration (OpenCV) ✅ → revise to 8 points

The user clicks **8 points** on the first frame: the 4 singles baseline corners, then the 4 points where the service lines meet the singles sidelines.

Court frame: x runs 0–8.23 m from the left singles sideline (**singles** width; the doubles width of 10.97 m is not used). y runs 0–23.77 m from the near baseline. The net is at y = 11.885.

```text
NL(0,0)      NR(8.23,0)      FL(0,23.77)      FR(8.23,23.77)
SNL(0,5.485) SNR(8.23,5.485) SFL(0,18.285)    SFR(8.23,18.285)
```

Fit by least squares: `H, _ = cv2.findHomography(img_pts, court_pts, 0)`. **Validate before saving:**
- The baseline quad must be convex and correctly ordered (near points below far points, left of right), and `H` must not be `None`.
- RMS reprojection error must be ≤ 3 px. Any point with a residual above 4 px is highlighted for re-clicking.
- The UI overlays **all** court lines (doubles sidelines, centre service line, net) through H⁻¹. These lines were never clicked, so they act as an independent visual check.
- If a sideline bows by more than 4 px, the lens is distorting the image. Reject the clip with the message "disable ultra-wide lens".

The calibration step also records **each player's handedness** (default right) and **which player the user is** (near or far).

`calibration.json` stores the image points, H, the RMS error in px and m, the handedness values, and the user's side. The UI writes it only when these values change.

**H is valid only for points at height 0.** Later: recover the camera pose from H, check it against the net-post tops (1.07 m) and net centre (0.914 m), auto-refine the court lines, and estimate lens distortion.

---

## 9. Phase 4 — Player Detection & Tracking (rtmlib / onnxruntime)

Player position comes from person boxes, not from pose. The far player is 60–100 px tall, which is too small for face-anchored pose detectors.

```text
Frame (stride 2)
  → per half: court-space ROI (net → 5 m behind baseline, ±2 m outside sidelines) → H⁻¹ → image crop (+10%)
  → person detector on crop (input sized so far player ≥ 50 px)
  → foot = box bottom-centre → H → court coords        # never hips: they are ~1 m above the ground
  → keep best box whose foot is inside the ROI and within 8 m/s of the last position
  → 5-frame median; gaps ≤ 0.5 s interpolated (source="interpolated"); longer gaps = None
```

```python
@dataclass(frozen=True)
class PlayerObs:
    box: tuple[float, float, float, float] | None    # xyxy px
    foot_px: tuple[float, float] | None
    foot_court: tuple[float, float] | None           # metres, via H
    keypoints: np.ndarray | None                     # (17,3); filled only in contact windows
    conf: float
    source: Literal["detected", "interpolated", "missed"]

class PlayerTracker(Protocol):
    def track(self, frames: Iterable[tuple[int, np.ndarray]], homography: np.ndarray) -> PlayerTrack: ...

class PoseEstimator(Protocol):
    def keypoints(self, frame: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray | None: ...
```

**Error budget (1σ, after smoothing):**
- Near half: about ±0.15 m.
- Far half: about ±0.2 m sideways, and ±0.4–0.8 m in depth, depending on camera height.

**Gate before building (spike):**
- Use 3 real clips, including one from a phone and one with an adjacent court in view.
- Hand-label foot points every 10th frame, and the hitting wrist at about 15 contacts per clip.
- Compare rtmlib against MediaPipe as a baseline.
- **Pass** = far foot within 0.75 m on at least 90% of labeled frames, near hitting wrist within 0.5× torso length on at least 80% of contacts, and at most 30 s of compute per 30 s clip.
- **Fallbacks:** 2× tiled far crops at 960 px input. If that still fails, use MOG2 background-subtraction blobs for the far player's position, and the ball-side rule for FH/BH.

Handedness comes from the UI toggle. Never trust keypoint left/right labels: the near player is seen from behind and the far player faces the camera.

---

## 10. Phase 5 — Ball Tracking (PyTorch, independent subsystem)

Tennis balls are tiny, fast and motion-blurred, so a generic detector fails. Use a **TrackNet-style heatmap model in PyTorch**: 3 frames in, 3 heatmaps out. Run it on MPS or CUDA, with its input cropped to the court area plus a margin.

```text
Frame stack → TrackNet heatmap → threshold + connected components → candidate (px) → short-gap fill
```

```json
{ "frame":376, "timestamp":12.53, "px":[812,233], "confidence":0.91, "interpolated":false }
```

- **The ball track stays in image pixels.** A ball 1 m up projects about 3 m off its true position, so in-flight court coordinates are deliberately undefined. Only bounces are projected (§11b).
- Never assume the ball is found in every frame. Fill gaps of at most 5 frames, never across a candidate hit or bounce, and flag the filled points.
- Do not skip frames: TrackNet relies on the motion across 3 consecutive frames.
- Public weights were trained on broadcast footage. Plan to fine-tune on about 2–5k labeled frames from your own camera setup. *Verify the license of any PyTorch reimplementation's weights.*

> Only if you want the original pretrained TrackNet weights: load them via TensorFlow **behind this module's interface** so nothing else depends on TF.

---

## 11. Phase 6 — Contact Events (Python + contact-window pose)

**Contact rule:**
- A **contact** is a direction change in the **raw** pixel ball track (never a smoothed one), while the ball is within the hitter's box expanded by 50% (about 1.5 m in court space).
- Hits must alternate between the two players.
- This stage also calls `PoseEstimator` on frames within ±0.25 s of each candidate contact. The wrist farther from the hip midpoint confirms the hitting side and refines contact timing.

**Wing (FH/BH)** is `sign(ball_x − player_x) × s × h`, where:
- `s` is +1 for the near player and −1 for the far player (the far player faces the camera);
- `h` is +1 for a right-hander and −1 for a left-hander.

The incoming ball mostly decides the wing, so wing is a **state** feature, not part of the action. The contact location is the hitter's foot position. A cheap optional extra signal is audio onset of the racket impact, via PyAV.

```json
{ "event_id":101, "frame":376, "timestamp":12.53, "hitter":"player_a",
  "hitter_position":[4.3,8.2], "opponent_position":[7.9,18.4],
  "hitter_wing":"forehand", "confidence":0.88 }
```

## 11b. Phase 6b — Bounce Detection (`DETECTING_BOUNCES`)

Contacts come first, because bounces are searched for between consecutive contacts.

**Algorithm.** For each segment between two contacts (or from the last contact to the end of the clip):
1. Fit the image-space (u, v) track twice: once with one parabola, and once with two parabolas joined at a breakpoint.
2. Fit with RANSAC, and ignore interpolated points.
3. The bounce is the breakpoint when the two-parabola fit improves BIC and the projected point lies in the receiver's half (court + 1 m margin).
4. Refine the bounce point by intersecting lines fitted to the points just before and just after it.
5. Project **only the bounce** through H (height = 0).

```json
{ "bounce_id":7, "after_event_id":101, "frame":402, "timestamp":13.40,
  "court":[2.1,20.3], "confidence":0.83, "source":"auto|manual" }
```

**Low-confidence bounces:**
- If confidence is below 0.6, the shot appears in a Streamlit **review scrubber**, where the user taps the bounce.
- The taps go into `bounces_manual.json`. It overrides the automatic result, and only the downstream stages re-run.
- Manual taps double as ground truth and as training data for a later bounce classifier.

Expected bounce accuracy: about ±0.3 m on the near half, ±0.5–1.0 m on the far half. That is fine for zone labels, but **line calls are out of scope**.

## 11c. MVP 2 upgrade — Learned Event Detection (TCN + grammar decode)

The MVP 1 contact and bounce rules (§11, §11b) are hand-tuned. Once the review scrubber and gold set have produced a few hundred corrected rallies, replace them with a learned sequence labeler. It follows the same recipe that took forehand phase detection to a 16.7 ms median boundary error in Aretes.

**Input.** Per-frame features over the whole clip:
- ball (u, v) in pixels, with velocity and acceleration;
- a ball confidence and a missing flag;
- each player's foot position and box size;
- the distance from the ball to each player's box;
- optional wrist keypoints in contact windows.

**Model.** A small non-causal TCN: 4 residual blocks of dilated 1D convolutions, kernel 5, dilations 1/2/4/8. Size its receptive field to cover one shot-to-bounce-to-shot cycle of about 1.5–2 s of real time.

**Output.** Per-frame classes: `{in_flight_from_near, bounce_far_half, contact_far, in_flight_from_far, bounce_near_half, contact_near, dead_ball}`.

**Training:**
- ordinal or neighbor label smoothing, so frames at a transition get partial credit on both sides;
- inverse-frequency class weights computed on the **train split only**, because contacts and bounces are only about 3% of frames;
- a masked soft cross-entropy loss over padding.

**Decoding.** A Viterbi decoder constrained to the **rally grammar**: `contact → flight → bounce on the receiver's half → contact by the receiver → …`, with exits to `dead_ball`. It finds the single most likely legal sequence, so flicker at transitions disappears and physically impossible sequences, such as two contacts by the same player with no bounce between them, can't occur.
- An event that never appears where the grammar requires one is a strong "needs review" signal, stronger than low confidence.
- Event timestamps come from the decoded boundaries. Bounce frames still go through §11b's line-intersection refinement before projection.

**Gate.** Ship it only if it beats the MVP 1 rules on held-out *players*, measured by contact and bounce timing error in ms (§16).

---

## 12. Phase 7 — Action Space & State

**Actions (8 landing zones):** `CROSS_DEEP · MIDDLE_DEEP · LINE_DEEP · CROSS_SHORT · MIDDLE_SHORT · LINE_SHORT · DROP · LOB`.

Direction is defined relative to the hitter, so it means the same from the deuce side, the ad side, or for a left-hander.

**Labeling rule.** The inputs are the contact (time t_c, hitter position (x_h, y_h)) and the next bounce (time t_b, position (x_b, y_b)). Let:
- c = 4.115 (the centre line);
- s = sign(x_h − c);
- d = |y_b − 11.885| (distance from the net);
- T = t_b − t_c (time from contact to bounce);
- v = ‖(x_b, y_b) − (x_h, y_h)‖ / T (average ground speed).

Apply the first rule that matches:
1. No bounce on the receiver's half before the next contact (a volley or a net ball): label = `None`. Exclude the shot from the policy loss but keep it in the rally.
2. `LOB` if T ≥ 1.8 s.
3. `DROP` if d < 4.5 m and v < 14 m/s.
4. Direction: `MIDDLE` if |x_b − c| < 1.37; else `CROSS` if sign(x_b − c) ≠ s or |x_h − c| < 0.5; else `LINE`.
5. Depth: `SHORT` if d < 6.40 (inside the service line), else `DEEP`.

A ball that lands out keeps its zone label; the outcome records the error. All thresholds are starting values to calibrate against the gold set (§13). Use class weights for the rare short classes.

**State vector** → Parquet. Coordinates are mirrored so the hitter is always at y ≥ 0 on the near side, and x is mirrored for left-handers.
```python
state = {"hitter_x":4.3, "hitter_y":8.2, "opponent_x":7.9, "opponent_y":18.4,
         "hitter_wing":"forehand", "hitter_handedness":"right",
         "incoming_bounce_x":4.8, "incoming_bounce_y":6.1, "incoming_flight_time":0.9,
         "opponent_distance_from_center":3.2, "opponent_at_net":False,
         "open_court_width":2.6,             # lateral distance from the opponent to the far sideline on the open side
         "last_shot_label":"CROSS_DEEP", "rally_length":7,
         "hitter_pos_interpolated":False, "opponent_pos_interpolated":False}
```
Features must tolerate about 1 m of depth error on the far half. No feature may depend on whether the ball landed in or out.

---

## 13. Phase 8 — Training-Data Strategy (decide before any model work)

**The outcome labels are not free.**
- Calling how a point ended (out, net, double bounce) is unreliable near the far baseline, where one pixel is about 15–20 cm.
- A clip holds only 1–2 points.
- CV errors compound: at roughly 85–90% accuracy per stage, joint label accuracy is about 60–70%.

The strategy comes in stages:

- **Stage 0, rule-based.** Use open-court and depth heuristics (§15). No data is needed. This is the baseline every model must beat.
- **Stage 1, prior from the Match Charting Project (MCP).**
  - Parse Jeff Sackmann's MCP: shot-by-shot charting of pro matches, with shot type, direction and outcome (winner, forced error or unforced error).
  - Context = rally-length bucket, incoming direction and type, hitter wing, handedness.
  - Build smoothed tables: π(direction | ctx) and P(hitter wins | ctx, direction).
  - MCP rally shots record direction (and drop/lob) but **not depth** (*verify*). So the prior covers {cross, middle, line, drop, lob}, and short versus deep is learned from your own gold set.
  - License: CC BY-NC-SA 4.0 (*verify*). Attribute it, keep the project non-commercial, and share derived tables and weights under the same license.
  - MCP doesn't depend on the CV pipeline, so it can be built in parallel with it.
- **Stage 2, gold set.**
  - Record about 150 clips of 20 s each with your own fixed camera: roughly 50 minutes of play over 2–3 sessions. Get consent from the people filmed.
  - Label them in a Streamlit UI: confirm the pre-filled action, tap the landing zone, and tap the point winner and how the point ended.
  - That's about 4 hours of work for about 1,000 shots and about 200 points.
  - Split by session: 100 clips for calibration and 50 held out.
  - It also provides ground truth for every CV stage.
- **Active-learning labeling loop** (adapted from Aretes, which labeled 884 of 974 clips with only 153 labeled by hand):
  1. **Seed.** Fully hand-label about 30 varied clips: both court sides, several players, at least one left-hander, at least one slow-mo clip.
  2. **Score.** Run the pipeline on the unlabeled pool and score each clip:
     - **composite confidence** = 0.6 × min + 0.4 × mean over its per-event confidences (contacts, bounces, action labels). The weight on the minimum stops one bad bounce hiding behind ten good ones;
     - a clip where the rally grammar requires an event the pipeline never found (e.g. a contact with no bounce) is flagged first, ahead of low confidence.
  3. **Calibrate.** Each round, review a stratified sample across fixed confidence bands, stratified by player, including some *high*-confidence clips as a calibration check. Choose the auto-accept threshold where reviewed accuracy reaches about 100%.
  4. **Label.** Review clips below the threshold, lowest first. Auto-accept clips above it as pseudo-labels.
  5. **Repeat** for 2–3 rounds within a fixed human budget, for example about 60 clips reviewed by hand.

  Pseudo-labels may be used for training, but **never for evaluation**: the 50 held-out clips are always fully hand-labeled. Use fixed absolute confidence bands, not percentile bands, so rounds are comparable.
- **Weak heuristics** (ball lost from tracking + last hitter) only pre-fill the labeling UI. They are never used as training labels.
- **Self-supervision was considered and rejected for labels.** Temporal-cycle-consistency style clustering separates segments but can't name them, so its clusters can't be anchored to the action or outcome classes without ground truth. That was the Aretes finding too.
- **Stage 3, optional.** Label more footage, or try a simulator, only if the Stage 2 metrics show the prior doesn't transfer to your footage.

**Output semantics (state these in the README).**
- The policy means "what charted pros tend to play in this context, adjusted to the user".
- The value means "the observed hitter win rate after this shot in this context, calibrated".
- Both are **associative, not causal**: pros choose shots partly based on how good the incoming ball is.

---

## 14. Phase 9 — Behavior Policy π(a|s) (PyTorch)

π is the **opponent model and propensity estimator**, not the recommender.
- Model: a small MLP (128-64), alongside a multinomial-logistic baseline.
- Training: initialized from the MCP prior over the shared action subset, then fine-tuned on gold-set states.
- π's probabilities feed both the expectimax weights and the importance weights in off-policy evaluation, so it must be **calibrated**: temperature scaling on a held-out match split, with NLL and ECE reported.
- Top-K accuracy is reported only as a measure of opponent-model quality.

---

## 15. Phase 10 — Decision Engine (staged)

**Definitions:**
- **V(s)** = P(the player to move at s wins the point).
- **Q(s,a)** = p_win(s,a) + (1 − p_win − p_err) · E_{s'~T(s,a)}[1 − V(s')].
  - p_win = an unreturnable ball (winner or forced error).
  - p_err = the hitter misses.
  - The `1 − V` term is the perspective flip, because the opponent is to move at s'.
- The opponent is modeled by **expectimax under π, never minimax**. V scores the leaves.

**Labels:**
- Only the last shot of a rally gets a terminal label: winner, forced error or unforced error. Every earlier shot is "continue".
- V is trained with TD(0): the target is the terminal label if there is one, else 1 − V(s_{t+1}).
- This fixes the credit assignment problem of blaming every shot for the point's outcome.

**Stages.** Each stage ships only if it beats the previous one in §16.
0. **Rule heuristic** (about 50 lines):
   - opponent pulled wide → play to the open side;
   - opponent deep and hitter inside the baseline → drop;
   - hitter defending → deep cross;
   - opponent at the net → lob or line.
1. **Calibrated LightGBM Q** (isotonic calibration) with MCP Beta priors per action class, depth 1, restricted to actions where π(a|s) ≥ ε. This limits recommending shots pros avoid (selection bias).
2. **Hand-built transition model T** with depth-2 expectimax (8 actions → 64 leaves; a few ms when batched):
   - action → landing point ~ N(zone target, Σ fitted from gold bounces);
   - the opponent runs toward the intercept at a fitted max speed, and the hitter recovers toward the centre;
   - p_err and p_win are logistic models on time-to-ball, run distance and margin to the line.
3. **Learned mixture-density T plus TD-trained V.** Only once there are about 10k shots, and validated by recovering the known-best actions on synthetic rallies from stage 2.

```python
def q(s, a, depth):
    pw, pe = p_winner(s, a), p_error(s, a)
    cont = sum(w * (1.0 - v_node(s2, depth - 1)) for s2, w in T(s, a))   # perspective flip
    return pw + (1.0 - pw - pe) * cont

def v_node(s, depth):
    if depth == 0:
        return V(s)                                                       # leaf: state value only
    pi = policy(s); sup = [b for b in ACTIONS if pi[b] > 1e-3]; z = sum(pi[b] for b in sup)
    return sum(pi[b] / z * q(s, b, depth) for b in sup)                    # expectimax

def recommend(s, depth=2, eps=0.05):
    pi = policy(s)
    scored = {a: q(s, a, depth) for a in ACTIONS if pi[a] >= eps}          # support restriction
    return max(scored, key=scored.get), scored                            # scored = the "why"
```

**Explanation.** For each candidate shot, the UI shows p_err, p_win and the continuation value. For example: "cross-deep: 4% error; opponent needs 1.1 s and runs 6 m."

---

## 16. Phase 11 — Evaluation & Metrics (do not skip)

Use a held-out **match/session and player** split (player-disjoint), never a rally split, with confidence intervals from a bootstrap over matches. Players must be disjoint because the model could otherwise learn a player's personal tempo and style instead of the general pattern.

1. **CV accuracy against the gold set:**
   - far and near foot error in meters;
   - ball recall within 5 px;
   - contact F1 within ±2 frames, plus contact and bounce **timing error in ms**. Report the mean and median together: a mean well above the median means a few clips are failing badly (e.g. slow-mo, occlusion), and those clips are worth investigating instead of averaging away;
   - bounce position error in meters;
   - action-label accuracy;
   - FH/BH accuracy.
2. **Calibration** of V, p_err and p_win: Brier score, Brier skill score against the base rate, ECE and reliability plots. Also report Brier on shots three or more from the rally's end, which shows whether V learned anything beyond base rates.
3. **Policy value:** clipped SNIPS (propensity floor 0.05) and the doubly-robust estimate of P(win) under each recommender, with effective sample size. A stage ships if the paired difference against the previous stage is greater than 0 at 90% confidence.
4. **π quality:** NLL, ECE and top-K, with majority-class and stroke-conditioned baselines. Top-3 of 8 is 37.5% by chance.
5. **Simulator check:** the stage-3 model recovers the known-best actions on stage-2 synthetic rallies.

Surface all of this in the README and a Streamlit metrics tab. With about 1,000 gold shots, expect confidence intervals of roughly ±1.5 pp on accuracy, and report them honestly.

---

## 17. Artifact Layout (same keys, local now / S3 later)

```text
./artifacts/                          s3://tennis-engine/ (same keys)
├── jobs/{job_id}/upload.mp4
├── jobs/{job_id}/calibration.json
├── jobs/{job_id}/probe.json
├── jobs/{job_id}/players.json
├── jobs/{job_id}/ball.json
├── jobs/{job_id}/events.json
├── jobs/{job_id}/bounces.json
├── jobs/{job_id}/bounces_manual.json   (optional)
├── jobs/{job_id}/states.parquet
├── jobs/{job_id}/recommendations.json
├── models/{detector,pose,ball,policy,value}/{version}/model.{onnx,pt,txt}
└── data/{mcp,gold}/…
```

- Keys are `(job_id, filename)`, and the extension comes from `StageSpec.output`.
- Each artifact's sha256 is recorded in `job_stages.output_hash`.
- Each artifact header carries `schema_version`, the code version, model versions and library versions.
- No frames or re-encoded videos are stored; frames are decoded on the fly.
- The layout is job-first rather than type-first, so deleting or listing a job is a single prefix operation.

How data is stored, by kind:
- **SQLite/Postgres** for job and stage metadata and the queue.
- **Parquet** for state vectors (and for large player and ball tracks if JSON grows past a few MB).
- **JSON** for debuggable tracks, events and bounces.

---

## 18. MVP Scope

**MVP 1** (single machine: UI process + worker subprocess, local storage, SQLite):
- non-blocking DAG pipeline with live progress;
- 8-point calibration with overlay validation;
- person detection and tracking with foot points;
- pretrained TrackNet ball tracking;
- contacts plus FH/BH;
- bounce detection with a review scrubber;
- state vectors;
- **rule-based recommender**;
- MCP prior tables;
- gold-set labeling UI;
- CV accuracy and calibration metrics.

**MVP 2:**
- calibrated LightGBM Q, then a hand-built transition model with depth-2 expectimax;
- TrackNet fine-tuned on your own footage, and CoreML export;
- learned event detection: a TCN with grammar-constrained Viterbi decoding (§11c);
- automatic court-line refinement;
- off-policy evaluation in the metrics tab.

**MVP 3 (scale-up, per the map):**
- swap in `S3Storage` + `SqsQueue`, run the worker on ECS/Fargate, move to RDS;
- FastAPI + Next.js frontend;
- GPU spot training;
- learned transition model, if the data volume allows.

---

## 19. Build Order (hand to Claude Code)

```text
1.  ✅ pyproject env + Storage/JobQueue seams + Streamlit shell + stub pipeline
2.  ✅ Manual court calibration (4-point) — revise to 8 points + validation overlay + handedness/user-side toggles
3.  DAG runner: StageSpec graph, job_stages + fingerprints, PROBE_VIDEO, explicit artifact names (§7)
4.  SqliteQueue (receive/ack/nack + lease) + tennis.worker subprocess + FAILED/error +
    query-param job URL + polling fragment + recent-jobs list (§7)
5.  pytest + CI on a committed 3–5 s sample clip with hand-labeled ground truth; per-stage benchmark script
6.  MCP ingest → context tables + rule-based recommender (no CV dependency; can run in parallel with 3–5)
7.  Spike: rtmlib vs MediaPipe on 3 real clips against the §9 pass thresholds → pick the player stack
8.  Player detection + tracking + foot projection (§9)
9.  TrackNet ball tracking on MPS/CUDA, 3-in/3-out, pixel-space output (§10)
10. Contact events + contact-window pose + FH/BH (§11)
11. Bounce detection + review scrubber (§11b)
12. State builder → Parquet; action labeler (§12)
13. Gold-set labeling UI + active-learning loop (composite confidence, banded calibration, auto-accept threshold);
    film ~150 clips, hand-label the seed set and the held-out 50 (§13)
14. Behavior policy π + calibrated LightGBM Q (stage 1) (§14–15)
15. Hand-built transition + depth-2 expectimax (stage 2) (§15)
16. Evaluation harness + Streamlit metrics tab (§16)
17. Event-detection TCN + rally-grammar Viterbi decode, gated on beating the §11/§11b rules (§11c)
18. (When needed) scale-up: SqliteQueue → SqsQueue, worker → ECS task, S3Storage, FastAPI+frontend
```

**Rationale:** a non-blocking single-machine app on a laptop → a working end-to-end system with a heuristic baseline → learned models only when they measurably beat it → scale up one row of the reversibility map at a time.

---

## 20. References & Prior Art

- **Aretes: forehand phase detection with active learning** (A. Panda and C. Luong). A pose-sequence TCN (YOLO-pose, 17 keypoints × (x, y, conf), 4 dilated residual blocks with kernel 5) with ordinal label smoothing, inverse-frequency class weights, and a monotonic Viterbi decoder. Active learning with composite confidence (0.6 × min + 0.4 × mean), fixed confidence bands, and an auto-accept threshold validated at 0.55.
  - Results: 884 of 974 clips labeled with 153 labeled by hand; a final 16.7 ms median / 40.1 ms mean boundary error on held-out players; CoreML deployment.
  - Code: <https://github.com/anshulpanda/phase_detection>. Its `pyproject.toml` declares MIT, but the repo has **no LICENSE file**, and it depends on Ultralytics (AGPL-3.0). **Reuse the ideas, not the code**, unless the author confirms the license.
  - Note: the article describes Gaussian ordinal smoothing, but the repo's `build_ordinal_smoothed_targets` implements simple nearest-neighbor smoothing (1 − ε on the true class, ε split between its two neighbors).
  - What carries over to this project: §7 (slow-mo normalization), §11c (TCN + grammar decode), §13 (active-learning loop), §16 (ms timing error, mean/median gap, player-disjoint splits).
  - What doesn't carry over: per-swing technique analysis of a single close-range player. That's a different product, and possibly a later feature for the near player.
- **Match Charting Project** (J. Sackmann): shot-level pro charting data used as the Stage 1 prior (§13). *Verify the license.*
- **TrackNet** (Huang et al., 2019) and its PyTorch reimplementations: ball tracking (§10). *Verify weights and licenses.*
- **rtmlib / RTMPose / RTMDet**: person detection and pose via onnxruntime (§9). *Verify weight licenses.*

**Interview story:**
- "The UI never blocks: one worker subprocess behind an SQS-shaped queue interface, and the stage graph caches by input fingerprint."
- "Pose detectors failed on the far player, so position comes from person boxes and pose runs only around contacts."
- "The ball is projected only at bounces, because the homography only holds on the ground."
- "The engine is expectimax over a hand-built transition model, and each stage had to beat a rule-based baseline on calibrated, off-policy metrics."
