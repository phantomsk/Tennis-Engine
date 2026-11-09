# Tennis Strategy Engine — Implementation Plan (All-Python)
 
> **Purpose:** a build plan for Claude Code. MVP is a **single-process, all-Python app you can run on a laptop** — MediaPipe for player pose, PyTorch as the one ML framework, OpenCV for video, and Streamlit for the UI. No TypeScript, no REST layer, no queue for MVP.
>
> **Core principle:** *Keep the interfaces, collapse the implementations.* Every network boundary the full system would use (S3, SQS, separate worker, FastAPI + web frontend) is hidden behind an interface. Scaling up = swap the implementation, not rewrite the code.
 
---
 
## 0. Guiding Principles
 
1. **Separate video processing from shot recommendation.** The pipeline produces structured tennis states; the recommendation engine consumes them. Independently testable.
2. **Every stage reads an artifact and writes an artifact** — local files in MVP. Keeps the pipeline resumable and debuggable.
3. **Constrain the input.** MVP assumes single-angle baseline footage, fixed camera, planar court, **short clips (10–30s of rally)**. This keeps synchronous processing fast and keeps MediaPipe pose reliable.
4. **One ML framework: PyTorch.** Do **not** mix PyTorch and TensorFlow — two dependency trees, two CUDA constraint sets, two serialization formats, for no benefit. TensorFlow appears in exactly one optional place (below) and only if you want a specific pretrained weight set.
5. **Two seams make it reversible:** a `Storage` interface (local FS ⇄ S3) and a `JobQueue` interface (in-process ⇄ SQS). Callers only touch these plus job status.
---
 
## 1. The Reversibility Map (read first)
 
Build the left column. Each row can move right independently when a real bottleneck forces it.
 
| Concern | MVP (build now) | Scale-up (swap in later) |
|---|---|---|
| UI | **Streamlit** (all-Python) | Next.js + REST frontend |
| API / orchestration | In-process function calls | FastAPI + SQS / Step Functions |
| Worker | Same process (or subprocess) | Separate ECS/Fargate worker |
| Artifact storage | Local `./artifacts/` | S3 |
| Database | SQLite | RDS PostgreSQL |
| Video upload | Streamlit file uploader → local FS | Presigned S3 PUT |
| Deployment | One box / `streamlit run` | ALB + ECS + RDS |
| GPU/CV work | Runs inline (or subprocess) | Dedicated GPU worker pool |
 
Because callers depend only on `Storage`, `JobQueue`, and job status, they don't care which column you're in.
 
---
 
## 2. Tech Stack (all-Python)
 
| Job | Library | Notes |
|---|---|---|
| Video I/O, frame extraction | **OpenCV** | `cv2.VideoCapture`, frame stepping |
| Court calibration | **OpenCV** | `cv2.findHomography` from 4 manual points |
| Overlays / annotation | **OpenCV** | Draw court, player dots, ball, recommendation |
| Player detection + keypoints | **MediaPipe Pose** | 33 landmarks per player; no training needed. Run per court-half |
| Two-player ID assignment | **Plain Python** | Trivial: exactly 2 players on known court halves |
| Shot type (FH / BH) | **Derived from MediaPipe landmarks** | Wrist-relative-to-body; no separate model for MVP |
| Ball tracking | **PyTorch** (TrackNet-style heatmap) | Over a stack of consecutive frames |
| Trajectory smoothing | **filterpy** (Kalman) + NumPy interpolation | Fills missing detections |
| Contact-event detection | **Plain Python** | Ball direction change near racket-side wrist landmark |
| Decision engine (policy / value / search) | **PyTorch** | The single ML framework |
| State vectors / numerics | **NumPy + pandas + PyArrow** | Persist as Parquet |
| UI (MVP) | **Streamlit** | Upload, scrub, court overlay, recommendation |
| Config / packaging | **Pydantic-settings**, **uv** or **poetry** | Reproducible env |
| Testing / CI | **pytest** + **GitHub Actions** | Runs pipeline on a short sample clip |
| **Optional only** | **TensorFlow / Keras** | ONLY if you want the original pretrained TrackNet weights; otherwise omit entirely |
| **Scale-up additions** | FastAPI, Next.js, S3, SQS/Step Functions, RDS, ECS/Fargate, ALB, Terraform, LocalStack, GPU spot / SageMaker | Added per the reversibility map, not now |
 
### Why not TensorFlow too? (interview-defensible)
The only reason TF shows up in tennis projects is that the original **TrackNet** ball tracker was published in TensorFlow/Keras. PyTorch TrackNet reimplementations exist, so the project stays single-framework. Pull TF in **only** if you want to load the original pretrained checkpoint directly and don't want to port weights — and if you do, isolate it behind the ball-tracker interface so nothing else depends on it.
 
---
 
## 3. Architecture (MVP)
 
```text
┌──────────────────────────────────────────────┐
│  Streamlit app (single Python process)        │
│                                               │
│  UI: upload clip · calibrate court · run ·    │
│      scrub video · view overlay + shot rec    │
│                     │                         │
│                     ▼                         │
│  pipeline.run_stages(job_id):                 │
│     video_loader     (OpenCV)                 │
│     court_calibration(OpenCV homography)      │
│     player_pose      (MediaPipe, per half)    │
│     ball_tracking    (PyTorch TrackNet)       │
│     event_detection  (pose ∩ ball, Python)    │
│     state_builder    (NumPy → Parquet)        │
│     inference        (PyTorch policy+value+search) │
│                     │                         │
│   Storage → ./artifacts/    DB → SQLite       │
└──────────────────────────────────────────────┘
```
 
One process. Streamlit calls the same `Storage` / `JobQueue` seams the scaled version will use behind FastAPI.
 
---
 
## 4. Repository Structure
 
```text
tennis-strategy-engine/
├── app.py                         # Streamlit entrypoint (MVP UI)
├── src/tennis/
│   ├── core/
│   │   ├── storage.py             # Storage interface + LocalStorage (S3Storage later)
│   │   ├── queue.py               # JobQueue interface + InProcessQueue (SqsQueue later)
│   │   └── config.py              # flags select implementations
│   ├── pipeline/
│   │   └── runner.py              # run_stages(): sequential stage loop
│   ├── cv/
│   │   ├── video_loader.py        # OpenCV
│   │   ├── court_calibration.py   # OpenCV homography
│   │   ├── player_pose.py         # MediaPipe Pose, per court-half + 2-player IDs
│   │   ├── ball_tracking.py       # PyTorch TrackNet + Kalman + interpolation
│   │   ├── event_detection.py     # contact events from pose ∩ ball
│   │   └── state_builder.py       # NumPy state vectors → Parquet
│   ├── ml/
│   │   ├── dataset.py
│   │   ├── policy_model.py
│   │   ├── value_model.py
│   │   ├── search.py              # shallow forward search
│   │   ├── train.py
│   │   ├── evaluate.py            # offline eval / metrics
│   │   └── inference.py
│   ├── db/{models,database}.py    # SQLAlchemy + SQLite
│   └── schemas/                   # Pydantic
├── tests/
├── pyproject.toml
└── README.md
```
 
Nothing outside `core/` imports boto3 or a queue lib. `player_pose.py` and `ball_tracking.py` each sit behind a small interface so models can be swapped.
 
---
 
## 5. The Two Seams (build early — they make scaling free)
 
```python
# core/storage.py
class Storage(Protocol):
    def write_artifact(self, job_id: str, stage: str, data: bytes) -> str: ...
    def read_artifact(self, job_id: str, stage: str) -> bytes: ...
    def write_upload(self, job_id: str, data: bytes) -> str: ...
 
class LocalStorage(Storage):
    """./artifacts/{job_id}/{stage}. MVP default."""
 
# class S3Storage(Storage): ...   # scale-up, same interface
 
# core/queue.py
class JobQueue(Protocol):
    def enqueue(self, job_id: str, stage: str) -> None: ...
    def process_next(self) -> None: ...
 
class InProcessQueue(JobQueue):
    """Runs the next stage immediately. MVP default."""
 
# class SqsQueue(JobQueue): ...   # scale-up, same interface
 
# core/config.py
STORAGE = LocalStorage()          # ← swap to S3Storage() later
QUEUE   = InProcessQueue()        # ← swap to SqsQueue() later
```
 
Streamlit and the pipeline do `from tennis.core.config import STORAGE, QUEUE`. Scaling up never touches CV/ML code.
 
---
 
## 6. Phase 1 — Skeleton + Streamlit shell
 
Prove the loop with stub processing before any CV: upload a clip → create a job → run stub stages → show a fake recommendation. Job status is a real state machine in SQLite.
 
**Acceptance:** the Streamlit app takes a clip through every stage status to a (stub) recommendation, with a per-stage progress bar.
 
---
 
## 7. Phase 2 — In-Process Pipeline
 
Lifecycle unchanged from the distributed design; only the advancer differs.
 
```text
UPLOADED → QUEUED → EXTRACTING_FRAMES → POSE_TRACKING →
BALL_TRACKING → DETECTING_EVENTS → BUILDING_STATES →
RUNNING_INFERENCE → COMPLETED        (any stage → FAILED)
```
 
```python
# pipeline/runner.py
STAGES = ["EXTRACTING_FRAMES","POSE_TRACKING","BALL_TRACKING",
          "DETECTING_EVENTS","BUILDING_STATES","RUNNING_INFERENCE"]
 
def run_stages(job_id, progress=None):
    job = load_job(job_id)
    for stage in STAGES:
        if job.stage_completed(stage):      # idempotent + resumable
            continue
        prev = STORAGE.read_artifact(job_id, prev_stage(stage)) if has_prev(stage) else None
        result = run_stage(stage, prev)
        STORAGE.write_artifact(job_id, stage, result)
        update_status(job, stage)
        if progress: progress(stage)
```
 
The loop body is identical to what a distributed worker would run per SQS message — splitting it out later means wrapping each iteration in `QUEUE.enqueue(next_stage)` instead of a `for` loop.
 
**Avoid a frozen UI:** cap MVP clips at 10–30s and show a per-stage progress bar. Reach for `multiprocessing` only if you feel the pain.
 
---
 
## 8. Phase 3 — CV MVP: Manual Court Calibration (OpenCV)
 
User clicks four court corners in the Streamlit UI; compute homography once.
 
```text
(x1,y1)─►(0,0)   (x2,y2)─►(10.97,0)
(x3,y3)─►(0,23.77)   (x4,y4)─►(10.97,23.77)
```
```python
H, _ = cv2.findHomography(image_points, court_points)
```
Encodes the fixed-camera, planar-court assumption. Automatic court detection is MVP 2.
 
---
 
## 9. Phase 4 — Player Pose & Tracking (MediaPipe)
 
MediaPipe Pose gives 33 landmarks per player — position **and** body orientation — with no training.
 
```text
Frame
  → crop to each court half (ROI)          # keeps each player reasonably sized
  → MediaPipe Pose per half → 33 landmarks
  → hip/ankle midpoint (pixels) → Homography → court coords
  → assign Player A / B by court half       # trivial, plain Python
```
 
Output per timestamp:
```json
{ "timestamp":12.53,
  "player_a": {"x":4.3,"y":8.2,"landmarks":[...]},
  "player_b": {"x":7.9,"y":18.4,"landmarks":[...]} }
```
 
**Caveat baked in:** MediaPipe Pose targets a single prominent person and struggles with two small distant figures in a wide shot. The per-court-half ROI crop is the required mitigation, not an optimization.
 
**Bonus the landmarks give you for free:** shot type (FH/BH) is derivable from racket-side wrist position relative to the torso — no separate classifier in MVP.
 
---
 
## 10. Phase 5 — Ball Tracking (PyTorch, independent subsystem)
 
Tennis balls are tiny, fast, motion-blurred; a generic detector fails. Use a **TrackNet-style heatmap model in PyTorch** over a stack of consecutive frames, then smooth.
 
```text
Frame stack → TrackNet heatmap → candidate → Kalman (filterpy) → trajectory
```
```json
{ "timestamp":12.53, "x":5.2, "y":14.1, "confidence":0.91 }
```
Interpolate missing detections — never assume the ball is found every frame. (Heaviest stage; main reason to cap clip length or use a subprocess.)
 
> Only if you want the original pretrained TrackNet weights: load them via TensorFlow **behind this module's interface** so nothing else depends on TF. Otherwise stay pure PyTorch.
 
---
 
## 11. Phase 6 — Contact Events (Python, no extra model)
 
A contact event = ball trajectory direction change **near a player's racket-side wrist landmark**. You already have the wrist (MediaPipe) and the ball (TrackNet) — their intersection in space and time is the event.
 
```json
{ "event_id":101, "timestamp":12.53, "hitter":"player_a",
  "hitter_position":[4.3,8.2], "opponent_position":[7.9,18.4],
  "incoming_ball_position":[4.8,8.5], "shot_type":"forehand" }
```
 
---
 
## 12. Phase 7 — Action Space & State
 
**Actions (9, discretized):**
```text
1 FH Cross Short · 2 FH Cross Deep · 3 FH Down-Line
4 BH Cross Short · 5 BH Cross Deep · 6 BH Down-Line
7 Drop · 8 Lob · 9 Attack Middle
```
 
**State vector** → Parquet in `states/` (your training set):
```python
state = {"hitter_x":4.3,"hitter_y":8.2,"opponent_x":7.9,"opponent_y":18.4,
         "ball_x":4.8,"ball_y":8.5,"opponent_distance_from_center":3.2,
         "open_court_area":0.64,"last_shot_direction":"cross_court",
         "last_shot_depth":"deep","rally_length":7}
```
 
---
 
## 13. Phase 8 — Training-Data Strategy (address FIRST — biggest risk)
 
No ready-made (state → optimal shot) dataset exists. Use **self-supervised from match outcomes**: label *what shot was hit* (pose + events) and *whether the hitter won the point* — the win label is **free** from the rally outcome. Policy net learns the shot distribution skilled players choose; value target is P(win point). State the choice in the README.
 
---
 
## 14. Phase 9 — Policy Network (PyTorch)
 
```python
class ShotPolicyNetwork(nn.Module):
    def __init__(self, input_size, num_actions):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(input_size,128), nn.ReLU(),
            nn.Linear(128,64), nn.ReLU(),
            nn.Linear(64,num_actions))
    def forward(self,x): return self.network(x)
```
 
---
 
## 15. Phase 10 — Value Network + Shallow Search (the real "chess engine")
 
**Value net:** `State + Candidate Action → Q → P(win point)`, trained on rally outcomes.
 
**Shallow forward search** (beats a single-step classifier):
1. For each candidate action X, predict opponent's likely return (policy net as opponent model).
2. Roll forward 1–2 plies over the 9 actions.
3. Score leaf states with the value net; recommend the best backed-up value.
```python
def recommend(state, depth=2):
    return max(ACTIONS, key=lambda a: backed_up_value(step(state,a), depth-1))
```
Depth-2 over 9 actions = 81 leaves; cheap and genuinely engine-like.
 
---
 
## 16. Phase 11 — Evaluation & Metrics (do not skip)
 
Offline eval on a held-out set of matches: **top-K accuracy** vs. what skilled players chose, **win-probability calibration**, strict **held-out match split**. Surface results in the README and a Streamlit metrics tab. A project that measures itself reads as far more senior.
 
---
 
## 17. Artifact Layout (same keys, local now / S3 later)
 
```text
./artifacts/                        s3://tennis-engine/  (scale-up)
├── uploads/match-123.mp4           (same keys)
├── tracks/match-123/pose.json
├── tracks/match-123/ball.json
├── events/match-123/events.json
├── states/match-123/states.parquet
├── recommendations/match-123/recommendations.json
└── models/{policy,value,ball}/v1/model.pt
```
Because `Storage` uses `(job_id, stage)` keys, the same paths work on local FS and S3. No re-encoded videos — decode frames on the fly.
 
Storage split, stated deliberately: **SQLite/Postgres** for job/match metadata, **Parquet** for training state vectors, **JSON** for debuggable pose/ball/event tracks.
 
---
 
## 18. MVP Scope
 
**MVP 1** — Streamlit upload · manual court calibration · MediaPipe pose/tracking (per half) · court coordinates · FH/BH from landmarks · pose∩ball contact events · state vectors · basic recommendation · offline eval. *Single process, local storage, SQLite.*
 
**MVP 2** — automatic court detection · PyTorch TrackNet ball tracking · policy+value trained on real outcomes · depth-2 search · top-K ranking · richer overlay.
 
**MVP 3 (scale-up, per the map)** — swap `S3Storage` + `SqsQueue`, split worker to ECS/Fargate, RDS, FastAPI + Next.js frontend, GPU spot training. *Justified scaling decisions, not upfront.*
 
---
 
## 19. Build Order (hand to Claude Code)
 
```text
1. pyproject env + Storage/JobQueue seams + Streamlit shell + stub pipeline
2. In-process stage runner, idempotent + resumable, per-stage progress bar
3. pytest + CI running the pipeline on a short sample clip
4. Manual court calibration (OpenCV homography) in the Streamlit UI
5. MediaPipe pose per court-half + 2-player ID + projection to court coords
6. FH/BH shot type from landmarks
7. PyTorch TrackNet ball tracking + Kalman + interpolation
8. Contact events from pose ∩ ball
9. State builder → Parquet
10. Training-data strategy + policy net
11. Value net + shallow forward search
12. Evaluation harness + Streamlit metrics tab
13. (When needed) scale-up: swap S3Storage/SqsQueue, FastAPI+frontend, worker split
```
 
**Rationale:** one runnable Streamlit process on a laptop → working end-to-end system → scale up one row of the reversibility map at a time. Interview story: *"All-Python single process behind storage/queue interfaces; MediaPipe gave pose for free so shot-type and contact detection needed no extra models; I split the worker out only when ball-tracking time forced it"* — concrete decisions, not tutorial defaults.
