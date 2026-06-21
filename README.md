# Dance Pose Desktop

Local desktop app for drawing a **MediaPipe Pose** skeleton overlay on dance (or any) videos. Runs entirely offline: PySide6 GUI, OpenCV for video I/O, MediaPipe for pose.

## Requirements

- **Python 3.10+** (64-bit recommended)
- Windows, macOS, or Linux
- Webcam-quality videos work best; very dark or tiny figures may lose tracking
- **ffmpeg** (optional but recommended) — required to **merge the original soundtrack** into the processed file after OpenCV writes the silent overlay. Without ffmpeg, processing still completes, but the result is **video-only** and the app shows a warning when “Keep Original Audio” is on.

### ffmpeg install

- **macOS** (Homebrew): `brew install ffmpeg`
- **Windows**: [ffmpeg.org](https://ffmpeg.org/download.html) builds, or `choco install ffmpeg` / `winget install ffmpeg`
- **Linux**: `apt install ffmpeg`, `dnf install ffmpeg`, etc.

`ffmpeg` and `ffprobe` should be on your `PATH` (typical installs do this automatically).

## Setup

```bash
cd dance_pose_desktop
python -m venv .venv
```

**Windows (PowerShell):**

```powershell
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux:**

```bash
source .venv/bin/activate
pip install -r requirements.txt
```

## Run

```bash
python main.py
```

## Frontend background reel

The Frontend can use a looping local MP4 background:

- `app/assets/frontend-dance-bg.mp4` — animated background, used when present.
- `app/assets/frontend-dance-bg.png` — still fallback, used when the MP4 is missing.

Build the MP4 from local videos you have permission to use:

```bash
python tools/build_frontend_background.py ~/Videos/dance-clips \
  --clips-per-video 6 \
  --clip-duration 2 \
  --max-total-duration 60 \
  --fit blur
```

For hand-picked moments, use a manifest:

```json
{
  "videos": [
    {
      "path": "clips/example.mp4",
      "segments": [
        { "start": 12.5, "duration": 2.0 },
        { "start": 43.0, "duration": 2.0 }
      ]
    }
  ]
}
```

Then run:

```bash
python tools/build_frontend_background.py --manifest reel_manifest.json
```

The output is muted and written to `app/assets/frontend-dance-bg.mp4`. Use `--fit blur`
for mixed aspect ratios: the full dance frame is preserved over a blurred 16:9 fill.

## Usage

1. **Load Video** — or drag-and-drop a file onto the window.
2. Choose **Detection mode** (default: **Legacy Single Person**):
   - **Legacy Single Person** — same idea as the original app: MediaPipe is run with **`num_poses=1`**, and the single returned pose is drawn. This can track differently than multi-person mode (useful for side-by-side comparison).
   - **All People** — **`num_poses` up to 6**; draws every pose returned for that frame. In the **preview**, the center-selected person is drawn slightly bolder for readability.
   - **Center Person Only** — multi-person detection, but only the person whose **body center** (hips → shoulders → torso) is closest to the frame center is drawn.
3. The first frame shows **joints and skeleton lines** where the model finds people.
4. **Keep Original Audio** — when checked (default), after the silent overlay is encoded, the app runs **ffmpeg** to mux **video from the processed file** with **audio from the original** (same order as: processed = input 0, source = input 1). Uncheck to keep an intentionally silent processed clip. If **ffmpeg** is missing, or mux fails, you still get the silent OpenCV output and a clear warning.
5. **Process Video** — encodes the full **silent** overlay clip under `temp/` on a **background thread** (GUI stays responsive); if “Keep Original Audio” is on and ffmpeg succeeds, a second file (e.g. `*_with_audio.mp4`) becomes the **final** path used for play and export. The status line may show **sampled people counts** while encoding.
6. **Play Processed Video** — timer-based **silent** frame playback in the preview (click again to stop). The saved file may include audio when merged; use **Export** and open in QuickTime or VLC to hear it.
7. **Export Processed Video** — copy the **final** processed file (with audio if mux succeeded) to a path you choose (defaults under `outputs/`).

### Dance comparison (reference vs performance)

Use the **Dance comparison** group in the main window:

1. **Load Reference Video** / **Load User Video** — pick the choreography reference and your take (paths and basic metadata appear in the panel).
2. **Process Reference** / **Process User** — each runs MediaPipe pose on the full clip on a **background thread** (same model as the overlay app). The pipeline always uses **center-selected** multi-person detection (`num_poses` up to 6) so the main dancer near frame center is scored, independent of the overlay **Detection mode** used for preview/process.
3. **Compare Videos** — runs **Dynamic Time Warping** on fixed-length per-frame feature vectors, then shows **overall %**, **timing**, **arms**, **legs**, **torso/posture**, and a short text summary. **Export scores JSON** saves the breakdown (including per-frame similarity along the DTW path) under a path you choose.

This comparison is **local**, **deterministic**, and **geometry-only** (no learned scorer). See **Scoring (MVP)** below for formulas and tunables.

### Dance library (save reference once, reuse later)

The **Dance library** group supports a reusable flow so the same reference is **not** reprocessed every session:

1. Load and **Process Reference** as usual (pose sequence must exist).
2. **Save Current Reference as Dance** — enter a name. The app writes a folder under `dance_library/<id>/` with **metadata** (`metadata.json`), **pose sequence** (`pose_sequence.json`, normalized joints per frame), a **copy** of the reference video (`reference_video.<ext>` when copying succeeds), an optional **thumbnail** (`thumbnail.jpg`), and updates **`dance_library/index.json`**.
3. Later, pick a dance in **Saved dances** and click **Load Selected Dance** — the reference video and stored pose frames load **without** running pose extraction again (unless the video file is missing).
4. Use **Play reference (practice)** for silent in-window playback of the reference clip. **Mirror reference for practice** flips the reference **horizontally** in the reference preview, practice playback, and the **aligned overlay** so you can follow a coach who faces you.
5. **Mirror reference for scoring** applies a **geometry mirror** to the stored reference pose (swap left/right landmark rows, negate x) **inside the comparison pipeline** so scores match what you see when practice mirroring is on. You can turn either mirror option off per saved dance (defaults are **on**).

**Delete Selected** removes the dance folder and index entry.

See **On-disk library layout** below for the file format.

### Comparison UI (scroll, dual preview, aligned overlay)

- The **main window** uses a **vertical `QScrollArea`** so tall content (controls + previews) fits on smaller displays.
- **Reference** and **User** each get a **side-by-side preview** (frame 0 after load), with clear titles and a short metadata line.
- After **Compare Videos**, an **Aligned overlay view** section appears: **reference BGR × (1 − α) + user BGR × α** with **α = 0.4** on the user layer. Frame pairs come from the **same DTW path** as scoring (uniformly **subsampled** to ~500 steps for responsive seeking). **Play overlay** steps through those pairs at roughly the average of the two clips’ FPS (clamped). No spatial warping—only **temporal** alignment and resize-to-match blending.

### Export timing (FPS)

Processing tries to match the **source frame rate** using OpenCV metadata. If the file has **no usable FPS tag** (common with some MP4s), the app **estimates** FPS from **frame timestamps** while reading, or falls back to **30 fps** and shows a status message. That keeps duration closer to the original than always assuming a fixed rate when the true rate is 25 / 29.97 / 24.

### Skeleton overlay (stabilized full body)

- **Head** — **Four compass points** (N/E/S/W) in a light diamond instead of dense face landmarks. In **video** mode they get light temporal smoothing to cut jitter.
- **Torso / shoulders / hips** — **Strongest** stabilization (lower-pass EMA + longer brief hold when a joint drops) so the core does not twitch.
- **Arms** — **Medium** smoothing; partial arms when wrist/elbow are weak (e.g. shoulder–elbow without a bogus forearm). Brief hold + fainter segments for flickery joints during encode.
- **Legs** — **Medium–strong** policy (same philosophy as before); in **side-on** poses the **weaker** leg may still be **omitted**. Profile / suppression logic is unchanged.
- **Limb drop** — Each arm/leg tracks whether **all** of its core joints (shoulder–elbow–wrist or knee–ankle) have gone **very weak** together. After about **one second** at the clip’s FPS without recovery, that limb is **not drawn** until the detector sees a strong core joint again (avoids forever “ghost” limbs during long occlusions).
- **Preview vs encode** — First-frame **preview** does not use temporal state, so stabilization applies mainly to **full video processing**. **Multi-person** stabilization is keyed by detection **index**, not identity across time.

### In-app playback vs export

- **Exported / final processed file**: frame count matches the decoded source; FPS is whatever was resolved for the writer (see above). With ffmpeg mux, **audio matches the source** stream when copy succeeds (otherwise the app may re-encode audio to AAC). **`-shortest`** is used so length follows the shorter of video/audio and avoids long silent tails.
- **Preview player**: interval is `1000 / (resolved_video_fps × speed_slider)` so 1.00× tracks real time when metadata is accurate. Playback is **video frames only** — **no sound** in the Qt/OpenCV preview even when the saved file has a soundtrack.

### Legacy vs multi-person

- **Legacy Single Person** and the two multi-person modes use the **same `.task` model** but **different `num_poses` settings** and code paths. Legacy does **not** estimate how many people are in frame—only whether one pose was returned.
- **All People** / **Center Person Only** use **`num_poses` = 6**. There is **no per-person tracking ID** across frames: center selection is recomputed **each frame**, which can switch if dancers cross.

## Scoring (MVP)

**Normalization** (`app/normalization.py`): For each frame, the hip midpoint (or shoulder midpoint / torso centroid fallback, matching `body_center_normalized` in `pose_utils`) is moved to the origin. Coordinates are divided by the mean of **shoulder width** and **hip width** in normalized image space (minimum scale clamp ~`0.04`) so size and depth-to-camera differences matter less.

**Features** (`app/sequence_features.py`): One **41-D** vector per frame: **8** joint-chain angles (elbows, knees, shoulder flex, hip flex), **3** posture angles (shoulder line tilt, hip line tilt, torso-axis angle), **11** unit **2D limb direction** pairs (arms, legs, torso, shoulder line, hip line), and **8** **scale-free distances** (wrist/ankle spans, wrist–torso-center, hip–ankle, hand height vs shoulder). Joints below a reliability gate (`0.35`) are treated as missing; their feature entries are NaN with zero weight so occlusions do not zero the whole score.

**Frame blend** (`app/scoring.py`): Group similarities are mapped to **0–100** (Gaussian soft penalties on angle differences; direction similarity via cosine; distance similarity via relative error). Combined frame score =  
`0.40 × angles + 0.30 × directions + 0.20 × distances + 0.10 × posture`  
(constants `WEIGHT_*`).

**DTW** (`app/alignment.py`): Standard `(i−1,j)`, `(i,j−1)`, `(i−1,j−1)` steps. Local cost = pose dissimilarity `100 − combined_frame_score` plus a small **time-sync prior** `λ × 100 × (i/(N−1) − j/(M−1))²` with `λ = _LAMBDA_DTW_TIME_SYNC` (default **0.42**) so that when pose costs tie, paths stay near **equal normalized progress** through both clips (important when lengths differ).

**Outputs**: **Overall** = mean combined similarity over the DTW path. **Arms / legs / torso** = mean similarity restricted to dimension masks for those regions. **Timing** = `100 × (1 − min(1, mean_stretch_error × 2.8))` where `mean_stretch_error` is the mean of `|j − i·(M−1)/(N−1)| / (M−1)` along the path (how well the warp follows a constant speed ratio). **JSON export** includes aligned indices and per-frame scores.

Tuning: edit weights and `_ANGLE_SIGMA_RAD`, `_DIST_SCALE`, `_LAMBDA_DTW_TIME_SYNC`, `_TIMING_LAG_SENSITIVITY`, and `_REL_GATE` in `sequence_features.py` / `scoring.py`.

**Tests** (optional): `pip install pytest` then `pytest tests/test_comparison_sanity.py tests/test_pose_mirror.py tests/test_dance_library_codec.py`.

## Project layout

| Path | Purpose |
|------|---------|
| `main.py` | Application entry |
| `app/window.py` | Main window, controls, playback, comparison UI |
| `app/video_utils.py` | OpenCV capture, metadata, writer |
| `app/pose_utils.py` | MediaPipe Pose + drawing; `pick_landmarks_for_scoring`, reliability helpers |
| `app/worker.py` | `QThread` full-video overlay processing |
| `app/comparison_types.py` | Dataclasses for sequences and scores |
| `app/normalization.py` | Torso-centered skeleton normalization |
| `app/sequence_features.py` | Per-frame feature vectors + group masks |
| `app/alignment.py` | DTW alignment |
| `app/scoring.py` | Frame similarity, aggregation, explanations |
| `app/pose_sequence_extract.py` | Video → `PoseSequence` |
| `app/comparison_worker.py` | `QThread` extract + compare |
| `app/comparison_view.py` | DTW path subsampling + alpha overlay blending |
| `app/ffmpeg_audio.py` | ffmpeg mux: overlay video + original audio |
| `app/ui_utils.py` | BGR → `QPixmap` for preview |
| `tests/` | Sanity tests for comparison math |
| `temp/` | Temporary encoded overlays |
| `outputs/` | Suggested folder for exports |
| `dance_library/` | Local dance library (created at runtime; gitignored by default) |
| `app/dance_library/` | Index, save/load, JSON codec for saved dances |
| `app/pose_mirror.py` | Mirror normalized poses (L/R swap + flip x) for scoring |

### On-disk library layout

All paths are under the project root:

- `dance_library/index.json` — list of dances (`dance_id`, `name`, `created_at`, folder name).
- `dance_library/<dance_id>/metadata.json` — display fields, original path, mirror flags, duration, resolution.
- `dance_library/<dance_id>/pose_sequence.json` — full `PoseSequence` as JSON (no raw MediaPipe objects; `joints_norm_xy` + `reliability` per frame).
- `dance_library/<dance_id>/reference_video.*` — copy of the source file when save succeeded.
- `dance_library/<dance_id>/thumbnail.jpg` — first-frame preview when generation succeeds.

Per-frame **feature vectors** for DTW are **not** stored separately; they are rebuilt from loaded pose frames when you **Compare Videos** (fast compared to video pose extraction).

## First run

The **pose landmarker model** (~5–10 MB) is downloaded automatically into `temp/` the first time pose runs. You need a working internet connection for that first download only.

## Known limitations

- **Comparison vs overlay mode**: Pose **extraction for scoring** always uses **center person** selection; it does not follow **Legacy Single Person** `num_poses=1` tracking. For a clean reference, film one dancer centered.
- **2D only**: Scoring uses **image-plane** landmarks after normalization, not true 3D joint angles.
- **Center switching**: If multiple people move, the “center” dancer index can change frame-to-frame; scores mix identities in that case.
- **Legacy vs multi**: With several dancers, **Legacy** still shows **at most one** skeleton (MediaPipe’s single-pose output), which may not match either “center” or “all” multi-person overlays.
- **Max people per frame** (multi modes): capped at **6**. Larger crowds may not all receive overlays.
- **Center mode**: “Center person” is chosen **per frame** using geometry only, not identity tracking across time.
- **Codec**: OpenCV may pick **mp4v**, **avc1**, or **AVI (MJPG/XVID)** depending on the OS; see logs. ffmpeg mux then **copies video** when possible.
- **Audio**: Only the **first** source audio stream is merged. Exotic multi-track files may need external tools.
- **ffmpeg**: If not installed, enable **Keep Original Audio** still finishes encoding but you only get a **silent** file and a warning dialog.
- **Progress**: If the container does not report frame count, the bar may show an **indeterminate** state while frames are counted during processing.
- **Performance**: Long 4K videos are heavy; preview is scaled, but processing uses full resolution.
- **Close while processing**: The worker is asked to stop; a partial file may remain in `temp/`.

## Troubleshooting

- **Import errors**: Ensure the venv is activated and `pip install -r requirements.txt` completed.
- **Black output video**: Try a shorter clip; confirm the source plays in other apps.
- **No skeleton**: No person detected, poor lighting, or subject too small — processing continues without crashing.
