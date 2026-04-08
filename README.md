# Dance Pose Desktop

Local desktop app for drawing a **MediaPipe Pose** skeleton overlay on dance (or any) videos. Runs entirely offline: PySide6 GUI, OpenCV for video I/O, MediaPipe for pose.

## Requirements

- **Python 3.10+** (64-bit recommended)
- Windows, macOS, or Linux
- Webcam-quality videos work best; very dark or tiny figures may lose tracking

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

## Usage

1. **Load Video** — or drag-and-drop a file onto the window.
2. Choose **Detection mode** (default: **Legacy Single Person**):
   - **Legacy Single Person** — same idea as the original app: MediaPipe is run with **`num_poses=1`**, and the single returned pose is drawn. This can track differently than multi-person mode (useful for side-by-side comparison).
   - **All People** — **`num_poses` up to 6**; draws every pose returned for that frame. In the **preview**, the center-selected person is drawn slightly bolder for readability.
   - **Center Person Only** — multi-person detection, but only the person whose **body center** (hips → shoulders → torso) is closest to the frame center is drawn.
3. The first frame shows **joints and skeleton lines** where the model finds people.
4. **Process Video** — encodes the full clip under `temp/` on a **background thread** (GUI stays responsive). The status line may show **sampled people counts** while encoding.
5. **Play Processed Video** — timer-based playback in the preview (click again to stop). Use **Playback speed** (0.25×–2.00×, default 1.00×) to review motion; this affects **only** the in-app timer, not the saved file.
6. **Export Processed Video** — copy the result to a path you choose (defaults under `outputs/`).

### Export timing (FPS)

Processing tries to match the **source frame rate** using OpenCV metadata. If the file has **no usable FPS tag** (common with some MP4s), the app **estimates** FPS from **frame timestamps** while reading, or falls back to **25 fps** and shows a status message. That keeps duration closer to the original than always assuming 25 fps when the true rate is 30 / 29.97 / 24.

### In-app playback vs export

- **Exported / processed file**: frame count matches the decoded source; FPS is whatever was resolved for the writer (see above).
- **Preview player**: interval is `1000 / (resolved_video_fps × speed_slider)` so 1.00× tracks real time when metadata is accurate. Variable speed is **video frames only** — there is **no audio track** in the OpenCV preview path.

### Legacy vs multi-person

- **Legacy Single Person** and the two multi-person modes use the **same `.task` model** but **different `num_poses` settings** and code paths. Legacy does **not** estimate how many people are in frame—only whether one pose was returned.
- **All People** / **Center Person Only** use **`num_poses` = 6**. There is **no per-person tracking ID** across frames: center selection is recomputed **each frame**, which can switch if dancers cross.

## Project layout

| Path | Purpose |
|------|---------|
| `main.py` | Application entry |
| `app/window.py` | Main window, controls, playback |
| `app/video_utils.py` | OpenCV capture, metadata, writer |
| `app/pose_utils.py` | MediaPipe Pose + drawing |
| `app/worker.py` | `QThread` full-video processing |
| `app/ui_utils.py` | BGR → `QPixmap` for preview |
| `temp/` | Temporary encoded overlays |
| `outputs/` | Suggested folder for exports |

## First run

The **pose landmarker model** (~5–10 MB) is downloaded automatically into `temp/` the first time pose runs. You need a working internet connection for that first download only.

## Known limitations

- **Legacy vs multi**: With several dancers, **Legacy** still shows **at most one** skeleton (MediaPipe’s single-pose output), which may not match either “center” or “all” multi-person overlays.
- **Max people per frame** (multi modes): capped at **6**. Larger crowds may not all receive overlays.
- **Center mode**: “Center person” is chosen **per frame** using geometry only, not identity tracking across time.
- **Codec**: Output is **MP4 (mp4v)**; some players prefer H.264 — re-encode externally if needed.
- **Progress**: If the container does not report frame count, the bar may show an **indeterminate** state while frames are counted during processing.
- **Performance**: Long 4K videos are heavy; preview is scaled, but processing uses full resolution.
- **Close while processing**: The worker is asked to stop; a partial file may remain in `temp/`.

## Troubleshooting

- **Import errors**: Ensure the venv is activated and `pip install -r requirements.txt` completed.
- **Black output video**: Try a shorter clip; confirm the source plays in other apps.
- **No skeleton**: No person detected, poor lighting, or subject too small — processing continues without crashing.
