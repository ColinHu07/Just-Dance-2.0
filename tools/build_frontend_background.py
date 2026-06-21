"""Build the looping Frontend background reel from local video files.

This script only uses local files you provide. It does not download videos.
"""

from __future__ import annotations

import argparse
import glob
import json
import subprocess
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.ffmpeg_audio import find_ffmpeg  # noqa: E402


VIDEO_SUFFIXES = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v"}
DEFAULT_OUTPUT = REPO_ROOT / "app" / "assets" / "frontend-dance-bg.mp4"


@dataclass(frozen=True)
class Segment:
    start: float
    duration: float


@dataclass(frozen=True)
class VideoPlan:
    path: Path
    segments: list[Segment]


def _video_duration(path: Path) -> float:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        return 0.0
    try:
        fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frames = float(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0.0)
    finally:
        cap.release()
    if fps <= 0 or frames <= 0:
        return 0.0
    return frames / fps


def _auto_segments(path: Path, *, clips_per_video: int, clip_duration: float) -> list[Segment]:
    duration = _video_duration(path)
    if duration <= 0.0:
        return []
    usable_clip = max(0.3, min(clip_duration, duration))
    max_count = max(1, int(duration // max(0.3, usable_clip)))
    count = max(1, min(clips_per_video, max_count))
    if duration <= usable_clip:
        return [Segment(0.0, usable_clip)]

    margin = min(duration * 0.08, max(1.0, usable_clip))
    first = min(margin, max(0.0, duration - usable_clip))
    last = max(first, duration - usable_clip - margin)
    if count == 1:
        starts = [(first + last) * 0.5]
    else:
        step = (last - first) / (count - 1)
        starts = [first + step * i for i in range(count)]
    return [
        Segment(max(0.0, start), min(usable_clip, max(0.3, duration - start)))
        for start in starts
    ]


def _resolve_input(value: str, *, base_dir: Path) -> list[Path]:
    raw = Path(value).expanduser()
    candidates: list[Path] = []
    if raw.is_absolute():
        path = raw
    else:
        path = (base_dir / raw).resolve()

    if path.is_dir():
        for suffix in VIDEO_SUFFIXES:
            candidates.extend(path.rglob(f"*{suffix}"))
            candidates.extend(path.rglob(f"*{suffix.upper()}"))
        return sorted({p.resolve() for p in candidates if p.is_file()})
    if path.is_file():
        return [path.resolve()] if path.suffix.lower() in VIDEO_SUFFIXES else []

    matches = glob.glob(str(path), recursive=True)
    for match in matches:
        p = Path(match)
        if p.is_file() and p.suffix.lower() in VIDEO_SUFFIXES:
            candidates.append(p.resolve())
    return sorted(set(candidates))


def _segment_from_json(item: Any, *, default_duration: float) -> Segment:
    if isinstance(item, (int, float)):
        return Segment(float(item), default_duration)
    if not isinstance(item, dict):
        raise ValueError(f"Invalid segment item: {item!r}")
    start = float(item.get("start", item.get("start_sec", 0.0)))
    duration = float(item.get("duration", item.get("duration_sec", default_duration)))
    return Segment(max(0.0, start), max(0.3, duration))


def _plans_from_manifest(
    manifest_path: Path,
    *,
    clips_per_video: int,
    clip_duration: float,
) -> list[VideoPlan]:
    data = json.loads(manifest_path.read_text(encoding="utf-8"))
    items = data.get("videos", data) if isinstance(data, dict) else data
    if not isinstance(items, list):
        raise ValueError("Manifest must be a list or an object with a 'videos' list.")
    plans: list[VideoPlan] = []
    for item in items:
        if not isinstance(item, dict) or "path" not in item:
            raise ValueError(f"Invalid manifest entry: {item!r}")
        paths = _resolve_input(str(item["path"]), base_dir=manifest_path.parent)
        if not paths:
            print(f"Skipping missing video: {item['path']}", file=sys.stderr)
            continue
        segments_data = item.get("segments")
        for path in paths:
            if segments_data:
                segments = [
                    _segment_from_json(seg, default_duration=clip_duration)
                    for seg in segments_data
                ]
            else:
                segments = _auto_segments(
                    path,
                    clips_per_video=clips_per_video,
                    clip_duration=clip_duration,
                )
            if segments:
                plans.append(VideoPlan(path=path, segments=segments))
    return plans


def _plans_from_inputs(
    inputs: list[str],
    *,
    clips_per_video: int,
    clip_duration: float,
) -> list[VideoPlan]:
    paths: list[Path] = []
    for value in inputs:
        paths.extend(_resolve_input(value, base_dir=Path.cwd()))
    plans: list[VideoPlan] = []
    for path in sorted(set(paths)):
        segments = _auto_segments(
            path,
            clips_per_video=clips_per_video,
            clip_duration=clip_duration,
        )
        if segments:
            plans.append(VideoPlan(path=path, segments=segments))
        else:
            print(f"Skipping unreadable video: {path}", file=sys.stderr)
    return plans


def _run(cmd: list[str]) -> None:
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        msg = (result.stderr or result.stdout or "").strip()
        raise RuntimeError(msg or f"Command failed: {' '.join(cmd)}")


def _encode_segment(
    *,
    ffmpeg: str,
    source: Path,
    segment: Segment,
    output: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    fit: str,
) -> None:
    if fit == "contain":
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos,"
            f"pad={width}:{height}:(ow-iw)/2:(oh-ih)/2:color=black,"
            f"fps={fps},setsar=1,format=yuv420p"
        )
    elif fit == "blur":
        vf = (
            "split=2[fg][bg];"
            f"[bg]scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},boxblur=24:2,eq=brightness=-0.08:saturation=0.85[bg];"
            f"[fg]scale={width}:{height}:force_original_aspect_ratio=decrease:flags=lanczos[fg];"
            f"[bg][fg]overlay=(W-w)/2:(H-h)/2,"
            f"fps={fps},setsar=1,format=yuv420p"
        )
    else:
        vf = (
            f"scale={width}:{height}:force_original_aspect_ratio=increase:flags=lanczos,"
            f"crop={width}:{height},fps={fps},setsar=1,format=yuv420p"
        )
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-ss",
        f"{segment.start:.3f}",
        "-i",
        str(source),
        "-t",
        f"{segment.duration:.3f}",
        "-vf",
        vf,
        "-an",
        "-c:v",
        "libx264",
        "-preset",
        "veryfast",
        "-crf",
        str(crf),
        "-pix_fmt",
        "yuv420p",
        str(output),
    ]
    try:
        _run(cmd)
    except RuntimeError:
        fallback = cmd.copy()
        i = fallback.index("libx264")
        fallback[i] = "mpeg4"
        _run(fallback)


def _concat_clips(*, ffmpeg: str, clips: list[Path], output: Path, crf: int) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", suffix=".txt", delete=False) as f:
        concat_path = Path(f.name)
        for clip in clips:
            safe = str(clip).replace("'", "'\\''")
            f.write(f"file '{safe}'\n")
    cmd = [
        ffmpeg,
        "-hide_banner",
        "-loglevel",
        "warning",
        "-y",
        "-f",
        "concat",
        "-safe",
        "0",
        "-i",
        str(concat_path),
        "-an",
        "-c",
        "copy",
        "-movflags",
        "+faststart",
        str(output),
    ]
    try:
        _run(cmd)
    except RuntimeError:
        fallback = [
            ffmpeg,
            "-hide_banner",
            "-loglevel",
            "warning",
            "-y",
            "-f",
            "concat",
            "-safe",
            "0",
            "-i",
            str(concat_path),
            "-an",
            "-c:v",
            "libx264",
            "-preset",
            "veryfast",
            "-crf",
            str(crf),
            "-pix_fmt",
            "yuv420p",
            "-movflags",
            "+faststart",
            str(output),
        ]
        _run(fallback)
    finally:
        concat_path.unlink(missing_ok=True)


def build_reel(
    plans: list[VideoPlan],
    *,
    output: Path,
    width: int,
    height: int,
    fps: int,
    crf: int,
    max_total_duration: float,
    fit: str,
) -> int:
    ffmpeg = find_ffmpeg()
    if not ffmpeg:
        raise RuntimeError("ffmpeg was not found. Install ffmpeg or reinstall requirements.")
    selected: list[tuple[Path, Segment]] = []
    running = 0.0
    for plan in plans:
        for segment in plan.segments:
            if max_total_duration > 0 and running >= max_total_duration:
                break
            duration = segment.duration
            if max_total_duration > 0:
                duration = min(duration, max(0.3, max_total_duration - running))
            selected.append((plan.path, Segment(segment.start, duration)))
            running += duration
        if max_total_duration > 0 and running >= max_total_duration:
            break
    if not selected:
        raise RuntimeError("No usable video segments found.")

    with tempfile.TemporaryDirectory(prefix="frontend-bg-") as tmp:
        tmp_dir = Path(tmp)
        clips: list[Path] = []
        for idx, (source, segment) in enumerate(selected, start=1):
            clip = tmp_dir / f"clip_{idx:04d}.mp4"
            print(
                f"[{idx:03d}/{len(selected):03d}] {source.name} "
                f"@ {segment.start:.2f}s for {segment.duration:.2f}s"
            )
            _encode_segment(
                ffmpeg=ffmpeg,
                source=source,
                segment=segment,
                output=clip,
                width=width,
                height=height,
                fps=fps,
                crf=crf,
                fit=fit,
            )
            clips.append(clip)
        _concat_clips(ffmpeg=ffmpeg, clips=clips, output=output, crf=crf)
    return len(selected)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build app/assets/frontend-dance-bg.mp4 from local video clips."
    )
    parser.add_argument(
        "inputs",
        nargs="*",
        help="Video files, directories, or glob patterns. Ignored when --manifest is used.",
    )
    parser.add_argument(
        "--manifest",
        type=Path,
        help="Optional JSON manifest with paths and exact segments.",
    )
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clips-per-video", type=int, default=6)
    parser.add_argument("--clip-duration", type=float, default=2.0)
    parser.add_argument("--max-total-duration", type=float, default=60.0)
    parser.add_argument("--width", type=int, default=1920)
    parser.add_argument("--height", type=int, default=1080)
    parser.add_argument("--fps", type=int, default=24)
    parser.add_argument("--crf", type=int, default=24)
    parser.add_argument(
        "--fit",
        choices=("cover", "contain", "blur"),
        default="cover",
        help="cover crops to fill; contain letterboxes; blur preserves full frame over a blurred fill.",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    clips_per_video = max(1, min(10, int(args.clips_per_video)))
    clip_duration = max(0.3, float(args.clip_duration))
    if args.manifest:
        plans = _plans_from_manifest(
            args.manifest.resolve(),
            clips_per_video=clips_per_video,
            clip_duration=clip_duration,
        )
    else:
        plans = _plans_from_inputs(
            args.inputs,
            clips_per_video=clips_per_video,
            clip_duration=clip_duration,
        )
    if not plans:
        print("No input videos found.", file=sys.stderr)
        return 2
    output = args.output.expanduser()
    if not output.is_absolute():
        output = (Path.cwd() / output).resolve()
    count = build_reel(
        plans,
        output=output,
        width=max(320, int(args.width)),
        height=max(180, int(args.height)),
        fps=max(1, int(args.fps)),
        crf=max(1, min(40, int(args.crf))),
        max_total_duration=max(0.0, float(args.max_total_duration)),
        fit=str(args.fit),
    )
    print(f"\nWrote {output} from {count} clips.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
