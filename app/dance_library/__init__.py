"""On-disk dance library helpers."""

from __future__ import annotations

import json
import shutil
import uuid
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import cv2

from app import video_utils
from app.comparison_types import PoseSequence
from app.dance_library.codec import pose_sequence_from_jsonable, pose_sequence_to_jsonable


@dataclass
class DanceMetadata:
    dance_id: str
    name: str
    created_at: str
    folder_name: str
    source_path: str
    video_path: str
    mirror_for_practice: bool = True
    mirror_for_scoring: bool = True
    duration_sec: float = 0.0
    video_width: int = 0
    video_height: int = 0
    fps: float = 30.0


def _root() -> Path:
    video_utils.ensure_app_dirs()
    return video_utils.DANCE_LIBRARY_DIR


def _index_path() -> Path:
    return _root() / "index.json"


def _read_index() -> list[dict[str, Any]]:
    path = _index_path()
    if not path.is_file():
        return []
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if isinstance(data, dict):
        data = data.get("dances", [])
    return data if isinstance(data, list) else []


def _write_index(items: list[dict[str, Any]]) -> None:
    path = _index_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump({"dances": items}, f, indent=2)


def _metadata_from_dict(data: dict[str, Any]) -> DanceMetadata:
    return DanceMetadata(
        dance_id=str(data["dance_id"]),
        name=str(data["name"]),
        created_at=str(data.get("created_at", "")),
        folder_name=str(data.get("folder_name", data["dance_id"])),
        source_path=str(data.get("source_path", "")),
        video_path=str(data.get("video_path", "")),
        mirror_for_practice=bool(data.get("mirror_for_practice", True)),
        mirror_for_scoring=bool(data.get("mirror_for_scoring", True)),
        duration_sec=float(data.get("duration_sec", 0.0)),
        video_width=int(data.get("video_width", 0)),
        video_height=int(data.get("video_height", 0)),
        fps=float(data.get("fps", 30.0)),
    )


def list_dances() -> list[DanceMetadata]:
    dances: list[DanceMetadata] = []
    for item in _read_index():
        try:
            dances.append(_metadata_from_dict(item))
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(dances, key=lambda dance: dance.created_at, reverse=True)


def save_dance_from_reference(
    *,
    name: str,
    reference_video_path: str,
    sequence: PoseSequence,
    meta: video_utils.VideoMetadata,
    mirror_for_practice: bool = True,
    mirror_for_scoring: bool = True,
) -> DanceMetadata:
    src = Path(reference_video_path).expanduser().resolve()
    if not src.is_file():
        raise OSError(f"Reference video missing; cannot save dance:\n{src}")

    dance_id = uuid.uuid4().hex
    folder = _root() / dance_id
    folder.mkdir(parents=True, exist_ok=False)

    copied_video = folder / f"reference_video{src.suffix or '.mp4'}"
    shutil.copy2(src, copied_video)
    video_path = str(copied_video)

    seq_for_disk = PoseSequence(
        source_path=video_path,
        fps=sequence.fps,
        frames=sequence.frames,
        video_width=sequence.video_width,
        video_height=sequence.video_height,
    )
    with (folder / "pose_sequence.json").open("w", encoding="utf-8") as f:
        json.dump(pose_sequence_to_jsonable(seq_for_disk), f)

    first = video_utils.read_frame_at_index(video_path, 0)
    if first is not None:
        cv2.imwrite(str(folder / "thumbnail.jpg"), first)

    md = DanceMetadata(
        dance_id=dance_id,
        name=name,
        created_at=datetime.now(timezone.utc).isoformat(),
        folder_name=dance_id,
        source_path=str(src),
        video_path=video_path,
        mirror_for_practice=mirror_for_practice,
        mirror_for_scoring=mirror_for_scoring,
        duration_sec=meta.duration_sec,
        video_width=meta.width,
        video_height=meta.height,
        fps=meta.fps,
    )
    with (folder / "metadata.json").open("w", encoding="utf-8") as f:
        json.dump(asdict(md), f, indent=2)

    items = [item for item in _read_index() if item.get("dance_id") != dance_id]
    items.append(asdict(md))
    _write_index(items)
    return md


def load_dance(dance_id: str) -> tuple[DanceMetadata, PoseSequence]:
    folder = _root() / dance_id
    md_path = folder / "metadata.json"
    seq_path = folder / "pose_sequence.json"
    if not md_path.is_file() or not seq_path.is_file():
        raise FileNotFoundError(f"Saved dance not found: {dance_id}")
    with md_path.open("r", encoding="utf-8") as f:
        md = _metadata_from_dict(json.load(f))
    with seq_path.open("r", encoding="utf-8") as f:
        seq = pose_sequence_from_jsonable(json.load(f))
    if md.video_path:
        seq.source_path = md.video_path
    return md, seq


def delete_dance(dance_id: str) -> None:
    shutil.rmtree(_root() / dance_id, ignore_errors=True)
    _write_index([item for item in _read_index() if item.get("dance_id") != dance_id])
