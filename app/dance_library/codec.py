"""JSON codec for pose sequences saved in the dance library."""

from __future__ import annotations

from typing import Any

import numpy as np

from app.comparison_types import PoseFrame, PoseSequence


def pose_sequence_to_jsonable(seq: PoseSequence) -> dict[str, Any]:
    """Convert a PoseSequence into JSON-friendly primitives."""
    return {
        "source_path": seq.source_path,
        "fps": float(seq.fps),
        "video_width": int(seq.video_width),
        "video_height": int(seq.video_height),
        "frames": [
            {
                "frame_index": int(frame.frame_index),
                "time_sec": float(frame.time_sec),
                "image_width": int(frame.image_width),
                "image_height": int(frame.image_height),
                "joints_norm_xy": frame.joints_norm_xy.astype(float).tolist(),
                "reliability": frame.reliability.astype(float).tolist(),
            }
            for frame in seq.frames
        ],
    }


def pose_sequence_from_jsonable(data: dict[str, Any]) -> PoseSequence:
    """Rehydrate a PoseSequence from JSON-friendly primitives."""
    frames = [
        PoseFrame(
            frame_index=int(item["frame_index"]),
            time_sec=float(item["time_sec"]),
            image_width=int(item["image_width"]),
            image_height=int(item["image_height"]),
            landmarks_raw=None,
            joints_norm_xy=np.asarray(item["joints_norm_xy"], dtype=np.float64),
            reliability=np.asarray(item["reliability"], dtype=np.float64),
        )
        for item in data.get("frames", [])
    ]
    return PoseSequence(
        source_path=str(data.get("source_path", "")),
        fps=float(data.get("fps", 30.0)),
        frames=frames,
        video_width=int(data.get("video_width", 0)),
        video_height=int(data.get("video_height", 0)),
    )
