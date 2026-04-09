"""Round-trip JSON for pose sequences saved in the dance library."""

from __future__ import annotations

import numpy as np

from app.comparison_types import PoseFrame, PoseSequence
from app.dance_library.codec import pose_sequence_from_jsonable, pose_sequence_to_jsonable


def test_pose_sequence_json_roundtrip() -> None:
    xy = np.random.RandomState(0).randn(33, 2).astype(np.float64) * 0.3
    rel = np.linspace(0.2, 1.0, 33)
    frames = [
        PoseFrame(
            frame_index=i,
            time_sec=i / 25.0,
            image_width=1280,
            image_height=720,
            landmarks_raw=None,
            joints_norm_xy=xy.copy(),
            reliability=rel.copy(),
        )
        for i in range(5)
    ]
    seq = PoseSequence("/tmp/video.mp4", 25.0, frames, 1280, 720)
    d = pose_sequence_to_jsonable(seq)
    back = pose_sequence_from_jsonable(d)
    assert back.fps == seq.fps
    assert back.video_width == seq.video_width
    assert len(back.frames) == len(seq.frames)
    np.testing.assert_allclose(back.frames[2].joints_norm_xy, seq.frames[2].joints_norm_xy)
    np.testing.assert_allclose(back.frames[2].reliability, seq.frames[2].reliability)
