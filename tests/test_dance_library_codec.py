"""Round-trip JSON for pose sequences saved in the dance library."""

from __future__ import annotations

import numpy as np

from app import video_utils
from app.comparison_types import PoseFrame, PoseSequence
from app.dance_library import load_dance, save_dance_from_reference
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


def test_saved_dance_copies_reference_video_into_library(tmp_path, monkeypatch) -> None:
    source = tmp_path / "reference.mp4"
    source.write_bytes(b"video bytes")
    monkeypatch.setattr(video_utils, "DANCE_LIBRARY_DIR", tmp_path / "library")
    monkeypatch.setattr(video_utils, "read_frame_at_index", lambda *_args, **_kwargs: None)

    frame = PoseFrame(
        frame_index=0,
        time_sec=0.0,
        image_width=640,
        image_height=480,
        landmarks_raw=None,
        joints_norm_xy=np.zeros((33, 2), dtype=np.float64),
        reliability=np.ones(33, dtype=np.float64),
    )
    seq = PoseSequence(str(source), 30.0, [frame], 640, 480)
    meta = video_utils.VideoMetadata(str(source), source.name, 640, 480, 30.0, 1)

    md = save_dance_from_reference(
        name="Whiplash",
        reference_video_path=str(source),
        sequence=seq,
        meta=meta,
    )
    loaded_md, loaded_seq = load_dance(md.dance_id)

    assert loaded_md.video_path.startswith(str(video_utils.DANCE_LIBRARY_DIR))
    assert loaded_md.video_path != str(source)
    assert loaded_seq.source_path == loaded_md.video_path
    assert (tmp_path / "library" / md.dance_id / "reference_video.mp4").read_bytes() == b"video bytes"
