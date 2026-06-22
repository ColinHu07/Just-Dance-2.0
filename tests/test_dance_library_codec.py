"""Round-trip JSON for pose sequences saved in the dance library."""

from __future__ import annotations

import numpy as np

from app import video_utils
from app.comparison_types import PoseFrame, PoseSequence
from app.dance_library import (
    append_score_record,
    best_score_for_dance,
    load_dance,
    load_score_history,
    save_dance_from_reference,
    update_dance_metadata,
)
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
        company="SM",
        artist="aespa",
        reference_video_path=str(source),
        sequence=seq,
        meta=meta,
    )
    loaded_md, loaded_seq = load_dance(md.dance_id)

    assert loaded_md.name == "Whiplash"
    assert loaded_md.company == "SM"
    assert loaded_md.artist == "aespa"
    assert loaded_md.video_path.startswith(str(video_utils.DANCE_LIBRARY_DIR))
    assert loaded_md.video_path != str(source)
    assert loaded_seq.source_path == loaded_md.video_path
    assert (tmp_path / "library" / md.dance_id / "reference_video.mp4").read_bytes() == b"video bytes"

    updated = update_dance_metadata(
        md.dance_id,
        name="Drama",
        company="SM",
        artist="aespa",
        mirror_for_practice=False,
        mirror_for_scoring=False,
    )
    assert updated.name == "Drama"
    assert updated.mirror_for_practice is False
    assert updated.mirror_for_scoring is False
    reloaded_md, _ = load_dance(md.dance_id)
    assert reloaded_md.name == "Drama"
    assert reloaded_md.company == "SM"
    assert reloaded_md.artist == "aespa"
    assert reloaded_md.mirror_for_practice is False
    assert reloaded_md.mirror_for_scoring is False


def test_score_history_remembers_best_score(tmp_path, monkeypatch) -> None:
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
        name="Super",
        company="HYBE",
        artist="SEVENTEEN",
        reference_video_path=str(source),
        sequence=seq,
        meta=meta,
    )

    append_score_record(md.dance_id, player_count=1, score=82.5)
    append_score_record(md.dance_id, player_count=1, score=91.0)
    append_score_record(md.dance_id, player_count=2, score=75.0)

    assert len(load_score_history(md.dance_id)) == 3
    assert best_score_for_dance(md.dance_id, player_count=1).score == 91.0
    assert best_score_for_dance(md.dance_id, player_count=2).score == 75.0
