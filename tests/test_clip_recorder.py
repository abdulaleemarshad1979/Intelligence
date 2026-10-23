"""Video clips saved around a confirmed live match."""

import os

import numpy as np

from app.casework.media_store import probe_video
from app.vision.clip_recorder import ClipRecorder


def _frame(i, w=320, h=240):
    f = np.full((h, w, 3), (i * 7) % 255, np.uint8)
    return f


def test_clip_has_pre_and_post_roll_and_reports_once(tmp_path):
    done = []
    rec = ClipRecorder(str(tmp_path), pre_roll_s=2.0, post_roll_s=1.0, on_finished=done.append)
    t = 1000.0
    for i in range(50):                                   # 5 s at 10 fps; only last 2 s are kept as pre-roll
        rec.push("CAM-1", _frame(i), t + i * 0.1)
    rec.start("CAM-1", "ALT-1", t + 5.0)
    for i in range(50, 80):                               # 3 s more, clip closes 1 s after the match
        rec.push("CAM-1", _frame(i), t + i * 0.1)
    assert len(done) == 1
    clip = done[0]
    assert clip["alert_ids"] == ["ALT-1"] and clip["url"].startswith("/data/captures/clip_CAM-1_")
    facts = probe_video(clip["path"])
    assert 2.5 < facts["duration_s"] < 4.5 and facts["width"] == 320
    assert len(clip["sha256"]) == 64


def test_nothing_is_written_without_a_match_and_matches_extend_one_clip(tmp_path):
    done = []
    rec = ClipRecorder(str(tmp_path), pre_roll_s=1.0, post_roll_s=1.0, on_finished=done.append)
    for i in range(40):
        rec.push("CAM-1", _frame(i), 50.0 + i * 0.1)
    assert os.listdir(tmp_path) == [] and not done          # buffering alone saves nothing
    rec.start("CAM-1", "ALT-A", 53.9)
    rec.start("CAM-1", "ALT-B", 54.4)                        # second match on the same camera extends it
    for i in range(40, 70):
        rec.push("CAM-1", _frame(i), 50.0 + i * 0.1)
    assert len(done) == 1 and done[0]["alert_ids"] == ["ALT-A", "ALT-B"]


def test_flush_finishes_open_clip_and_start_needs_buffered_frames(tmp_path):
    done = []
    rec = ClipRecorder(str(tmp_path), on_finished=done.append)
    rec.start("CAM-9", "ALT-X", 1.0)                         # no frames seen: no clip
    rec.flush()
    assert not done
    for i in range(20):
        rec.push("CAM-2", _frame(i), 10 + i * 0.1)
    rec.start("CAM-2", "ALT-Y", 12.0)
    rec.flush()
    assert len(done) == 1 and os.path.isfile(done[0]["path"])
