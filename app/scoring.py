"""Geometry-based frame similarity, DTW aggregation, and score breakdown."""

from __future__ import annotations

import math
from typing import Callable, List, Tuple

import numpy as np

from app.alignment import dtw_align
from app.comparison_types import ComparisonResult, FrameFeatures, PoseSequence, ScoreBreakdown
from app.sequence_features import FEATURE_DIM, SL_ANGLES, SL_DIR, SL_DIST, build_frame_features

# --- Tunable weights (frame-level blend); must sum to 1.0 for interpretability.
WEIGHT_ANGLES = 0.40
WEIGHT_DIRECTIONS = 0.30
WEIGHT_DISTANCES = 0.20
WEIGHT_POSTURE = 0.10

# Soft penalties
_ANGLE_SIGMA_RAD = 0.52
_DIST_SCALE = 0.35
_TIMING_LAG_SENSITIVITY = 2.8  # larger → timing score drops faster with normalized lag

# Tiny prior so DTW prefers advancing both clips together when pose costs tie (same pose, different lengths).
_LAMBDA_DTW_TIME_SYNC = 0.42

# Visibility/silhouette mismatch: geometry only compares shared confident joints,
# but persistent extra/missing visible limbs still count as a small pose mismatch.
_VIS_HIDDEN_GATE = 0.35
_VIS_VISIBLE_GATE = 0.70
_VIS_MISMATCH_PENALTY_SCALE = 28.0
_VIS_MISMATCH_MAX_PENALTY = 18.0
_VIS_ARM_JOINTS = (13, 15, 14, 16)  # elbows/wrists
_VIS_LEG_JOINTS = (25, 27, 26, 28)  # knees/ankles
_VIS_COMPARE_JOINTS = _VIS_ARM_JOINTS + _VIS_LEG_JOINTS

# Overall dance scoring blend. Torso/posture matters, but it should not rescue
# a run where the dancer skipped the arm/leg hits or barely moved.
_OVERALL_W_ARMS = 0.28
_OVERALL_W_LEGS = 0.22
_OVERALL_W_TORSO = 0.12
_OVERALL_W_TIMING = 0.16
_OVERALL_W_MOVEMENT = 0.22
_MOTION_DIFF_SIGMA = 0.65
_MOTION_ACTIVITY_FLOOR = 0.035

assert abs(WEIGHT_ANGLES + WEIGHT_DIRECTIONS + WEIGHT_DISTANCES + WEIGHT_POSTURE - 1.0) < 1e-6
assert abs(
    _OVERALL_W_ARMS
    + _OVERALL_W_LEGS
    + _OVERALL_W_TORSO
    + _OVERALL_W_TIMING
    + _OVERALL_W_MOVEMENT
    - 1.0
) < 1e-6


def _angle_diff(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return math.pi
    d = abs(a - b)
    while d > math.pi:
        d = abs(d - 2 * math.pi)
    return min(d, abs(2 * math.pi - d))


def _sim_from_angle_diff(d: float) -> float:
    s = d / max(_ANGLE_SIGMA_RAD, 1e-6)
    return 100.0 * math.exp(-(s * s))


def _sim_from_dist_pair(a: float, b: float) -> float:
    if not (math.isfinite(a) and math.isfinite(b)):
        return 0.0
    denom = max(abs(a), abs(b), 0.12)
    r = abs(a - b) / denom
    return 100.0 * math.exp(-((r / max(_DIST_SCALE, 1e-6)) ** 2))


def _group_similarity(
    va: np.ndarray,
    wa: np.ndarray,
    vb: np.ndarray,
    wb: np.ndarray,
    mask: np.ndarray,
    *,
    is_direction_block: bool,
) -> Tuple[float, float]:
    """Return ``(weighted_sim_sum, weight_sum)`` for dimensions in ``mask``."""
    sim_sum = 0.0
    w_sum = 0.0
    if is_direction_block:
        idxs = np.where(mask)[0]
        i = 0
        while i < len(idxs):
            k = int(idxs[i])
            if k + 1 < FEATURE_DIM and mask[k] and mask[k + 1]:
                wx = min(float(wa[k]), float(wb[k]))
                if wx <= 0 or not (
                    np.isfinite(va[k])
                    and np.isfinite(va[k + 1])
                    and np.isfinite(vb[k])
                    and np.isfinite(vb[k + 1])
                ):
                    i += 2
                    continue
                dot = va[k] * vb[k] + va[k + 1] * vb[k + 1]
                dot = max(-1.0, min(1.0, dot))
                sim = 100.0 * (dot + 1.0) * 0.5
                sim_sum += sim * wx
                w_sum += wx
                i += 2
            else:
                i += 1
        return sim_sum, w_sum

    for k in np.where(mask)[0]:
        wx = min(float(wa[k]), float(wb[k]))
        if wx <= 0:
            continue
        if not (np.isfinite(va[k]) and np.isfinite(vb[k])):
            continue
        if k >= SL_DIST.start:
            sim = _sim_from_dist_pair(float(va[k]), float(vb[k]))
        else:
            sim = _sim_from_angle_diff(_angle_diff(float(va[k]), float(vb[k])))
        sim_sum += sim * wx
        w_sum += wx
    return sim_sum, w_sum


def frame_similarity_parts(
    a: FrameFeatures,
    b: FrameFeatures,
) -> Tuple[float, float, float, float, float]:
    """
    Return ``(angles_sim, dirs_sim, dist_sim, posture_sim, combined)`` using group weights.

    Each group is a 0–100 similarity; ``combined`` blends with ``WEIGHT_*``.
    """
    va, wa = a.vector, a.dim_weight
    vb, wb = b.vector, b.dim_weight
    g_ang = a.group_masks["angles"]
    g_dir = a.group_masks["directions"]
    g_dist = a.group_masks["distances"]
    g_post = a.group_masks["posture"]

    sa, wa_sum = _group_similarity(va, wa, vb, wb, g_ang, is_direction_block=False)
    sd, wd_sum = _group_similarity(va, wa, vb, wb, g_dir, is_direction_block=True)
    st, wt_sum = _group_similarity(va, wa, vb, wb, g_dist, is_direction_block=False)
    sp, wp_sum = _group_similarity(va, wa, vb, wb, g_post, is_direction_block=False)

    def norm(val: float, denom: float) -> float:
        if denom < 1e-8:
            return 100.0
        return val / denom

    sim_a = norm(sa, wa_sum)
    sim_d = norm(sd, wd_sum)
    sim_t = norm(st, wt_sum)
    sim_p = norm(sp, wp_sum)

    comb = (
        WEIGHT_ANGLES * sim_a
        + WEIGHT_DIRECTIONS * sim_d
        + WEIGHT_DISTANCES * sim_t
        + WEIGHT_POSTURE * sim_p
    )
    return sim_a, sim_d, sim_t, sim_p, comb


def frame_dissimilarity(a: FrameFeatures, b: FrameFeatures) -> float:
    """DTW local cost: ``100 - combined_similarity`` (0 = identical pose features)."""
    *_, comb = frame_similarity_parts(a, b)
    return max(0.0, min(100.0, 100.0 - comb))


def _visibility_mismatch_ratios(
    a: FrameFeatures,
    b: FrameFeatures,
    joints: Tuple[int, ...],
) -> Tuple[float, float, float]:
    """
    Return (any_mismatch, user_extra_visible, user_missing_visible) ratios.

    Geometry features already require both clips to have source confidence. This
    separate pass catches persistent silhouette mismatches where one clip clearly
    shows a limb that the other clip does not.
    """
    ra = a.joint_reliability
    rb = b.joint_reliability
    if ra is None or rb is None or not joints:
        return 0.0, 0.0, 0.0

    total = 0
    mismatch = 0
    user_extra = 0
    user_missing = 0
    for idx in joints:
        if idx >= len(ra) or idx >= len(rb):
            continue
        total += 1
        ref_rel = float(ra[idx])
        user_rel = float(rb[idx])
        ref_hidden_user_visible = (
            ref_rel < _VIS_HIDDEN_GATE and user_rel >= _VIS_VISIBLE_GATE
        )
        ref_visible_user_hidden = (
            ref_rel >= _VIS_VISIBLE_GATE and user_rel < _VIS_HIDDEN_GATE
        )
        if ref_hidden_user_visible:
            mismatch += 1
            user_extra += 1
        elif ref_visible_user_hidden:
            mismatch += 1
            user_missing += 1

    if total == 0:
        return 0.0, 0.0, 0.0
    return mismatch / total, user_extra / total, user_missing / total


def _visibility_penalty_from_ratio(ratio: float) -> float:
    return min(_VIS_MISMATCH_MAX_PENALTY, ratio * _VIS_MISMATCH_PENALTY_SCALE)


def _feature_delta(a: FrameFeatures, b: FrameFeatures, mask: np.ndarray) -> Tuple[float, float]:
    """Return weighted per-feature movement between two frames in the same clip."""
    num = 0.0
    den = 0.0
    for k in np.where(mask)[0]:
        wt = min(float(a.dim_weight[k]), float(b.dim_weight[k]))
        if wt <= 0:
            continue
        av = float(a.vector[k])
        bv = float(b.vector[k])
        if not (math.isfinite(av) and math.isfinite(bv)):
            continue
        if SL_ANGLES.start <= k < SL_ANGLES.stop:
            d = _angle_diff(av, bv)
        elif k >= SL_DIST.start:
            denom = max(abs(av), abs(bv), 0.12)
            d = abs(av - bv) / denom
        else:
            d = abs(av - bv)
        num += d * wt
        den += wt
    if den < 1e-8:
        return 0.0, 0.0
    return num / den, den


def _motion_delta_similarity(ref_delta: float, user_delta: float) -> float:
    activity = max(ref_delta, user_delta)
    if activity < _MOTION_ACTIVITY_FLOOR:
        return 100.0
    ratio = abs(ref_delta - user_delta) / max(activity, 1e-6)
    score = 100.0 * math.exp(-((ratio / _MOTION_DIFF_SIGMA) ** 2))
    if ref_delta >= _MOTION_ACTIVITY_FLOOR and user_delta < ref_delta * 0.35:
        score *= max(0.18, user_delta / max(ref_delta, 1e-6))
    return max(0.0, min(100.0, score))


def _path_motion_similarity(
    path: np.ndarray,
    feats_r: List[FrameFeatures],
    feats_u: List[FrameFeatures],
    mask: np.ndarray,
) -> float:
    """
    Compare how much motion happened between aligned frames.

    This catches the common bad case where a static camera performance gets a
    decent pose match because DTW can align many frames to similar standing
    shapes while the reference is actually dancing.
    """
    if path.shape[0] < 2:
        return 100.0
    num = 0.0
    den = 0.0
    active_pairs = 0
    for t in range(1, path.shape[0]):
        r0 = int(path[t - 1, 0])
        u0 = int(path[t - 1, 1])
        r1 = int(path[t, 0])
        u1 = int(path[t, 1])
        ref_delta, ref_w = _feature_delta(feats_r[r0], feats_r[r1], mask)
        user_delta, user_w = _feature_delta(feats_u[u0], feats_u[u1], mask)
        weight = max(ref_delta, user_delta, _MOTION_ACTIVITY_FLOOR) * max(ref_w, user_w)
        if weight <= 1e-8:
            continue
        num += _motion_delta_similarity(ref_delta, user_delta) * weight
        den += weight
        if max(ref_delta, user_delta) >= _MOTION_ACTIVITY_FLOOR:
            active_pairs += 1
    if den < 1e-8 or active_pairs == 0:
        return 100.0
    return num / den


def _mean_visibility_mismatch_ratios(
    path: np.ndarray,
    feats_r: List[FrameFeatures],
    feats_u: List[FrameFeatures],
    joints: Tuple[int, ...],
) -> Tuple[float, float, float]:
    if path.shape[0] == 0:
        return 0.0, 0.0, 0.0
    total_m = 0.0
    total_extra = 0.0
    total_missing = 0.0
    for t in range(path.shape[0]):
        i = int(path[t, 0])
        j = int(path[t, 1])
        m, extra, missing = _visibility_mismatch_ratios(feats_r[i], feats_u[j], joints)
        total_m += m
        total_extra += extra
        total_missing += missing
    n = float(path.shape[0])
    return total_m / n, total_extra / n, total_missing / n


def _make_dtw_local_cost(
    feats_r: List[FrameFeatures],
    feats_u: List[FrameFeatures],
) -> Callable[[int, int], float]:
    n_r = len(feats_r)
    n_u = len(feats_u)
    den_r = max(n_r - 1, 1)
    den_u = max(n_u - 1, 1)

    def local_cost(i: int, j: int) -> float:
        pose = frame_dissimilarity(feats_r[i], feats_u[j])
        tr = i / den_r
        tu = j / den_u
        sync_pen = (tr - tu) ** 2
        return pose + _LAMBDA_DTW_TIME_SYNC * sync_pen * 100.0

    return local_cost


def _mask_mean_similarity(
    path: np.ndarray,
    feats_r: List[FrameFeatures],
    feats_u: List[FrameFeatures],
    mask: np.ndarray,
    *,
    is_direction_block: bool,
) -> float:
    if path.shape[0] == 0:
        return float("nan")
    num = 0.0
    den = 0.0
    for t in range(path.shape[0]):
        i = int(path[t, 0])
        j = int(path[t, 1])
        va, wa = feats_r[i].vector, feats_r[i].dim_weight
        vb, wb = feats_u[j].vector, feats_u[j].dim_weight
        s, w = _group_similarity(va, wa, vb, wb, mask, is_direction_block=is_direction_block)
        num += s
        den += w
    if den < 1e-8:
        return float("nan")
    return num / den


def _region_similarity_path(
    path: np.ndarray,
    feats_r: List[FrameFeatures],
    feats_u: List[FrameFeatures],
    region_mask: np.ndarray,
) -> float:
    """Blend angles (non-dir) + dirs + distances inside ``region_mask``."""
    if path.shape[0] == 0:
        return float("nan")
    ang = region_mask & feats_r[0].group_masks["angles"]
    dist = region_mask & feats_r[0].group_masks["distances"]
    # Directions: take pairs whose both indices are in region_mask
    dir_mask = np.zeros(FEATURE_DIM, dtype=bool)
    gdir = feats_r[0].group_masks["directions"]
    idxs = np.where(gdir & region_mask)[0]
    i = 0
    while i < len(idxs):
        k = int(idxs[i])
        if k + 1 < FEATURE_DIM and (gdir[k] and gdir[k + 1] and region_mask[k] and region_mask[k + 1]):
            dir_mask[k : k + 2] = True
            i += 2
        else:
            i += 1

    num = 0.0
    den = 0.0
    for t in range(path.shape[0]):
        i = int(path[t, 0])
        j = int(path[t, 1])
        va, wa = feats_r[i].vector, feats_r[i].dim_weight
        vb, wb = feats_u[j].vector, feats_u[j].dim_weight
        for m, is_dir in ((ang, False), (dir_mask, True), (dist, False)):
            s, w = _group_similarity(va, wa, vb, wb, m, is_direction_block=is_dir)
            num += s
            den += w
    if den < 1e-8:
        return float("nan")
    return num / den


def _timing_score_from_path(path: np.ndarray, n_ref: int, n_user: int) -> Tuple[float, float, float]:
    """
    Return ``(timing_score_0_100, mean_abs_norm_time_error, mean_abs_frame_lag)``.

    ``mean_abs_norm_time_error`` is mean ``|j/(m-1) - i/(n-1)|`` (legacy diagnostic).

    Timing score uses deviation from the **ideal stretch line** ``j ≈ i * (m-1)/(n-1)``,
    so clips of different lengths still score well when DTW follows a consistent speed ratio.
    """
    if path.shape[0] == 0 or n_ref <= 1 or n_user <= 1:
        return 100.0, 0.0, 0.0
    den_r = max(n_ref - 1, 1)
    den_u = max(n_user - 1, 1)
    ideal_slope = den_u / den_r
    norm_lags: List[float] = []
    frame_lags: List[float] = []
    stretch_err: List[float] = []
    for t in range(path.shape[0]):
        i = int(path[t, 0])
        j = int(path[t, 1])
        tr = i / den_r
        tu = j / den_u
        norm_lags.append(abs(tu - tr))
        j_ideal = i * ideal_slope
        frame_lags.append(abs(j - j_ideal))
        stretch_err.append(abs(j - j_ideal) / den_u)
    mean_norm_lag = float(sum(norm_lags) / len(norm_lags))
    mean_frame_lag = float(sum(frame_lags) / len(frame_lags))
    mean_stretch_err = float(sum(stretch_err) / len(stretch_err))
    score = 100.0 * (1.0 - min(1.0, mean_stretch_err * _TIMING_LAG_SENSITIVITY))
    return score, mean_norm_lag, mean_frame_lag


def _left_right_arm_masks() -> Tuple[np.ndarray, np.ndarray]:
    m_left = np.zeros(FEATURE_DIM, dtype=bool)
    m_right = np.zeros(FEATURE_DIM, dtype=bool)
    m_left[[0, 4, 11, 12, 13, 14, 35, 39]] = True
    m_right[[1, 5, 15, 16, 17, 18, 36, 40]] = True
    return m_left, m_right


def _build_explanation(
    path: np.ndarray,
    feats_r: List[FrameFeatures],
    feats_u: List[FrameFeatures],
    arms: float,
    legs: float,
    torso: float,
    timing: float,
    movement: float,
    visibility_penalty: float = 0.0,
    user_extra_ratio: float = 0.0,
    user_missing_ratio: float = 0.0,
) -> List[str]:
    lines: List[str] = []
    ml, mr = _left_right_arm_masks()
    sl = _region_similarity_path(path, feats_r, feats_u, ml)
    sr = _region_similarity_path(path, feats_r, feats_u, mr)
    if math.isfinite(sl) and math.isfinite(sr):
        if sr < sl - 4.0:
            lines.append(
                "Nice left-side control. The right arm drifted a bit more, "
                "so give that side a sharper finish on the hits."
            )
        elif sl < sr - 4.0:
            lines.append(
                "Your right-side shapes are reading cleaner. Bring the left arm "
                "through with the same intention and the score should climb."
            )
    if math.isfinite(legs) and legs >= 78.0 and movement >= 65.0:
        lines.append(
            "Footwork is a bright spot: your leg shapes matched the reference closely overall."
        )
    elif math.isfinite(legs) and legs < 60.0:
        lines.append(
            "The groove is there, but the lower-body shapes need the next polish pass. "
            "Recheck knee and foot placement on the bigger transitions."
        )
    if math.isfinite(torso) and torso >= 88.0 and movement >= 58.0:
        lines.append(
            "Your posture stayed clean, which makes the whole run easier to read."
        )
    if math.isfinite(torso) and torso < 62.0:
        lines.append(
            "Keep the energy, then clean up the frame: torso and posture are the main places to square up."
        )
    if math.isfinite(timing) and timing < 68.0:
        lines.append(
            "The moves are recognizable; the biggest unlock is timing. "
            "Count into the first beat and try landing each accent a touch closer to the reference."
        )
    elif math.isfinite(timing) and timing >= 85.0 and movement >= 65.0:
        lines.append(
            "Timing felt locked in; your rhythm stayed strong against the reference."
        )
    if math.isfinite(movement) and movement < 58.0:
        lines.append(
            "The shapes had moments, but the camera saw much less movement than the reference. "
            "Commit to the travel, arm hits, and level changes so the score can climb."
        )
    elif math.isfinite(movement) and movement >= 82.0:
        lines.append(
            "Movement energy matched the reference nicely; you kept the run alive instead of marking it."
        )
    if visibility_penalty >= 2.0:
        if user_extra_ratio > user_missing_ratio * 1.25:
            lines.append(
                "Tracking note: your video had extra visible limbs where the reference hid them, "
                "so part of this penalty may be camera angle or occlusion rather than performance."
            )
        elif user_missing_ratio > user_extra_ratio * 1.25:
            lines.append(
                "Tracking note: the reference showed limbs that your clip did not track consistently. "
                "Keep hands and feet fully in frame when you can."
            )
        else:
            lines.append(
                "Tracking note: limb visibility changed between the two clips, so treat that part of the score lightly."
            )
    if not lines:
        lines.append(
            "Solid run. The motion reads close overall; use the category scores to choose the next tiny polish point."
        )
    return lines


def compare_pose_sequences(ref: PoseSequence, user: PoseSequence) -> ComparisonResult:
    feats_r = [build_frame_features(f) for f in ref.frames]
    feats_u = [build_frame_features(f) for f in user.frames]
    return compare_feature_sequences(feats_r, feats_u)


def compare_feature_sequences(
    feats_r: List[FrameFeatures],
    feats_u: List[FrameFeatures],
) -> ComparisonResult:
    n_r = len(feats_r)
    n_u = len(feats_u)
    if n_r == 0 or n_u == 0:
        z = np.zeros(0, dtype=np.float64)
        p = np.zeros((0, 2), dtype=np.int64)
        bd = ScoreBreakdown(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
        return ComparisonResult(
            overall_score=0.0,
            breakdown=bd,
            explanation_lines=["Not enough pose data in one or both videos to compare."],
            per_frame_similarity=z,
            alignment_path=p,
            dtw_total_cost=0.0,
            dtw_mean_cost=0.0,
            timing_mean_abs_lag_frames=0.0,
        )

    path, total_cost = dtw_align(feats_r, feats_u, _make_dtw_local_cost(feats_r, feats_u))
    k = int(path.shape[0])
    mean_cost = float(total_cost / k) if k > 0 else 0.0

    per_frame_sim = np.zeros(k, dtype=np.float64)
    sim_sum = 0.0
    for t in range(k):
        i = int(path[t, 0])
        j = int(path[t, 1])
        *_, comb = frame_similarity_parts(feats_r[i], feats_u[j])
        vis_ratio, _, _ = _visibility_mismatch_ratios(
            feats_r[i], feats_u[j], _VIS_COMPARE_JOINTS
        )
        adjusted = max(0.0, comb - _visibility_penalty_from_ratio(vis_ratio))
        per_frame_sim[t] = adjusted
        sim_sum += adjusted
    frame_overall = float(sim_sum / k) if k > 0 else 0.0

    g = feats_r[0].group_masks
    sim_angles = _mask_mean_similarity(path, feats_r, feats_u, g.get("angles", np.zeros(FEATURE_DIM, dtype=bool)), is_direction_block=False)
    sim_dir = _mask_mean_similarity(path, feats_r, feats_u, g.get("directions", np.zeros(FEATURE_DIM, dtype=bool)), is_direction_block=True)
    sim_dist = _mask_mean_similarity(path, feats_r, feats_u, g.get("distances", np.zeros(FEATURE_DIM, dtype=bool)), is_direction_block=False)
    sim_post = _mask_mean_similarity(path, feats_r, feats_u, g.get("posture", np.zeros(FEATURE_DIM, dtype=bool)), is_direction_block=False)

    arms = _region_similarity_path(path, feats_r, feats_u, g.get("arms", np.zeros(FEATURE_DIM, dtype=bool)))
    legs = _region_similarity_path(path, feats_r, feats_u, g.get("legs", np.zeros(FEATURE_DIM, dtype=bool)))
    torso = _region_similarity_path(path, feats_r, feats_u, g.get("torso", np.zeros(FEATURE_DIM, dtype=bool)))
    visibility_ratio, user_extra_ratio, user_missing_ratio = _mean_visibility_mismatch_ratios(
        path, feats_r, feats_u, _VIS_COMPARE_JOINTS
    )
    arm_visibility_ratio, _, _ = _mean_visibility_mismatch_ratios(
        path, feats_r, feats_u, _VIS_ARM_JOINTS
    )
    leg_visibility_ratio, _, _ = _mean_visibility_mismatch_ratios(
        path, feats_r, feats_u, _VIS_LEG_JOINTS
    )
    if math.isfinite(arms):
        arms = max(0.0, arms - _visibility_penalty_from_ratio(arm_visibility_ratio))
    if math.isfinite(legs):
        legs = max(0.0, legs - _visibility_penalty_from_ratio(leg_visibility_ratio))

    timing, _mean_norm_lag, mean_frame_lag = _timing_score_from_path(
        path, max(n_r, 1), max(n_u, 1)
    )

    def nan_to_num(x: float, fallback: float) -> float:
        return float(x) if math.isfinite(x) else fallback

    motion_mask = (
        g.get("angles", np.zeros(FEATURE_DIM, dtype=bool))
        | g.get("directions", np.zeros(FEATURE_DIM, dtype=bool))
        | g.get("distances", np.zeros(FEATURE_DIM, dtype=bool))
    )
    movement = _path_motion_similarity(path, feats_r, feats_u, motion_mask)

    arms_score = nan_to_num(arms, max(0.0, frame_overall - _visibility_penalty_from_ratio(arm_visibility_ratio)))
    legs_score = nan_to_num(legs, max(0.0, frame_overall - _visibility_penalty_from_ratio(leg_visibility_ratio)))
    torso_score = nan_to_num(torso, frame_overall)
    overall = (
        _OVERALL_W_ARMS * arms_score
        + _OVERALL_W_LEGS * legs_score
        + _OVERALL_W_TORSO * torso_score
        + _OVERALL_W_TIMING * timing
        + _OVERALL_W_MOVEMENT * movement
    )
    overall = max(0.0, min(100.0, overall - 0.35 * _visibility_penalty_from_ratio(visibility_ratio)))
    if movement < 60.0:
        # If the reference has real motion and the user mostly marks/stands,
        # stable torso and clip length should not inflate the final grade.
        overall = min(overall, 34.0 + movement * 0.55)

    bd = ScoreBreakdown(
        overall=overall,
        timing=timing,
        arms=arms_score,
        legs=legs_score,
        torso_posture=torso_score,
        joint_angles=nan_to_num(sim_angles, frame_overall),
        limb_directions=nan_to_num(sim_dir, frame_overall),
        relative_distances=nan_to_num(sim_dist, frame_overall),
        movement=movement,
    )

    expl = _build_explanation(
        path,
        feats_r,
        feats_u,
        bd.arms,
        bd.legs,
        bd.torso_posture,
        timing,
        movement,
        _visibility_penalty_from_ratio(visibility_ratio),
        user_extra_ratio,
        user_missing_ratio,
    )

    return ComparisonResult(
        overall_score=overall,
        breakdown=bd,
        explanation_lines=expl,
        per_frame_similarity=per_frame_sim,
        alignment_path=path,
        dtw_total_cost=total_cost,
        dtw_mean_cost=mean_cost,
        timing_mean_abs_lag_frames=mean_frame_lag,
    )
