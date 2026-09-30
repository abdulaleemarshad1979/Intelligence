"""
app/features/skeleton_gait.py

Skeleton (body-proportion) + gait signature engine.

Input : per-person sequences of 2D keypoints in COCO-17 order, shape (T, 17, 3) = (x, y, conf),
        from any pose model (YOLOv8/11-pose, RTMPose, ViTPose) run on tracked person crops.
Output: GaitSignature per time window, GaitProfile per enrolled person, MatchResult per comparison.

Two kinds of evidence:
  1. Skeleton   - bone-length ratios (leg/torso, thigh/shin, arm/leg, ...). Scale-free, so they work
                  at any distance and even when the person is standing still.
  2. Gait       - cadence, step asymmetry (limp), stride, knee range of motion, arm swing,
                  arm-leg coupling, vertical bob, trunk lean, plus a phase-normalised joint-angle
                  template for one gait cycle.

All sigmas / thresholds marked "calibrate" must be tuned on footage from your own cameras.
"""
from __future__ import annotations

import math
from collections import defaultdict, deque
from dataclasses import dataclass, field

import numpy as np

# ---------------------------------------------------------------- COCO-17 indices
NOSE = 0
LSH, RSH, LEL, REL, LWR, RWR = 5, 6, 7, 8, 9, 10
LHIP, RHIP, LKN, RKN, LAN, RAN = 11, 12, 13, 14, 15, 16

VIEW_BINS = ("frontal", "oblique", "side")
VIEW_FACTOR = (1.0, 0.75, 0.5)  # reliability multiplier by view-bin distance


@dataclass
class GaitConfig:
    kp_conf: float = 0.35            # keypoints below this are treated as missing
    max_gap_frames: int = 4          # interpolate gaps up to this length
    smooth_frames: int = 3
    min_duration_s: float = 2.5      # need ~2 full gait cycles
    min_cycle_s: float = 0.55        # children walk fast
    max_cycle_s: float = 2.0         # elderly / crowded shuffling
    min_periodicity: float = 0.30    # autocorrelation peak needed to call it walking
    min_cycles: int = 2
    template_len: int = 32
    view_frontal: float = 0.55       # shoulder_w / torso_len above this -> frontal   (calibrate)
    view_side: float = 0.30          # below this -> side                            (calibrate)
    bone_percentile: float = 85.0    # foreshortening only shortens bones -> use a high percentile
    skeleton_veto_sigmas: float = 4.0


# feature -> (sigma, weight, views where the 2D measurement is meaningful)   sigmas: calibrate
FEATURES = {
    # skeleton proportions
    "leg_torso":        (0.08, 2.0, VIEW_BINS),
    "thigh_shin":       (0.08, 1.0, VIEW_BINS),
    "arm_leg":          (0.06, 1.0, ("side", "oblique")),
    "upper_fore":       (0.10, 0.5, ("side", "oblique")),
    "shoulder_hip":     (0.10, 1.0, ("frontal", "oblique")),
    # gait dynamics
    "cadence":          (8.0,  2.0, VIEW_BINS),      # steps / minute
    "step_asym":        (0.06, 1.5, VIEW_BINS),      # limp indicator
    "vertical_bob":     (0.02, 1.0, VIEW_BINS),
    "arm_leg_coupling": (0.30, 1.0, VIEW_BINS),      # low when carrying a bag / child
    "stride_amp":       (0.12, 1.5, ("side", "oblique")),
    "knee_rom":         (10.0, 1.5, ("side", "oblique")),
    "arm_swing":        (0.15, 1.0, ("side", "oblique")),
    "trunk_lean":       (4.0,  1.0, ("side",)),
}
SKELETON_KEYS = ("leg_torso", "thigh_shin", "arm_leg", "upper_fore", "shoulder_hip")


@dataclass
class GaitSignature:
    view: str
    features: dict
    template: np.ndarray | None      # (4, template_len) hip swing L/R, knee flex L/R
    n_cycles: int
    quality: float                   # 0..1
    duration_s: float
    deep_embedding: np.ndarray | None = None   # optional: OpenGait/GaitBase silhouette embedding


@dataclass
class GaitProfile:
    per_view: dict                   # view -> aggregated GaitSignature
    pooled: dict                     # view-independent features, median over all windows
    n_windows: int


@dataclass
class MatchResult:
    score: float                     # 0..1
    reliability: float               # 0..1, how much to trust the score
    veto: bool                       # skeleton proportions are physically incompatible
    used_view: str
    breakdown: dict = field(default_factory=dict)


# ================================================================ low-level helpers
def _fill_short_gaps(x: np.ndarray, max_gap: int) -> np.ndarray:
    x = x.copy()
    n = len(x)
    good = ~np.isnan(x)
    if good.sum() < 2:
        return x
    idx = np.arange(n)
    filled = np.interp(idx, idx[good], x[good])
    i = 0
    while i < n:
        if np.isnan(x[i]):
            j = i
            while j < n and np.isnan(x[j]):
                j += 1
            if i > 0 and j < n and (j - i) <= max_gap:
                x[i:j] = filled[i:j]
            i = j
        else:
            i += 1
    return x


def _nan_smooth(x: np.ndarray, w: int) -> np.ndarray:
    k = np.ones(w)
    flat = x.reshape(x.shape[0], -1)
    out = np.full_like(flat, np.nan)
    for i in range(flat.shape[1]):
        v = flat[:, i]
        m = ~np.isnan(v)
        num = np.convolve(np.where(m, v, 0.0), k, "same")
        den = np.convolve(m.astype(float), k, "same")
        out[:, i] = np.where(m & (den > 0), num / np.maximum(den, 1e-9), np.nan)
    return out.reshape(x.shape)


def _moving_avg(x: np.ndarray, w: int) -> np.ndarray:
    if w <= 1:
        return x
    k = np.ones(w) / w
    pad = w // 2
    xp = np.pad(x, (pad, w - 1 - pad), mode="edge")
    return np.convolve(xp, k, "valid")


def _full(sig: np.ndarray, min_valid: float = 0.8) -> np.ndarray | None:
    m = ~np.isnan(sig)
    if m.mean() < min_valid or m.sum() < 2:
        return None
    i = np.arange(len(sig))
    return np.interp(i, i[m], sig[m])


def _mid(xy, a, b):
    return (xy[:, a] + xy[:, b]) / 2.0


def _len(xy, a, b):
    return np.linalg.norm(xy[:, a] - xy[:, b], axis=-1)


def _pct(x: np.ndarray, q: float) -> float:
    x = x[~np.isnan(x)]
    return float(np.percentile(x, q)) if len(x) >= 5 else float("nan")


def _pair(a: float, b: float) -> float:
    vals = [v for v in (a, b) if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


def _angle(a, b, c) -> np.ndarray:
    """Angle at joint b in degrees, arrays (T, 2)."""
    v1, v2 = a - b, c - b
    cos = np.sum(v1 * v2, -1) / (np.linalg.norm(v1, axis=-1) * np.linalg.norm(v2, axis=-1) + 1e-9)
    return np.degrees(np.arccos(np.clip(cos, -1.0, 1.0)))


# ================================================================ extraction
def _clean(kpts: np.ndarray, cfg: GaitConfig):
    xy = kpts[..., :2].astype(float).copy()
    conf = kpts[..., 2].astype(float)
    xy[conf < cfg.kp_conf] = np.nan
    for j in range(xy.shape[1]):
        for c in range(2):
            xy[:, j, c] = _fill_short_gaps(xy[:, j, c], cfg.max_gap_frames)
    if cfg.smooth_frames > 1:
        xy = _nan_smooth(xy, cfg.smooth_frames)
    return xy, conf


def _skeleton(xy: np.ndarray, cfg: GaitConfig):
    q = cfg.bone_percentile
    L = lambda a, b: _pct(_len(xy, a, b), q)
    thigh = _pair(L(LHIP, LKN), L(RHIP, RKN))
    shin = _pair(L(LKN, LAN), L(RKN, RAN))
    upper = _pair(L(LSH, LEL), L(RSH, REL))
    fore = _pair(L(LEL, LWR), L(REL, RWR))
    torso = _pair(L(LSH, LHIP), L(RSH, RHIP))
    sh_w = _pct(_len(xy, LSH, RSH), 50)
    hip_w = _pct(_len(xy, LHIP, RHIP), 50)
    leg, arm = thigh + shin, upper + fore
    feats = {
        "leg_torso": leg / torso,
        "thigh_shin": thigh / shin,
        "arm_leg": arm / leg,
        "upper_fore": upper / fore,
        "shoulder_hip": sh_w / hip_w,
    }
    dims = {"leg": leg, "arm": arm, "thigh": thigh, "torso": torso}
    return feats, dims


def _view(xy: np.ndarray, cfg: GaitConfig) -> str:
    torso = np.nanmean(np.stack([_len(xy, LSH, LHIP), _len(xy, RSH, RHIP)]), axis=0)
    r = _pct(_len(xy, LSH, RSH) / np.maximum(torso, 1e-6), 50)
    if not np.isfinite(r):
        return "oblique"
    if r >= cfg.view_frontal:
        return "frontal"
    if r <= cfg.view_side:
        return "side"
    return "oblique"


def _period(sig: np.ndarray, fps: float, cfg: GaitConfig):
    s = sig - sig.mean()
    n = len(s)
    ac = np.correlate(s, s, "full")[n - 1:]
    ac = ac / (ac[0] + 1e-9) * n / np.maximum(n - np.arange(n), 1)   # unbiased
    lo = max(2, int(cfg.min_cycle_s * fps))
    hi = min(int(cfg.max_cycle_s * fps), int(n * 0.6))
    if hi <= lo + 2:
        return None, 0.0
    seg = ac[lo:hi]
    peaks = [i for i in range(1, len(seg) - 1) if seg[i] >= seg[i - 1] and seg[i] >= seg[i + 1]]
    if not peaks:
        return None, 0.0
    best = max(seg[i] for i in peaks)
    k = next(i for i in peaks if seg[i] >= 0.8 * best)   # prefer fundamental over harmonics
    return (lo + k) / fps, float(min(seg[k], 1.0))


def _gait(xy, conf, fps, dims, cfg: GaitConfig):
    leg, arm, thigh = dims["leg"], dims["arm"], dims["thigh"]
    if not (np.isfinite(leg) and leg > 0):
        return None
    hip = _mid(xy, LHIP, RHIP)
    sh = _mid(xy, LSH, RSH)
    ok = ~np.isnan(hip).any(axis=1)
    if ok.sum() < cfg.min_duration_s * fps * 0.8:
        return None

    # walking direction in the image (falls back to vertical for toward/away-from-camera walking)
    t = np.arange(len(xy)) / fps
    vx = np.polyfit(t[ok], hip[ok, 0], 1)[0]
    vy = np.polyfit(t[ok], hip[ok, 1], 1)[0]
    speed = math.hypot(vx, vy) / leg
    u = np.array([vx, vy]) / (math.hypot(vx, vy) + 1e-9) if speed > 0.25 else np.array([0.0, 1.0])

    # ankle separation along walking direction: one period = one full gait cycle
    d = _full(((xy[:, LAN] - xy[:, RAN]) @ u) / leg)
    if d is None:
        return None
    d = d - np.polyval(np.polyfit(t, d, 1), t)
    period, strength = _period(d, fps, cfg)
    if period is None or strength < cfg.min_periodicity:
        return None

    ds = _moving_avg(d, max(1, int(fps * 0.08)))
    up = np.where((ds[:-1] < 0) & (ds[1:] >= 0))[0]
    cycles = [(a, b) for a, b in zip(up[:-1], up[1:]) if 0.6 * period <= (b - a) / fps <= 1.5 * period]
    if len(cycles) < cfg.min_cycles:
        return None

    f = {"cadence": 120.0 / period, "stride_amp": float(np.percentile(d, 95) - np.percentile(d, 5))}

    zc = np.where(np.sign(ds[:-1]) != np.sign(ds[1:]))[0]
    iv = np.diff(zc) / fps
    if len(iv) >= 4:
        f["step_asym"] = float(abs(iv[0::2].mean() - iv[1::2].mean()) / iv.mean())

    hy = _full(hip[:, 1])
    if hy is not None:
        hy = hy - _moving_avg(hy, max(2, int(period * fps)))
        f["vertical_bob"] = float((np.percentile(hy, 95) - np.percentile(hy, 5)) / leg)

    if np.isfinite(arm) and arm > 0:
        a = _full(((xy[:, LWR] - xy[:, RWR]) @ u) / arm)
        if a is not None:
            f["arm_swing"] = float(np.percentile(a, 95) - np.percentile(a, 5))
            f["arm_leg_coupling"] = float(-np.corrcoef(a, d)[0, 1])   # normal walkers ~ +0.6..0.9

    kf_l = 180.0 - _angle(xy[:, LHIP], xy[:, LKN], xy[:, LAN])
    kf_r = 180.0 - _angle(xy[:, RHIP], xy[:, RKN], xy[:, RAN])
    roms = []
    for kf in (kf_l, kf_r):
        for a_, b_ in cycles:
            seg = kf[a_:b_ + 1]
            if np.isfinite(seg).mean() > 0.8:
                roms.append(np.nanmax(seg) - np.nanmin(seg))
    if roms:
        f["knee_rom"] = float(np.median(roms))

    if abs(u[0]) > 0.6:
        trunk = sh - hip
        lean = np.degrees(np.arctan2(trunk[:, 0] * np.sign(u[0]), -trunk[:, 1]))
        f["trunk_lean"] = float(np.nanmedian(lean))

    # phase-normalised template: hip swing L/R, knee flexion L/R over one cycle
    template = None
    tn = thigh if np.isfinite(thigh) and thigh > 0 else leg / 2
    chans = [
        _full(((xy[:, LKN] - xy[:, LHIP]) @ u) / tn),
        _full(((xy[:, RKN] - xy[:, RHIP]) @ u) / tn),
        _full(kf_l / 60.0),
        _full(kf_r / 60.0),
    ]
    if all(c is not None for c in chans):
        C = np.stack(chans)
        grid = np.linspace(0, 1, cfg.template_len)
        reps = []
        for a_, b_ in cycles:
            seg = C[:, a_:b_ + 1]
            src = np.linspace(0, 1, seg.shape[1])
            reps.append(np.stack([np.interp(grid, src, ch) for ch in seg]))
        template = np.mean(reps, axis=0)
        template = template - template.mean(axis=1, keepdims=True)

    lower = conf[:, [LHIP, RHIP, LKN, RKN, LAN, RAN]]
    quality = float(np.clip(np.mean(lower), 0, 1) * strength)
    return f, template, len(cycles), quality


def extract_signature(kpts: np.ndarray, fps: float, cfg: GaitConfig | None = None) -> GaitSignature | None:
    """kpts: (T, 17, 3) for ONE tracked person at a constant fps."""
    cfg = cfg or GaitConfig()
    kpts = np.asarray(kpts, dtype=float)
    if kpts.ndim != 3 or kpts.shape[1] < 17 or kpts.shape[0] < cfg.min_duration_s * fps:
        return None
    xy, conf = _clean(kpts, cfg)
    skel, dims = _skeleton(xy, cfg)
    view = _view(xy, cfg)
    g = _gait(xy, conf, fps, dims, cfg)

    feats = dict(skel)
    template, n_cycles = None, 0
    quality = float(np.clip(np.nanmean(conf[:, 5:17]), 0, 1)) * 0.5   # skeleton-only (not walking)
    if g is not None:
        gf, template, n_cycles, quality = g
        feats.update(gf)

    feats = {k: float(v) for k, v in feats.items()
             if k in FEATURES and np.isfinite(v) and view in FEATURES[k][2]}
    if not feats:
        return None
    return GaitSignature(view, feats, template, n_cycles, quality, kpts.shape[0] / fps)


# ================================================================ enrollment
def signatures_from_sequence(kpts, fps, cfg=None, window_s=4.0, hop_s=1.0) -> list[GaitSignature]:
    cfg = cfg or GaitConfig()
    w, h = int(window_s * fps), max(1, int(hop_s * fps))
    out = []
    for s in range(0, max(1, len(kpts) - w + 1), h):
        sig = extract_signature(kpts[s:s + w], fps, cfg)
        if sig is not None:
            out.append(sig)
    return out


def build_profile(signatures: list[GaitSignature]) -> GaitProfile | None:
    sigs = [s for s in signatures if s is not None]
    if not sigs:
        return None
    per_view = {}
    for view in VIEW_BINS:
        group = [s for s in sigs if s.view == view]
        if not group:
            continue
        keys = {k for s in group for k in s.features}
        feats = {k: float(np.median([s.features[k] for s in group if k in s.features])) for k in keys}
        temps = [s.template for s in group if s.template is not None]
        embs = [s.deep_embedding for s in group if s.deep_embedding is not None]
        emb = None
        if embs:
            emb = np.mean(embs, axis=0)
            emb = emb / (np.linalg.norm(emb) + 1e-9)
        per_view[view] = GaitSignature(
            view, feats, np.mean(temps, axis=0) if temps else None,
            sum(s.n_cycles for s in group), float(np.mean([s.quality for s in group])),
            sum(s.duration_s for s in group), emb)
    pooled = {}
    for k, (_, _, views) in FEATURES.items():
        if views == VIEW_BINS:
            vals = [s.features[k] for s in sigs if k in s.features]
            if vals:
                pooled[k] = float(np.median(vals))
    return GaitProfile(per_view, pooled, len(sigs))


def enroll_person(sequences: list[tuple[np.ndarray, float]], cfg=None) -> GaitProfile | None:
    """sequences: [(kpts (T,17,3), fps), ...] of the target person from reference video(s).
    Walking past the camera in several directions gives profiles for more view bins."""
    sigs = []
    for kpts, fps in sequences:
        sigs.extend(signatures_from_sequence(kpts, fps, cfg))
    return build_profile(sigs)


# ================================================================ matching
def _template_corr(a: np.ndarray, b: np.ndarray) -> float:
    best = -1.0
    for tb in (b, b[[1, 0, 3, 2]]):                 # left/right swap
        for s in range(a.shape[1]):                  # phase shift
            c = np.corrcoef(a.ravel(), np.roll(tb, s, axis=1).ravel())[0, 1]
            if np.isfinite(c):
                best = max(best, float(c))
    return best


class SkeletonGaitMatcher:
    def __init__(self, cfg: GaitConfig | None = None):
        self.cfg = cfg or GaitConfig()

    def compare(self, profile: GaitProfile, probe: GaitSignature) -> MatchResult | None:
        if profile is None or probe is None or not profile.per_view:
            return None
        pi = VIEW_BINS.index(probe.view)
        ref_view = min(profile.per_view, key=lambda v: abs(VIEW_BINS.index(v) - pi))
        ref = profile.per_view[ref_view]
        view_factor = VIEW_FACTOR[abs(VIEW_BINS.index(ref_view) - pi)]

        breakdown, num, den, veto = {}, 0.0, 0.0, False
        for k, (sigma, w, views) in FEATURES.items():
            if k not in probe.features:
                continue
            if views == VIEW_BINS:
                rv = profile.pooled.get(k, ref.features.get(k))
            elif ref_view in views:
                rv = ref.features.get(k)
            else:
                rv = None
            if rv is None:
                continue
            z = (probe.features[k] - rv) / sigma
            sim = math.exp(-0.5 * z * z)
            breakdown[k] = round(sim, 3)
            num += w * sim
            den += w
            if k in SKELETON_KEYS and abs(z) > self.cfg.skeleton_veto_sigmas and probe.quality > 0.4:
                veto = True

        parts = []
        if den > 0:
            parts.append((0.6, num / den))
        if probe.template is not None and ref.template is not None and ref_view == probe.view:
            c = _template_corr(ref.template, probe.template)
            ts = float(np.clip((c - 0.5) / 0.45, 0, 1))     # calibrate
            breakdown["template"] = round(ts, 3)
            parts.append((0.25, ts))
        if probe.deep_embedding is not None and ref.deep_embedding is not None:
            e = probe.deep_embedding / (np.linalg.norm(probe.deep_embedding) + 1e-9)
            ds = float(np.clip((float(e @ ref.deep_embedding) - 0.3) / 0.5, 0, 1))   # calibrate
            breakdown["deep_gait"] = round(ds, 3)
            parts.append((0.15, ds))
        if not parts:
            return None

        score = sum(w * s for w, s in parts) / sum(w for w, _ in parts)
        cycles_factor = min(1.0, probe.n_cycles / 4) if probe.n_cycles else 0.3   # skeleton-only
        reliability = float(np.clip(cycles_factor * view_factor * probe.quality * 1.5, 0, 1))
        return MatchResult(round(score, 3), round(reliability, 3), veto, ref_view, breakdown)


# ================================================================ live CCTV buffer
class GaitTrackBuffer:
    """Collects keypoints per (camera, track) from the live tracker and produces signatures.
    Timestamps may be irregular (dropped frames); they are resampled to a uniform grid."""

    def __init__(self, cfg: GaitConfig | None = None, window_s: float = 4.0, fps: float = 25.0):
        self.cfg = cfg or GaitConfig()
        self.window_s, self.fps = window_s, fps
        self.buf = defaultdict(lambda: deque(maxlen=int(window_s * fps * 3)))

    def add(self, cam_id: str, track_id: int, t: float, kpts: np.ndarray) -> None:
        self.buf[(cam_id, track_id)].append((t, np.asarray(kpts, dtype=float)))

    def drop(self, cam_id: str, track_id: int) -> None:
        self.buf.pop((cam_id, track_id), None)

    def signature(self, cam_id: str, track_id: int) -> GaitSignature | None:
        items = self.buf.get((cam_id, track_id))
        if not items or len(items) < 10:
            return None
        ts = np.array([t for t, _ in items])
        K = np.stack([k for _, k in items])                          # (N, 17, 3)
        t1 = ts[-1]
        t0 = max(ts[0], t1 - self.window_s)
        if t1 - t0 < self.cfg.min_duration_s:
            return None
        grid = np.arange(t0, t1, 1.0 / self.fps)
        out = np.empty((len(grid), K.shape[1], 3))
        for j in range(K.shape[1]):
            for c in range(3):
                out[:, j, c] = np.interp(grid, ts, K[:, j, c])
        idx = np.clip(np.searchsorted(ts, grid), 1, len(ts) - 1)
        gap = np.minimum(np.abs(ts[idx] - grid), np.abs(ts[idx - 1] - grid))
        out[gap > 1.5 / self.fps, :, 2] = 0.0                        # no real frame nearby
        return extract_signature(out, self.fps, self.cfg)


if __name__ == "__main__":
    print("skeleton_gait engine loaded")
