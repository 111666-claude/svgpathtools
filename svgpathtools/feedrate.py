"""Feed-rate (traversal time) planning along a :class:`Path`.

This module turns a geometric :class:`~svgpathtools.path.Path` into a
time-parameterized trajectory a machine tool (or any moving point) can follow.
Given a maximum speed, a tangential acceleration limit and a normal
acceleration limit, it produces an arc-length-to-time lookup and the speed at
every sampling point.

Kinematic model
---------------
Let ``s`` be the arc length travelled along the path and ``v(s) = ds/dt`` the
feed speed.  The acceleration vector decomposes into

* a tangential component ``a_t = dv/dt = v * dv/ds`` bounded in magnitude by
  ``tangential_accel``, and
* a normal component ``a_n = kappa(s) * v**2`` (centripetal acceleration,
  ``kappa`` being the curvature) bounded by ``normal_accel``.

Hence every point has a geometrically admissible ceiling
``v_lim(s) = min(max_speed, sqrt(normal_accel / kappa(s)))``.  Sharp corners
(discontinuous tangent, including cusp points where the derivative vanishes)
carry an additional joint ceiling derived from a corner-deviation model:
cornering at speed ``v`` deviates from the exact corner by at most
``corner_tolerance`` (with ``corner_tolerance=0`` corners are exact stops).

Two reachability sweeps (forward acceleration and backward deceleration) then
propagate the boundary speeds through those ceilings using the exact constant
tangential-acceleration relation, so the resulting profile never exceeds any
limit and is time-optimal for the chosen sampling grid.

The input path is never modified; only its read-only geometric queries are
used.
"""

from __future__ import annotations

import bisect
from warnings import warn

import numpy as np

from .path import Path

# Numerical guard: arc lengths smaller than this are treated as coincident.
_EPS_S = 1e-12
# Dot product this close to 1 means tangent-continuous.
_SMOOTH_COS = 1.0 - 1e-10
# Dot product at/below this means the path reverses through a single point.
_REVERSAL_COS = -1.0 + 1e-12
# Samples per segment in the arc-length lookup table.  Error in the resulting
# monotone s->t interpolation is O(1/m^2) and far below the planner's own
# grid discretization.
_PARAM_SAMPLES = 128
# Conservative factor on curvature-derived speed ceilings.  The s->t lookup
# interpolates the segment parameter, so the true curvature peak can sit a
# hair above the sampled value (measured worst-case curvature underestimate
# ~1e-5 relative at the default table resolution).  A matching speed margin
# keeps the realized normal acceleration inside the limit.  Joint/corner
# blend ceilings are exact and deliberately not scaled.
_CURV_SAFETY = 1.0 - 1.2e-5


class FeedRatePlan:
    """A discretized, kinematically feasible speed schedule for one path.

    Instances are normally produced by :func:`plan_feedrate`, not constructed
    directly.  The plan stores, at every grid node,

    * the cumulative arc length ``s`` from the path start,
    * the feasible speed ``v`` at that point,
    * the absolute time ``t`` at which the point is reached.

    Between two adjacent nodes the tangential acceleration is constant, so the
    analytic constant-acceleration formulas are used for every lookup.

    Attributes
    ----------
    path : Path
        The path the plan was built for (not modified by the planner).
    s : numpy.ndarray
        Strictly increasing arc-length coordinates of the grid nodes.
    v : numpy.ndarray
        Feasible speed at each grid node.
    t : numpy.ndarray
        Reach time of each grid node, starting at 0.
    max_speed, tangential_accel, normal_accel : float
        The kinematic limits the plan respects.
    closed : bool
        Whether the path is a closed loop.
    """

    def __init__(self, path, s, v, t, max_speed, tangential_accel,
                 normal_accel, corner_tolerance, closed):
        self.path = path
        self.s = np.asarray(s, dtype=float)
        self.v = np.asarray(v, dtype=float)
        self.t = np.asarray(t, dtype=float)
        self.max_speed = float(max_speed)
        self.tangential_accel = float(tangential_accel)
        self.normal_accel = float(normal_accel)
        self.corner_tolerance = float(corner_tolerance)
        self.closed = bool(closed)

    # ------------------------------------------------------------------ basics
    def __len__(self):
        return len(self.s)

    def __repr__(self):
        return ("FeedRatePlan(nodes={}, length={:.6g}, duration={:.6g}, "
                "closed={})".format(len(self.s), self.path_length(),
                                    self.duration(), self.closed))

    def path_length(self):
        """Total travelled arc length (the last grid node's ``s``)."""
        return float(self.s[-1]) if len(self.s) else 0.0

    def duration(self):
        """Total traversal time."""
        return float(self.t[-1]) if len(self.t) else 0.0

    def samples(self):
        """Return the ``(s, v, t)`` grid as a triplet of numpy arrays."""
        return self.s, self.v, self.t

    # ----------------------------------------------------------------- lookups
    def _interval(self, x, grid):
        """Index ``i`` with ``grid[i] <= x <= grid[i+1]`` (end-clamped)."""
        if len(grid) <= 1:
            return 0
        if x <= grid[0]:
            return 0
        if x >= grid[-1]:
            return len(grid) - 2
        return min(bisect.bisect_right(grid, x) - 1, len(grid) - 2)

    def velocity_at_s(self, s):
        """Speed ``v`` reached at arc length ``s``.

        Uses the constant-acceleration law ``v^2 = v0^2 + 2 a ds`` inside
        each grid interval.
        """
        s = float(s)
        if len(self.v) == 1:
            return float(self.v[0])
        i = self._interval(s, self.s)
        s0, s1 = self.s[i], self.s[i + 1]
        v0, v1 = self.v[i], self.v[i + 1]
        ds = s1 - s0
        if ds <= _EPS_S:
            return float(max(v0, v1))
        frac = (s - s0) / ds
        vsq = v0 * v0 + (v1 * v1 - v0 * v0) * frac
        return float(np.sqrt(max(vsq, 0.0)))

    def _interval_time(self, i, s):
        """Time spent inside interval ``i`` up to arc length ``s``."""
        s0, s1 = self.s[i], self.s[i + 1]
        v0, v1 = self.v[i], self.v[i + 1]
        ds = s1 - s0
        if ds <= _EPS_S:
            return 0.0
        local_s = min(max(s - s0, 0.0), ds)
        if abs(v1 - v0) <= _EPS_S:
            return local_s / v0 if v0 > _EPS_S else 0.0
        a = (v1 * v1 - v0 * v0) / (2.0 * ds)
        v_here = np.sqrt(max(v0 * v0 + 2.0 * a * local_s, 0.0))
        if abs(a) > _EPS_S:
            return float((v_here - v0) / a)
        return float(local_s / v0) if v0 > _EPS_S else 0.0

    def time_at_s(self, s):
        """Absolute time at which arc length ``s`` is reached."""
        s = float(s)
        if len(self.t) == 1:
            return float(self.t[0])
        if s >= self.s[-1]:
            return float(self.t[-1])
        i = self._interval(s, self.s)
        return float(self.t[i] + self._interval_time(i, s))

    def s_at_time(self, t):
        """Arc length reached at absolute time ``t``."""
        t = float(t)
        if len(self.s) == 1:
            return float(self.s[0])
        if t <= self.t[0]:
            return float(self.s[0])
        if t >= self.t[-1]:
            return float(self.s[-1])
        i = self._interval(t, self.t)
        s0, s1 = self.s[i], self.s[i + 1]
        v0, v1 = self.v[i], self.v[i + 1]
        dt = t - self.t[i]
        ds = s1 - s0
        if dt <= 0.0 or ds <= _EPS_S:
            return float(s0)
        if abs(v1 - v0) <= _EPS_S:
            return float(min(s0 + v0 * dt, s1))
        a = (v1 * v1 - v0 * v0) / (2.0 * ds)
        s_val = s0 + v0 * dt + 0.5 * a * dt * dt
        return float(min(max(s_val, s0), s1))

    def point_at_s(self, s):
        """Complex point on the path at arc length ``s``."""
        length = self.path_length()
        if length <= _EPS_S:
            return self.path[0].start if len(self.path) else 0j
        s_clamped = min(max(float(s), 0.0), length)
        T = self.path.ilength(s_clamped)
        return self.path.point(T)

    def point_at_time(self, t):
        """Complex point on the path reached at absolute time ``t``."""
        return self.point_at_s(self.s_at_time(t))

    def acceleration_at_s(self, s):
        """Signed tangential acceleration active at arc length ``s``."""
        if len(self.v) == 1:
            return 0.0
        i = self._interval(float(s), self.s)
        ds = self.s[i + 1] - self.s[i]
        if ds <= _EPS_S:
            return 0.0
        return float((self.v[i + 1] ** 2 - self.v[i] ** 2) / (2.0 * ds))


# ---------------------------------------------------------------------------
# Geometric helpers
# ---------------------------------------------------------------------------

class _ArcLengthMap:
    """Precomputed, O(1) arc-length <-> segment-parameter lookup for a path.

    For every segment it tabulates cumulative arc length on a uniform
    parameter grid once (``m+1`` length evaluations per segment instead of
    ~40 per node via inverse bisection) and answers subsequent
    ``s -> (segment, t)`` queries with monotone linear interpolation.  Exact
    endpoint values (t = 0, 1) are preserved, so segment joints are exact.
    """

    def __init__(self, path, m=_PARAM_SAMPLES):
        self.path = path
        self.seg_lengths = [float(seg.length()) for seg in path]
        self.offsets = np.concatenate([[0.0], np.cumsum(self.seg_lengths)])
        self.total = float(self.offsets[-1])
        self.tables = []
        ts = np.linspace(0.0, 1.0, m + 1)
        for seg, length in zip(path, self.seg_lengths):
            if length <= _EPS_S:
                self.tables.append(None)
                continue
            # Cumulative length at each uniform parameter value; exact at 0,1.
            cum = np.empty(m + 1)
            cum[0] = 0.0
            for k in range(1, m):
                cum[k] = float(seg.length(0.0, float(ts[k])))
            cum[m] = length
            self.tables.append((ts, cum))

    def locate(self, s_val):
        """Map a global arc length to ``(seg_idx, local t)``."""
        s_val = min(max(float(s_val), 0.0), self.total)
        idx = int(np.searchsorted(self.offsets, s_val, side="right") - 1)
        idx = min(max(idx, 0), len(self.path) - 1)
        length = self.seg_lengths[idx]
        local_s = s_val - self.offsets[idx]
        if length <= _EPS_S or local_s <= _EPS_S:
            return idx, 0.0
        if local_s >= length - _EPS_S:
            return idx, 1.0
        return idx, self._local_t(idx, local_s)

    def _local_t(self, idx, local_s):
        table = self.tables[idx]
        if table is None:
            return 0.0
        ts, cum = table
        j = int(np.searchsorted(cum, local_s, side="left"))
        j = min(max(j, 1), len(cum) - 1)
        s0, s1 = cum[j - 1], cum[j]
        if s1 <= s0:
            return float(ts[j])
        frac = (local_s - s0) / (s1 - s0)
        return float(ts[j - 1] + frac * (ts[j] - ts[j - 1]))


def _segment_curvature(seg, t):
    """Curvature of one segment at its local parameter ``t``.

    Returns ``inf`` at cusp points (vanishing derivative with no well-defined
    tangent); the caller is responsible for treating zero-length segments as
    stops.
    """
    try:
        kappa = float(seg.curvature(float(t)))
    except (ValueError, ZeroDivisionError, FloatingPointError,
            AssertionError):
        return float("inf")
    return kappa if np.isfinite(kappa) else float("inf")


def _curvature_at(amap, seg_idx, t):
    """Curvature at segment ``seg_idx`` / local parameter ``t``.

    Returns ``inf`` at a genuine corner, at cusp points (vanishing derivative
    with no well-defined tangent) and on zero-length segments (which are
    mandatory stops).
    """
    seg = amap.path[int(seg_idx)]
    if float(seg.length()) <= _EPS_S:
        return float("inf")
    return _segment_curvature(seg, t)


def _normal_ceiling(kappa, normal_accel):
    """Speed allowed by the normal-acceleration limit at curvature ``kappa``.

    A small safety factor accounts for arc-length lookup interpolation; see
    ``_CURV_SAFETY``.
    """
    if not np.isfinite(kappa) or kappa <= 0.0:
        return float("inf")
    return _CURV_SAFETY * float(np.sqrt(normal_accel / kappa))


def _joint_ceiling(path, seg_idx, corner_tolerance, normal_accel):
    """Speed ceiling at the joint *starting* segment ``seg_idx``.

    Returns ``(ceiling, is_kink)``: ``ceiling=inf`` for a tangent-continuous
    joint (curvature sampling then governs), finite speed for a corner, and
    ``0.0`` for exact stops (zero-length pieces and reversal cusps).
    """
    prev_seg = path[(seg_idx - 1) % len(path)]
    seg = path[seg_idx]

    if prev_seg.length() <= _EPS_S or seg.length() <= _EPS_S:
        return 0.0, True

    try:
        u = prev_seg.unit_tangent(1)
        w = seg.unit_tangent(0)
    except (ValueError, ZeroDivisionError, FloatingPointError):
        return 0.0, True

    cos_theta = float(np.real(np.conj(u) * w))
    cos_theta = min(max(cos_theta, -1.0), 1.0)
    if cos_theta >= _SMOOTH_COS:
        return float("inf"), False
    if cos_theta <= _REVERSAL_COS or corner_tolerance <= 0.0:
        return 0.0, True

    # Corner-deviation model.  A circular blend arc staying within
    # ``corner_tolerance`` of the corner has radius
    # R = eps / (1/cos(theta/2) - 1); its centripetal limit is sqrt(a_n R).
    half = 0.5 * np.arccos(cos_theta)
    denom = 1.0 / np.cos(half) - 1.0
    if denom <= _EPS_S:
        return float("inf"), False
    blend_radius = corner_tolerance / denom
    return float(np.sqrt(max(normal_accel * blend_radius, 0.0))), True


# ---------------------------------------------------------------------------
# Grid construction
# ---------------------------------------------------------------------------

def _build_grid(amap, max_step, corner_tolerance, normal_accel,
                length_tolerance, max_refines):
    """Build an adaptive arc-length grid with per-node speed ceilings.

    Returns ``(s, vlim, is_joint)`` where ``is_joint[i]`` marks segment
    boundaries and cusp samples.  Every segment joint is included regardless
    of ``max_step``.
    """
    path = amap.path
    seg_lengths = amap.seg_lengths
    total = amap.total
    closed = bool(path.isclosedac())

    # Each node is a tuple so joint ceilings and curvature ceilings combine
    # cleanly even when two pieces share an arc-length coordinate.
    nodes = []  # (s, seg_idx, t, is_joint, ceiling)
    seen = {}   # quantized arc length -> index into nodes

    def place(s_val, seg_idx, t_val, is_joint, ceiling):
        key = round(s_val / max(length_tolerance, _EPS_S))
        if key in seen:
            i = seen[key]
            ns, ni, nt, nj, nc = nodes[i]
            nodes[i] = (ns, ni, nt, nj or is_joint, min(nc, ceiling))
            return
        seen[key] = len(nodes)
        nodes.append((float(s_val), int(seg_idx), float(t_val),
                      bool(is_joint), float(ceiling)))

    s_offset = 0.0
    for seg_idx, seg in enumerate(path):
        length = seg_lengths[seg_idx]

        # Joint at the start of this segment.
        if seg_idx == 0 and not closed:
            place(s_offset, seg_idx, 0.0, True, float("inf"))
        else:
            ceiling, _is_kink = _joint_ceiling(
                path, seg_idx, corner_tolerance, normal_accel)
            place(s_offset, seg_idx, 0.0, True, ceiling)

        if length > _EPS_S:
            n_pieces = max(1, int(np.ceil(length / max_step)))
            step = length / n_pieces
            for k in range(1, n_pieces):
                local_s = k * step
                t_val = amap._local_t(seg_idx, local_s)
                place(s_offset + local_s, seg_idx, t_val, False, float("inf"))
        s_offset += length

    # Final endpoint / closing seam.
    if closed:
        ceiling, _is_kink = _joint_ceiling(
            path, 0, corner_tolerance, normal_accel)
        place(total, 0, 0.0, True, ceiling)
    else:
        place(total, len(path) - 1, 1.0, True, float("inf"))

    s = np.array([n[0] for n in nodes])
    seg_of = np.array([n[1] for n in nodes])
    t_of = np.array([n[2] for n in nodes])
    is_joint = np.array([n[3] for n in nodes])
    vlim = np.array([n[4] for n in nodes])

    # Curvature at every node, including curvature cusp points (inf).
    for i in range(len(s)):
        kappa = _curvature_at(amap, seg_of[i], t_of[i])
        vlim[i] = min(vlim[i], _normal_ceiling(kappa, normal_accel))

    # Adaptive refinement: subdivide intervals whose midpoint needs a
    # materially lower ceiling than both endpoints so tight spots cannot hide
    # between samples.
    for _ in range(max_refines):
        inserts = []
        for i in range(len(s) - 1):
            if s[i + 1] - s[i] <= length_tolerance:
                continue
            mid_s = 0.5 * (s[i] + s[i + 1])
            seg_idx, local_t = amap.locate(mid_s)
            kappa = _curvature_at(amap, seg_idx, local_t)
            mid_ceiling = _normal_ceiling(kappa, normal_accel)
            if mid_ceiling + length_tolerance < min(vlim[i], vlim[i + 1]):
                inserts.append((mid_s, seg_idx, local_t, mid_ceiling))
        if not inserts:
            break
        for mid_s, seg_idx, local_t, mid_ceiling in inserts:
            s = np.append(s, mid_s)
            seg_of = np.append(seg_of, seg_idx)
            t_of = np.append(t_of, local_t)
            is_joint = np.append(is_joint, False)
            vlim = np.append(vlim, mid_ceiling)
        order = np.argsort(s, kind="stable")
        s, seg_of, t_of = s[order], seg_of[order], t_of[order]
        is_joint, vlim = is_joint[order], vlim[order]
        # Merge numerically coincident nodes, keeping the strictest ceiling.
        keep = np.concatenate([[True], np.diff(s) > length_tolerance])
        if not np.all(keep):
            for dropped in np.where(~keep)[0]:
                j = dropped - 1
                is_joint[j] = is_joint[j] or is_joint[dropped]
                vlim[j] = min(vlim[j], vlim[dropped])
            s, seg_of, t_of = s[keep], seg_of[keep], t_of[keep]
            is_joint, vlim = is_joint[keep], vlim[keep]

    return s, vlim, is_joint


# ---------------------------------------------------------------------------
# Reachability sweeps
# ---------------------------------------------------------------------------

def _forward_sweep(vlim, ds, a_t, v_start):
    """Largest feasible speeds reachable accelerating from ``v_start``."""
    v = np.array(vlim, dtype=float, copy=True)
    v[0] = min(v[0], v_start)
    for i in range(len(v) - 1):
        reachable_sq = v[i] ** 2 + 2.0 * ds[i] * a_t
        v[i + 1] = min(v[i + 1], np.sqrt(max(reachable_sq, 0.0)))
    return v


def _backward_sweep(v, ds, a_t, v_end):
    """Deceleration reachability from the end, mutating and returning ``v``."""
    v[-1] = min(v[-1], v_end)
    for i in range(len(v) - 2, -1, -1):
        reachable_sq = v[i + 1] ** 2 + 2.0 * ds[i] * a_t
        v[i] = min(v[i], np.sqrt(max(reachable_sq, 0.0)))
    return v


def _integrate_time(s, v):
    """Reach times, exact for constant per-interval acceleration."""
    t = np.zeros(len(s))
    for i in range(len(s) - 1):
        ds = s[i + 1] - s[i]
        v0, v1 = v[i], v[i + 1]
        if ds <= _EPS_S:
            t[i + 1] = t[i]
        elif abs(v1 - v0) <= _EPS_S:
            t[i + 1] = t[i] + (ds / v0 if v0 > _EPS_S else 0.0)
        else:
            # ds = (v0 + v1)/2 * dt for constant acceleration.
            t[i + 1] = t[i] + 2.0 * ds / (v0 + v1)
    return t


def _solve_closed(vlim, ds, a_t, seam_speed, seam_pinned, max_cycles):
    """Cyclic forward/backward sweeps for a closed loop.

    The seam node appears at both ends with one shared ceiling.  Sweeps are
    repeated until a fixed point; each iteration only lowers speeds, so the
    iteration converges monotonically.

    ``seam_speed`` is the seam's ceiling (already the geometric joint limit,
    possibly further reduced by an explicit user request).  When
    ``seam_pinned`` is True the seam must equal that value; otherwise it is
    only an upper bound and the sweep settles on the fastest periodic value
    reachable around the loop.
    """
    n = len(vlim)
    v = np.array(vlim, dtype=float, copy=True)
    v[0] = min(v[0], seam_speed)
    v[-1] = min(v[-1], seam_speed)

    for _ in range(max_cycles):
        prev = v.copy()
        for i in range(n - 1):
            reachable_sq = v[i] ** 2 + 2.0 * ds[i] * a_t
            v[i + 1] = min(v[i + 1], np.sqrt(max(reachable_sq, 0.0)))
        v[0] = min(v[0], v[-1])
        for i in range(n - 2, -1, -1):
            reachable_sq = v[i + 1] ** 2 + 2.0 * ds[i] * a_t
            v[i] = min(v[i], np.sqrt(max(reachable_sq, 0.0)))
        v[-1] = min(v[-1], v[0])
        if seam_pinned:
            v[0] = min(seam_speed, v[0])
            v[-1] = v[0]
        if np.max(np.abs(v - prev)) <= 1e-12 * (1.0 + np.max(v)):
            break
    return v


# ---------------------------------------------------------------------------
# Time-tolerance refinement
# ---------------------------------------------------------------------------

def _refine_by_time(amap, s, vlim, is_joint, max_speed, a_t, a_n,
                    time_tolerance, length_tolerance, max_refines,
                    v_start, v_end, closed, seam_pinned, max_cycles):
    """Repeatedly insert interval midpoints until the duration stabilizes."""
    prev_duration = None
    for _ in range(max_refines):
        ds = np.diff(s)
        if closed:
            v = _solve_closed(vlim, ds, a_t, min(v_start, v_end),
                              seam_pinned, max_cycles)
        else:
            v = _forward_sweep(vlim, ds, a_t, v_start)
            v = _backward_sweep(v, ds, a_t, v_end)
        duration = float(_integrate_time(s, v)[-1])
        if prev_duration is not None and abs(
                duration - prev_duration) <= time_tolerance:
            break
        prev_duration = duration

        mid_s = 0.5 * (s[:-1] + s[1:])
        extra_vlim = []
        for ms in mid_s:
            seg_idx, local_t = amap.locate(ms)
            kappa = _curvature_at(amap, seg_idx, local_t)
            extra_vlim.append(min(max_speed, _normal_ceiling(kappa, a_n)))
        s = np.concatenate([s, mid_s])
        vlim = np.concatenate([vlim, extra_vlim])
        is_joint = np.concatenate([is_joint,
                                   np.zeros(len(mid_s), dtype=bool)])
        order = np.argsort(s, kind="stable")
        s, vlim, is_joint = s[order], vlim[order], is_joint[order]
        keep = np.concatenate([[True], np.diff(s) > length_tolerance])
        if not np.all(keep):
            for dropped in np.where(~keep)[0]:
                j = dropped - 1
                is_joint[j] = is_joint[j] or is_joint[dropped]
                vlim[j] = min(vlim[j], vlim[dropped])
            s, vlim, is_joint = s[keep], vlim[keep], is_joint[keep]
    return s, vlim, is_joint


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def plan_feedrate(path, max_speed, tangential_accel, normal_accel,
                  start_speed=None, end_speed=None, corner_tolerance=0.0,
                  max_step=None, time_tolerance=None,
                  length_tolerance=1e-9, max_refines=6, max_cycles=100):
    """Plan a feasible, time-optimal feed-rate schedule along ``path``.

    Parameters
    ----------
    path : Path
        The geometry to traverse.  It is treated as read-only.
    max_speed : float
        Global speed ceiling ``v_max`` (length / time), must be positive.
    tangential_accel : float
        Magnitude limit on the tangential acceleration ``|dv/dt|``, must be
        positive.
    normal_accel : float
        Centripetal acceleration limit ``kappa * v^2``, must be positive.
    start_speed, end_speed : float or None, optional
        Boundary speeds.  Open paths default to rest (0).  For a closed loop
        the two ends are the same physical seam, so ``None`` (the default for
        closed loops) lets the planner use the seam's geometric ceiling;
        passing a number pins the seam to that speed.  Negative values raise
        ``ValueError``; a boundary speed above the local geometric ceiling is
        clamped down to that ceiling (with a warning), so a feasible result
        always exists.
    corner_tolerance : float, optional
        Maximum geometric deviation allowed while rounding a sharp tangent
        corner.  ``0.0`` (default) makes every corner an exact stop.  Larger
        values permit cornering at the speed whose circular blend arc fits
        inside the tolerance.
    max_step : float, optional
        Maximum arc length between grid nodes.  Defaults to 1/200 of the path
        length.  Segment joints are always included regardless of this value.
    time_tolerance : float, optional
        Maximum accepted change in total traversal time under one more grid
        refinement.  When given, the grid is refined until the duration
        stabilizes within this tolerance.  ``None`` (default) runs only the
        curvature-driven refinement.
    length_tolerance : float, optional
        Arc-length resolution below which no further subdivision is attempted
        and two nodes are considered coincident.
    max_refines : int, optional
        Upper bound on adaptive refinement passes.
    max_cycles : int, optional
        Upper bound on cyclic sweep iterations used for closed loops.

    Returns
    -------
    FeedRatePlan
        A queryable trajectory holding arc lengths, speeds and times.

    Raises
    ------
    ValueError
        If a kinematic limit is non-positive, a boundary speed is negative,
        the path is empty, or it is not geometrically continuous (use
        :meth:`Path.continuous_subpaths` first in that case).
    """
    if not isinstance(path, Path):
        raise TypeError("first argument must be a Path instance")
    if max_speed <= 0.0:
        raise ValueError("max_speed must be positive, got %r" % max_speed)
    if tangential_accel <= 0.0:
        raise ValueError("tangential_accel must be positive, got %r"
                         % tangential_accel)
    if normal_accel <= 0.0:
        raise ValueError("normal_accel must be positive, got %r"
                         % normal_accel)
    if (start_speed is not None and start_speed < 0.0) or \
            (end_speed is not None and end_speed < 0.0):
        raise ValueError("start_speed and end_speed must be non-negative")
    if corner_tolerance < 0.0:
        raise ValueError("corner_tolerance must be non-negative")
    if len(path) == 0:
        raise ValueError("cannot plan a feed rate for an empty Path")
    if not path.iscontinuous():
        raise ValueError(
            "path is not geometrically continuous; split it with "
            "Path.continuous_subpaths() and plan each subpath separately")

    a_t = float(tangential_accel)
    total_length = float(path.length())
    closed = bool(path.isclosedac())

    # A closed loop's two ends are one point.  Absent an explicit request the
    # seam takes its natural geometric ceiling instead of an artificial stop;
    # open paths start and end at rest unless told otherwise.
    if closed:
        requested = [v for v in (start_speed, end_speed) if v is not None]
        seam_request = min(requested) if requested else None
    else:
        seam_request = None
        if start_speed is None:
            start_speed = 0.0
        if end_speed is None:
            end_speed = 0.0

    # ----------------------- degenerate path: everything is coincident ------
    if total_length <= _EPS_S:
        requested = [v for v in (start_speed, end_speed)
                     if v is not None and v > 0.0]
        if requested or (seam_request is not None and seam_request > 0.0):
            warn("zero-length path cannot carry speed; using v = 0")
        s = np.array([0.0])
        v = np.array([0.0])
        t = np.array([0.0])
        return FeedRatePlan(path, s, v, t, max_speed, a_t, normal_accel,
                            corner_tolerance, closed)

    if max_step is None:
        max_step = total_length / 200.0
    max_step = max(float(max_step), length_tolerance)

    amap = _ArcLengthMap(path)
    s, vlim, is_joint = _build_grid(
        amap, max_step, corner_tolerance, normal_accel,
        length_tolerance, max_refines)
    vlim = np.minimum(vlim, max_speed)

    # ----------------------- infeasible boundary speeds -> deterministic ----
    if closed:
        if seam_request is not None:
            if seam_request > vlim[0] + 1e-12:
                warn("seam speed=%s exceeds the local ceiling %s; clamping"
                     % (seam_request, vlim[0]))
            v_start = v_end = min(float(seam_request), float(vlim[0]))
        else:
            v_start = v_end = float(vlim[0])
    else:
        if start_speed > vlim[0] + 1e-12:
            warn("start_speed=%s exceeds the local ceiling %s; clamping"
                 % (start_speed, vlim[0]))
        if end_speed > vlim[-1] + 1e-12:
            warn("end_speed=%s exceeds the local ceiling %s; clamping"
                 % (end_speed, vlim[-1]))
        v_start = min(float(start_speed), float(vlim[0]))
        v_end = min(float(end_speed), float(vlim[-1]))

    if time_tolerance is not None:
        s, vlim, is_joint = _refine_by_time(
            amap, s, vlim, is_joint, max_speed, a_t, normal_accel,
            float(time_tolerance), length_tolerance, max_refines,
            v_start, v_end, closed, seam_request is not None, max_cycles)

    ds = np.diff(s)
    if closed:
        v = _solve_closed(vlim, ds, a_t, min(v_start, v_end),
                          seam_request is not None, max_cycles)
    else:
        v = _forward_sweep(vlim, ds, a_t, v_start)
        v = _backward_sweep(v, ds, a_t, v_end)
        v[0] = v_start
        v[-1] = v_end
    t = _integrate_time(s, v)

    return FeedRatePlan(path, s, v, t, max_speed, a_t, normal_accel,
                        corner_tolerance, closed)
