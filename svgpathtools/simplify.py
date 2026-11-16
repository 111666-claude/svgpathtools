"""This submodule provides path simplification with a guaranteed bound on
the geometric deviation from the original path.

The main entry point is :func:`simplify_path`.  The helper
:func:`max_deviation` measures the (sampled) maximum geometric deviation
between two paths; it is used internally to verify results and is also
useful in user code and tests.

Method overview
---------------
For every continuous subpath of the input, ``simplify_path``

1. samples each segment adaptively into a dense polyline that deviates
   from the original curve by at most ``0.2 * tolerance``,
2. marks points that must be kept: subpath endpoints, the seam of closed
   subpaths, and joints whose turning angle exceeds ``corner_angle``
   (significant corners),
3. simplifies the dense polyline with the Ramer-Douglas-Peucker algorithm
   (never removing a marked point) with threshold ``0.5 * tolerance``,
4. rebuilds segments: runs of nearly collinear chords are merged into a
   single ``Line`` and runs that can be fitted are replaced by (merged)
   ``CubicBezier`` segments whose sampled deviation from the dense
   polyline is at most ``0.5 * tolerance``,
5. verifies, by sampling, that the maximum geometric deviation between
   input and output does not exceed ``tolerance``.

Because the error budget (0.2 + 0.5, resp. 0.2 + 0.5) sums to less than
1.0, the verified deviation stays below ``tolerance``.  Everything is a
deterministic function of the input path and the tolerance.
"""

# External dependencies
from __future__ import division, absolute_import, print_function
import math

import numpy as np

# Internal dependencies
from .path import Path, Line, QuadraticBezier, CubicBezier

__all__ = ['simplify_path', 'max_deviation', 'SimplificationError',
           'DEFAULT_CORNER_ANGLE']

#: joints whose turning angle exceeds this are kept as significant corners
DEFAULT_CORNER_ANGLE = math.radians(60.0)

# Fractions of the (working) tolerance spent on each pipeline stage.  The
# worst-case total is _SAMPLE_FRAC + max(_RDP_FRAC, _FIT_FRAC) < 1.
_SAMPLE_FRAC = 0.2
_RDP_FRAC = 0.5
_FIT_FRAC = 0.5
_VERIFY_SAMPLE_FRAC = 0.05

_MAX_SAMPLE_DEPTH = 20
_MAX_FIT_DEPTH = 16
_MAX_ATTEMPTS = 6
_CHUNK = 256
# runs with more sample points than this are fitted on a downsampled copy
# first (the result is always verified against all points)
_FIT_SUBSET_SIZE = 200


class SimplificationError(Exception):
    """Raised when a path cannot be simplified within the requested error
    bound (e.g. because the verified deviation exceeds the tolerance, or
    ``preserve_topology=True`` was requested but every attempt changed the
    intersection structure of the path)."""


# -------------------------------------------------------------------------
# small geometry helpers (points are (n, 2) float arrays or complex scalars)
# -------------------------------------------------------------------------

def _c(v):
    """(x, y) array-like -> complex"""
    return complex(v[0], v[1])


def _point_seg_dist(p, a, b):
    """distance from complex point p to the segment [a, b] (complex)"""
    ab = b - a
    len2 = ab.real * ab.real + ab.imag * ab.imag
    if len2 == 0.0:
        return abs(p - a)
    t = ((p - a).real * ab.real + (p - a).imag * ab.imag) / len2
    t = min(1.0, max(0.0, t))
    return abs(p - (a + t * ab))


def _dists_to_segment(P, a, b):
    """distances from points P (n, 2) to the segment [a, b] ((2,) arrays)"""
    ab = b - a
    len2 = float(ab @ ab)
    if len2 == 0.0:
        return np.hypot(P[:, 0] - a[0], P[:, 1] - a[1])
    t = ((P - a) @ ab) / len2
    np.clip(t, 0.0, 1.0, out=t)
    D = P - (a + t[:, None] * ab)
    return np.hypot(D[:, 0], D[:, 1])


def _point_polyline_maxdist(P, Q):
    """Returns (max distance, argmax) from points P to the polyline Q."""
    if len(P) == 0:
        return 0.0, 0
    A = Q[:-1]
    B = Q[1:]
    AB = B - A
    L2 = np.einsum('ij,ij->i', AB, AB)
    nonzero = L2 > 0.0
    safe_L2 = np.where(nonzero, L2, 1.0)
    A_AB = np.einsum('ij,ij->i', A, AB)
    A2 = np.einsum('ij,ij->i', A, A)
    best = -1.0
    best_i = 0
    for s in range(0, len(P), _CHUNK):
        Pc = P[s:s + _CHUNK]
        # (P - A) . AB for all point/segment pairs
        dot = Pc @ AB.T - A_AB[None, :]
        t = dot / safe_L2[None, :]
        t = np.where(nonzero[None, :], t, 0.0)
        np.clip(t, 0.0, 1.0, out=t)
        # |P - (A + t AB)|^2 = |P-A|^2 - 2 t (P-A).AB + t^2 |AB|^2
        d2 = ((Pc * Pc).sum(axis=1)[:, None] - 2.0 * (Pc @ A.T)
              + A2[None, :] - 2.0 * t * dot + t * t * L2[None, :])
        d = np.sqrt(np.maximum(d2, 0.0)).min(axis=1)
        i = int(np.argmax(d))
        if d[i] > best:
            best = float(d[i])
            best_i = s + i
    return best, best_i


# -------------------------------------------------------------------------
# adaptive sampling of segments into dense polylines
# -------------------------------------------------------------------------

def _seg_flatness(seg):
    """An upper bound on the distance from a Bezier segment to its chord,
    or None for segment types without a cheap bound (e.g. Arc)."""
    if isinstance(seg, Line):
        return 0.0
    if isinstance(seg, (QuadraticBezier, CubicBezier)):
        a, b = seg.start, seg.end
        # The curve lies in the convex hull of its control points and the
        # set of points within distance d of a segment is convex, so the
        # curve is within max(control point distance) of the chord.
        return max((_point_seg_dist(cp, a, b)
                    for cp in seg.bpoints()[1:-1]), default=0.0)
    return None


def _sample_segment(seg, tol, depth=0):
    """Returns a list of complex points (start ... end) approximating seg
    to within tol of its chords."""
    if depth < _MAX_SAMPLE_DEPTH:
        flat = _seg_flatness(seg)
        if flat is not None:
            done = flat <= tol
        else:
            a, b = seg.start, seg.end
            done = all(_point_seg_dist(seg.point(t), a, b) <= tol
                       for t in (0.25, 0.5, 0.75))
        if not done:
            left, right = seg.split(0.5)
            return (_sample_segment(left, tol, depth + 1)[:-1] +
                    _sample_segment(right, tol, depth + 1))
    return [seg.start, seg.end]


def _sampled_subpaths(path, sample_tol):
    """list of (n, 2) arrays: dense polylines of the continuous subpaths"""
    polylines = []
    for sub in path.continuous_subpaths():
        if len(sub) == 0:
            continue
        pts = []
        for seg in sub:
            pts.extend(_sample_segment(seg, sample_tol)[:-1])
        pts.append(sub[-1].end)
        polylines.append(np.array([[z.real, z.imag] for z in pts]))
    return polylines


def max_deviation(path_a, path_b, sample_tol=1e-3):
    """Returns the (sampled) maximum geometric deviation between two paths.

    Both paths are sampled into dense polylines (accurate to
    ``sample_tol``) and the symmetric maximum point-to-polyline distance
    is returned.  A path with no segments has deviation 0 from another
    empty path and infinity from any non-empty path.
    """
    A = _sampled_subpaths(path_a, sample_tol)
    B = _sampled_subpaths(path_b, sample_tol)
    if not A or not B:
        return 0.0 if not A and not B else float('inf')
    dev = 0.0
    for P in A:
        dev = max(dev, min(_point_polyline_maxdist(P, Q)[0] for Q in B))
    for Q in B:
        dev = max(dev, min(_point_polyline_maxdist(Q, P)[0] for P in A))
    return dev


# -------------------------------------------------------------------------
# Ramer-Douglas-Peucker with points that must be kept
# -------------------------------------------------------------------------

def _rdp(pts, tol, forced):
    """Returns the sorted indices of the kept points of the polyline pts
    ((n, 2) array).  Points where forced is True are never removed; the
    first and last points are always kept."""
    n = len(pts)
    keep = forced.copy()
    keep[0] = keep[-1] = True
    forced_idx = np.flatnonzero(keep).tolist()
    stack = [(forced_idx[i], forced_idx[i + 1])
             for i in range(len(forced_idx) - 1)]
    while stack:
        i, j = stack.pop()
        if j - i < 2:
            continue
        d = _dists_to_segment(pts[i + 1:j], pts[i], pts[j])
        k = int(np.argmax(d))
        if d[k] > tol:
            k += i + 1
            keep[k] = True
            stack.append((i, k))
            stack.append((k, j))
    return np.flatnonzero(keep)


# -------------------------------------------------------------------------
# cubic Bezier fitting (Schneider's endpoint-tangent least squares fit,
# with split-at-max-error recursion)
# -------------------------------------------------------------------------

def _estimate_tangents(pts):
    """unit tangent estimates at both ends of the polyline pts"""
    n = len(pts)
    chord = pts[-1] - pts[0]
    norm = np.hypot(chord[0], chord[1])
    fallback = chord / norm if norm > 0 else np.array([1.0, 0.0])
    t0 = fallback
    for k in range(1, min(4, n)):
        d = pts[k] - pts[0]
        norm = np.hypot(d[0], d[1])
        if norm > 0:
            t0 = d / norm
            break
    t1 = fallback
    for k in range(n - 2, max(n - 5, -1), -1):
        d = pts[-1] - pts[k]
        norm = np.hypot(d[0], d[1])
        if norm > 0:
            t1 = d / norm
            break
    return t0, t1


def _fit_cubic_once(pts, t0, t1, u):
    """Fits a single cubic Bezier to pts ((n, 2) array, n >= 2) with end
    tangents t0, t1 (unit (2,) arrays, travel direction) and parameter
    values u ((n,) array in [0, 1]).  Returns a CubicBezier or None if a
    line is more appropriate."""
    n = len(pts)
    p0, p3 = pts[0], pts[-1]
    chord = p3 - p0
    L = float(np.hypot(chord[0], chord[1]))
    if n == 2:
        return None
    if L > 0.0:
        scale = L
    else:
        # closed run (start == end): use the spread of the data as scale
        scale = float(np.sqrt(((pts - p0) ** 2).sum(axis=1)).max())
        if scale == 0.0:
            return None
    u1 = 1.0 - u
    b0 = u1 ** 3
    b1 = 3.0 * u * u1 ** 2
    b2 = 3.0 * u ** 2 * u1
    b3 = u ** 3
    A1 = t0[None, :] * b1[:, None]
    A2 = (-t1)[None, :] * b2[:, None]
    rhs = pts - (p0[None, :] * (b0 + b1)[:, None] +
                 p3[None, :] * (b2 + b3)[:, None])
    c11 = float((A1 * A1).sum())
    c12 = float((A1 * A2).sum())
    c22 = float((A2 * A2).sum())
    x1 = float((A1 * rhs).sum())
    x2 = float((A2 * rhs).sum())
    det = c11 * c22 - c12 * c12
    alpha1 = alpha2 = None
    if det > 0.0:
        a1 = (x1 * c22 - x2 * c12) / det
        a2 = (c11 * x2 - c12 * x1) / det
        if (np.isfinite(a1) and np.isfinite(a2)
                and 0.0 < a1 <= 3.0 * scale and 0.0 < a2 <= 3.0 * scale):
            alpha1, alpha2 = a1, a2
    if alpha1 is None:
        alpha1 = alpha2 = scale / 3.0
    c1 = p0 + alpha1 * t0
    c2 = p3 - alpha2 * t1
    return CubicBezier(_c(p0), _c(c1), _c(c2), _c(p3))


def _fit_cubic_free(pts, u):
    """Unconstrained least-squares cubic fit to pts ((n, 2) array) with
    fixed endpoints and parameter values u ((n,) array in [0, 1]).
    Returns a CubicBezier or None if the system is singular."""
    p0, p3 = pts[0], pts[-1]
    u1 = 1.0 - u
    b0 = u1 ** 3
    b1 = 3.0 * u * u1 ** 2
    b2 = 3.0 * u ** 2 * u1
    b3 = u ** 3
    rhs = pts - (p0[None, :] * b0[:, None] + p3[None, :] * b3[:, None])
    m11 = float((b1 * b1).sum())
    m12 = float((b1 * b2).sum())
    m22 = float((b2 * b2).sum())
    r1 = (b1[:, None] * rhs).sum(axis=0)
    r2 = (b2[:, None] * rhs).sum(axis=0)
    det = m11 * m22 - m12 * m12
    if not det > 0.0:
        return None
    v1 = (r1 * m22 - r2 * m12) / det
    v2 = (m11 * r2 - m12 * r1) / det
    if not (np.all(np.isfinite(v1)) and np.all(np.isfinite(v2))):
        return None
    return CubicBezier(_c(p0), _c(v1), _c(v2), _c(p3))


def _reparameterize(cub, pts, u):
    """Newton iteration moving each u[i] to the parameter of the point on
    cub closest to pts[i] (Schneider's reparameterization step)."""
    B = cub.poly()
    B1 = B.deriv()
    B2 = B1.deriv()
    t = u.copy()
    for _ in range(3):
        b = B(t)
        b1 = B1(t)
        b2 = B2(t)
        dx = b.real - pts[:, 0]
        dy = b.imag - pts[:, 1]
        num = dx * b1.real + dy * b1.imag
        den = b1.real ** 2 + b1.imag ** 2 + dx * b2.real + dy * b2.imag
        nonzero = np.abs(den) > 1e-14
        dt = np.where(nonzero, num / np.where(nonzero, den, 1.0), 0.0)
        t = np.clip(t - dt, 0.0, 1.0)
    return t


def _cubic_sample_points(cub, n):
    """samples of cub on a regular grid of max(33, 2n) points, (m, 2)"""
    ts = np.linspace(0.0, 1.0, max(33, 2 * n))
    cz = cub.poly()(ts)
    return np.column_stack([cz.real, cz.imag])


def _fit_cubic(pts, t0, t1, tol):
    """Fits a single cubic Bezier to pts ((n, 2) array, n >= 2) with end
    tangents t0, t1, iterating least-squares fits with reparameterization
    (Schneider's algorithm, alternating endpoint-tangent-constrained and
    unconstrained fits).  Iteration uses the cheap one-sided deviation
    and stops as soon as it is at most tol.

    Returns (cubic, err, split_idx): the best cubic found, its symmetric
    sampled deviation from pts, and the index into pts of the worst
    point (for splitting).  Returns (None, inf, -1) if no fit is
    possible (e.g. all points coincide)."""
    n = len(pts)
    if n == 2:
        return None, float('inf'), -1
    # initial chord-length parameterization
    steps = np.sqrt((np.diff(pts, axis=0) ** 2).sum(axis=1))
    u = np.concatenate(([0.0], np.cumsum(steps)))
    if u[-1] == 0.0:
        return None, float('inf'), -1
    u /= u[-1]
    cub = _fit_cubic_once(pts, t0, t1, u)
    if cub is None:
        return None, float('inf'), -1
    best = None
    best_err = float('inf')
    stale = 0
    for _ in range(30):
        C = _cubic_sample_points(cub, n)
        err, idx = _point_polyline_maxdist(pts, C)
        if err < best_err * 0.999:
            stale = 0
        else:
            stale += 1
        if err < best_err:
            best, best_err = cub, err
        if err <= tol:
            # the reverse deviation lags the forward one; check it
            # before stopping
            rev = _point_polyline_maxdist(C, pts)[0]
            if max(err, rev) <= tol:
                return cub, max(err, rev), idx
        if stale >= 3:
            break
        u = _reparameterize(cub, pts, u)
        cub = _fit_cubic_free(pts, u)
        if cub is None:
            break
    # symmetric sampled deviation of the winner
    C = _cubic_sample_points(best, n)
    fwd, idx = _point_polyline_maxdist(pts, C)
    rev = _point_polyline_maxdist(C, pts)[0]
    return best, max(fwd, rev), idx


def _fit_spline(pts, tol, depth=0):
    """Returns a list of segments (CubicBezier or Line) covering the
    polyline pts ((n, 2) array) within tol."""
    n = len(pts)
    if n == 2:
        return [Line(_c(pts[0]), _c(pts[1]))]
    if depth < _MAX_FIT_DEPTH:
        subset = n > _FIT_SUBSET_SIZE
        if subset:
            # fit a downsampled copy first (cheap); the result is always
            # verified against all points before being accepted
            step = int(np.ceil(n / _FIT_SUBSET_SIZE))
            fpts = pts[::step]
            if not np.array_equal(fpts[-1], pts[-1]):
                fpts = np.vstack([fpts, pts[-1:]])
            fit_tol = 0.8 * tol
        else:
            fpts = pts
            fit_tol = tol
        t0, t1 = _estimate_tangents(fpts)
        cub, err, idx = _fit_cubic(fpts, t0, t1, fit_tol)
        if cub is not None:
            if subset and err <= tol:
                # verify the subset fit against all dense points
                C = _cubic_sample_points(cub, n)
                fwd, idx = _point_polyline_maxdist(pts, C)
                rev = _point_polyline_maxdist(C, pts)[0]
                err = max(fwd, rev)
                subset = False  # idx now refers to pts
            if err <= tol:
                return [cub]
            if subset or idx <= 0 or idx >= n - 1:
                idx = n // 2
            return (_fit_spline(pts[:idx + 1], tol, depth + 1) +
                    _fit_spline(pts[idx:], tol, depth + 1))
    # fallback: chords between adjacent sample points (always within the
    # sampling tolerance of the original curve)
    return [Line(_c(pts[i]), _c(pts[i + 1])) for i in range(n - 1)]


# -------------------------------------------------------------------------
# subpath simplification
# -------------------------------------------------------------------------

def _turning_angle(seg_in, seg_out):
    """angle between the incoming and outgoing unit tangents at a joint"""
    try:
        u = seg_in.unit_tangent(1)
        v = seg_out.unit_tangent(0)
        dot = u.real * v.real + u.imag * v.imag
    except Exception:
        # tangent not well-defined (e.g. degenerate segment): keep the joint
        return math.pi
    return math.acos(max(-1.0, min(1.0, dot)))


class _Budget(object):
    def __init__(self, tol):
        self.sample = _SAMPLE_FRAC * tol
        self.rdp = _RDP_FRAC * tol
        self.fit = _FIT_FRAC * tol


def _max_chord_dev(pts):
    if len(pts) <= 2:
        return 0.0
    return float(_dists_to_segment(pts[1:-1], pts[0], pts[-1]).max())


def _emit_run(ring, run, budget):
    """Builds the output segments for one run of the kept polyline.

    ring: (m, 2) array of all sample points; run: list of kept indices
    into ring (at least 2) delimiting the run."""
    a, b = run[0], run[-1]
    if len(run) == 2:
        return [Line(_c(ring[a]), _c(ring[b]))]
    run_pts = ring[a:b + 1]
    # merge nearly-collinear runs (redundant lines) into a single Line
    if _max_chord_dev(run_pts) <= budget.rdp:
        return [Line(_c(ring[a]), _c(ring[b]))]
    # merge adjacent curves that can be fitted into CubicBezier segments
    fitted = _fit_spline(run_pts, budget.fit)
    if len(fitted) >= len(run) - 1:
        # fitting did not compress: fall back to the RDP chords
        return [Line(_c(ring[run[i]]), _c(ring[run[i + 1]]))
                for i in range(len(run) - 1)]
    return fitted


def _simplify_subpath(sub, budget, corner_angle):
    """Returns the list of output segments for one continuous subpath."""
    n_seg = len(sub)
    closed = sub.isclosed()

    # 1. dense sampling
    pts = []
    boundary = []  # sample-point index at which each segment starts
    for i, seg in enumerate(sub):
        boundary.append(len(pts))
        pts.extend(_sample_segment(seg, budget.sample)[:-1])
    pts.append(sub[-1].end)
    P = np.array([[z.real, z.imag] for z in pts])

    # 2. closed seam: anchor the ring at the (duplicated) start point
    if closed:
        P = P[:-1]
        ring = np.vstack([P, P[0:1]])
    else:
        ring = P
    n = len(ring)

    # 3. points that must be kept: endpoints / seam and significant corners
    forced = np.zeros(n, dtype=bool)
    forced[0] = True
    forced[-1] = True
    for i in range(0 if closed else 1, n_seg):
        prev = sub[i - 1] if i > 0 else sub[-1]
        if _turning_angle(prev, sub[i]) > corner_angle:
            forced[boundary[i]] = True

    # 4. polyline simplification
    kept = _rdp(ring, budget.rdp, forced).tolist()

    # 5. group kept points into runs, split at forced points
    runs = []
    current = [kept[0]]
    for k in kept[1:]:
        current.append(k)
        if forced[k]:
            runs.append(current)
            current = [k]

    # 6. rebuild segments
    segments = []
    for run in runs:
        segments.extend(_emit_run(ring, run, budget))
    return segments


def _simplify(path, tol, corner_angle):
    budget = _Budget(tol)
    segments = []
    for sub in path.continuous_subpaths():
        if len(sub) == 0:
            continue
        segments.extend(_simplify_subpath(sub, budget, corner_angle))
    return Path(*segments)


# -------------------------------------------------------------------------
# topology check (opt-in)
# -------------------------------------------------------------------------

def _self_intersection_count(path):
    """Number of intersections between non-adjacent segments of path
    (bounding-box prefiltered).  Intersections at the shared endpoints of
    adjacent segments (including the seam of closed paths) are ignored."""
    segs = list(path)
    n = len(segs)
    boxes = [seg.bbox() for seg in segs]  # (xmin, xmax, ymin, ymax)
    count = 0
    for i in range(n - 2):
        bi = boxes[i]
        for j in range(i + 2, n):
            if i == 0 and j == n - 1 and segs[0].start == segs[-1].end:
                continue  # seam of a closed path
            bj = boxes[j]
            if bi[1] < bj[0] or bj[1] < bi[0] or bi[3] < bj[2] or bj[3] < bi[2]:
                continue
            count += len(segs[i].intersect(segs[j]))
    return count


def _topology_changed(original, simplified):
    o_sub = [s for s in original.continuous_subpaths() if len(s)]
    s_sub = [s for s in simplified.continuous_subpaths() if len(s)]
    if len(o_sub) != len(s_sub):
        return True
    for a, b in zip(o_sub, s_sub):
        if a.isclosed() != b.isclosed():
            return True
    return (_self_intersection_count(original)
            != _self_intersection_count(simplified))


# -------------------------------------------------------------------------
# main entry point
# -------------------------------------------------------------------------

def simplify_path(path, tolerance, corner_angle=DEFAULT_CORNER_ANGLE,
                  preserve_topology=False, verify=True):
    """Returns a simplified copy of ``path`` whose geometric deviation
    from ``path`` is at most ``tolerance``.

    Redundant (nearly collinear) straight segments are merged and runs of
    adjacent curve segments that can be fitted are replaced by fewer
    cubic Bezier segments.  The input path is never modified and the
    result is a deterministic function of ``path`` and ``tolerance``.

    Parameters
    ----------
    path : Path
        The path to simplify.
    tolerance : float
        A finite, non-negative bound on the maximum geometric deviation
        between the input and the output (verified by sampling).  A
        tolerance of 0 returns a copy of the input unchanged.
    corner_angle : float
        Joints whose turning angle exceeds this (radians, in
        [0, pi]) are treated as significant corners and preserved.
    preserve_topology : bool
        If True, the result is checked to have the same number of
        continuous subpaths, the same closedness, and the same number of
        self-intersections as the input.  If a simplification would
        change the topology (possible e.g. for self-intersecting paths),
        it is retried with a halved tolerance; a SimplificationError is
        raised if no attempt preserves the topology.
    verify : bool
        If True (default), the maximum geometric deviation between input
        and output is computed by sampling and a SimplificationError is
        raised if it exceeds ``tolerance`` (after retries with a halved
        working tolerance).

    Returns
    -------
    Path
        A new Path.  Subpath boundaries, closed seams, significant
        corners and (sub)path endpoints are preserved.

    Raises
    ------
    TypeError
        If ``path`` is not a Path.
    ValueError
        If ``tolerance`` is negative or not finite, or ``corner_angle``
        is outside [0, pi].
    SimplificationError
        If the error bound (or the requested topology preservation)
        cannot be satisfied.
    """
    if not isinstance(path, Path):
        raise TypeError("simplify_path expects a Path object, got %r"
                        % type(path))
    tol = float(tolerance)
    if not math.isfinite(tol) or tol < 0.0:
        raise ValueError("tolerance must be a finite, non-negative "
                         "number, got %r" % (tolerance,))
    if not 0.0 <= corner_angle <= math.pi:
        raise ValueError("corner_angle must be in [0, pi], got %r"
                         % (corner_angle,))
    if len(path) == 0:
        return Path()
    if tol == 0.0:
        return Path(*path)

    working_tol = tol
    failure = None
    for _ in range(_MAX_ATTEMPTS):
        result = _simplify(path, working_tol, corner_angle)
        if preserve_topology and _topology_changed(path, result):
            failure = ('the simplification changes the topology '
                       '(self-intersection structure) of the path')
            working_tol *= 0.5
            continue
        if verify:
            dev = max_deviation(path, result,
                                sample_tol=max(working_tol *
                                               _VERIFY_SAMPLE_FRAC, 1e-12))
            if dev > tol:
                failure = ('the verified maximum deviation (%r) exceeds '
                           'the tolerance' % dev)
                working_tol *= 0.5
                continue
        return result
    raise SimplificationError(
        "Unable to simplify the path within tolerance %r: %s.  Try a "
        "larger tolerance, or disable preserve_topology/verify if the "
        "trade-off is acceptable." % (tol, failure))
