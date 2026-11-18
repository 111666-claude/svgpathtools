"""Minimum-distance queries between path segments and paths.

This module finds the closest pair of points between two of the main
svgpathtools objects (``Line``, ``QuadraticBezier``, ``CubicBezier``,
``Arc``, or a ``Path`` made up of such segments).

The public entry points are:

* :func:`segment_distance` -- distance between two path segments,
* :func:`path_distance` -- distance between two (possibly multi-segment)
  paths,
* :func:`distance` -- dispatches on the argument types.

Each query returns a named tuple carrying the distance, both closest
points (as complex numbers) and the parameters that locate them.  The
routines only ever *read* the input geometry; the inputs are never
modified.

Algorithm
---------
Exact intersections (verified geometrically before being trusted)
short-circuit to a zero distance.  Otherwise the squared-distance
function ``|seg1.point(t1) - seg2.point(t2)|`` is minimized on the unit
square ``[0, 1] x [0, 1]``.  The search is seeded from every relevant
source -- the four endpoint projections, exact stationarity roots for
polynomial segments (sampled stationary brackets for arcs), and every
local minimum of a uniform parameter grid, each sharpened by nested-grid
refinement so narrow near-tangent contacts cannot fall between cells.
Distinct basins are then polished: L-BFGS-B locates the basin and exact
closest-point projection drives it to full convergence.  Endpoint
candidates are always included, so boundary minima cannot be missed.
Ties are resolved by a fixed lexicographic rule, making the output
reproducible bit-for-bit across runs.  When scipy is unavailable the
projection stage alone is used.
"""

from collections import namedtuple

import numpy as np

try:
    from scipy.optimize import minimize
    _scipy_minimize_available = True
except Exception:  # pragma: no cover - environments without scipy
    _scipy_minimize_available = False

from .path import Path, Line, QuadraticBezier, CubicBezier, Arc
from .polytools import polyroots01, real, imag

# Default geometric tolerance: a closest pair no farther apart than this
# is reported as touching (distance zero).
DEFAULT_DISTANCE_TOL = 1e-9

# Number of uniformly spaced sample points per segment used to seed the
# two-dimensional minimization.
_GRID_N = 21
# Refinement levels: around each coarse-grid local minimum the square is
# re-sampled on a nested grid this many times, halving the cell width
# each time.  This is what lets near-tangent minima narrower than a
# coarse cell still get a seed in their basin.
_GRID_REFINE_LEVELS = 3
_GRID_REFINE_N = 9
# How many of the best grid seeds get polished.
_GRID_SEEDS = 6
# How many of the lowest-distance distinct basins get the full
# L-BFGS-B polish; remaining seeds use cheap projection-only refinement.
_FULL_POLISH_SEEDS = 4
# Maximum alternating-refinement rounds per seed.  Convergence usually
# happens within a handful of rounds; the cap only guards pathological
# flat (near-tangent) contacts.
_POLISH_ITERS = 30
# Convergence threshold (in parameter space) for alternating refinement.
_POLISH_EPS = 1e-14
# Sample count used when locating the closest point on an Arc.
_ARC_SAMPLES = 41
# Golden-section iterations when refining a bracket on an Arc.
_GOLDEN_ITERS = 60

_GOLDEN_RATIO = (np.sqrt(5.0) - 1.0) / 2.0  # 0.618...


# Result containers ###########################################################

SegmentDistance = namedtuple(
    'SegmentDistance',
    ['dist', 'point1', 'point2', 't1', 't2'])
SegmentDistance.__doc__ = """\
Result of a segment-to-segment minimum-distance query.

Fields:
    dist (float): Euclidean distance between the two segments; ``0.0``
        when the segments touch within the configured tolerance.
    point1 (complex): closest point on the first segment.
    point2 (complex): closest point on the second segment.
    t1 (float): segment parameter locating ``point1``, in ``[0, 1]``.
    t2 (float): segment parameter locating ``point2``, in ``[0, 1]``.
"""

PathDistance = namedtuple(
    'PathDistance',
    ['dist', 'point1', 'point2', 'T1', 'T2',
     'seg1_idx', 'seg2_idx', 't1', 't2'])
PathDistance.__doc__ = """\
Result of a path-to-path minimum-distance query.

Fields:
    dist (float): Euclidean distance between the two paths; ``0.0`` when
        the paths touch within the configured tolerance.
    point1 (complex): closest point on the first path.
    point2 (complex): closest point on the second path.
    T1 (float): arc-length path parameter locating ``point1`` on the
        first path, in ``[0, 1]``.
    T2 (float): arc-length path parameter locating ``point2`` on the
        second path, in ``[0, 1]``.
    seg1_idx (int): index of the segment of the first path holding the
        closest point.
    seg2_idx (int): index of the segment of the second path holding the
        closest point.
    t1 (float): local segment parameter locating ``point1``, in
        ``[0, 1]``.
    t2 (float): local segment parameter locating ``point2``, in
        ``[0, 1]``.
"""


# Validation and small helpers ###############################################

def _validate_tol(tol):
    """Checks that ``tol`` is a non-negative finite real number."""
    if isinstance(tol, bool) or not isinstance(
            tol, (int, float, np.integer, np.floating)):
        raise TypeError(
            "tol must be a non-negative real number, got {}.".format(
                type(tol).__name__))
    tol = float(tol)
    if not np.isfinite(tol) or tol < 0.0:
        raise ValueError(
            "tol must be a finite, non-negative number, got {}.".format(tol))
    return tol


def _is_degenerate(seg):
    """A segment is degenerate when it collapses to a single point."""
    if isinstance(seg, Line):
        return seg.start == seg.end
    if isinstance(seg, (QuadraticBezier, CubicBezier)):
        start = seg.start
        return all(bp == start for bp in seg.bpoints())
    # Arc refuses zero radii and equal endpoints in its constructor, so
    # normally this cannot happen; stay defensive anyway.
    if isinstance(seg, Arc):
        return seg.start == seg.end
    raise TypeError(
        "Expected a Line, QuadraticBezier, CubicBezier, or Arc object, "
        "got {}.".format(type(seg).__name__))


def _clip01(t):
    """Clips a parameter to [0, 1], swallowing floating-point overshoot."""
    if t < 0.0:
        return 0.0
    if t > 1.0:
        return 1.0
    return float(t)


def _golden_section(f, lo, hi, iters=_GOLDEN_ITERS):
    """Deterministic golden-section minimization of ``f`` on [lo, hi]."""
    x1 = hi - _GOLDEN_RATIO * (hi - lo)
    x2 = lo + _GOLDEN_RATIO * (hi - lo)
    f1, f2 = f(x1), f(x2)
    for _ in range(iters):
        if f2 < f1:
            lo = x1
            x1, f1 = x2, f2
            x2 = lo + _GOLDEN_RATIO * (hi - lo)
            f2 = f(x2)
        else:
            hi = x2
            x2, f2 = x1, f1
            x1 = hi - _GOLDEN_RATIO * (hi - lo)
            f1 = f(x1)
    return (x1 + x2) / 2.0


def _arc_radial_min(arc, z):
    """Closest point on an Arc to a fixed point ``z``.

    The distance along an elliptic arc as a function of ``t`` may have
    several local minima.  A uniform scan locates every candidate
    bracket; golden-section refinement then picks the best one.  The
    endpoints are checked explicitly.
    """
    ts = np.linspace(0.0, 1.0, _ARC_SAMPLES)
    vals = np.array([abs(arc.point(t) - z) for t in ts])

    best_idx = int(np.argmin(vals))
    best_t, best_d = ts[best_idx], float(vals[best_idx])

    for i in range(1, _ARC_SAMPLES - 1):
        if vals[i] <= vals[i - 1] and vals[i] <= vals[i + 1]:
            t = _golden_section(
                lambda tau: abs(arc.point(tau) - z),
                ts[i - 1], ts[i + 1])
            d = abs(arc.point(t) - z)
            if d < best_d:
                best_t, best_d = t, float(d)
    return best_d, _clip01(best_t)


def _radial_min(seg, z):
    """Returns ``(distance, t)`` minimizing ``|seg.point(t) - z|``."""
    if isinstance(seg, Arc):
        return _arc_radial_min(seg, z)
    # Line/QuadraticBezier/CubicBezier all provide algebraic radialrange().
    d, t = seg.radialrange(z)[0]
    return float(d), _clip01(t)


class _RadialCache:
    """Per-query memoization of closest-point projections.

    Alternating refinement revisits (segment, point) pairs heavily; the
    underlying ``radialrange`` solves a polynomial each time.  Caching on
    a rounded point key avoids that repeated work while keeping the
    precision well below any geometric tolerance.
    """

    def __init__(self, digits=11):
        self._digits = digits
        self._store = {}

    def get(self, seg, z):
        key = (id(seg), round(float(z.real), self._digits),
               round(float(z.imag), self._digits))
        cached = self._store.get(key)
        if cached is None:
            cached = _radial_min(seg, z)
            self._store[key] = cached
        return cached


def _arc_stationary_roots(arc, z):
    """All interior parameters ``t`` making (arc.point(t) - z) orthogonal
    to the arc's tangent -- i.e. the local extrema of the distance from
    the fixed point ``z``.

    The arc is trigonometric, so the condition has no polynomial form;
    a dense uniform scan locates every sign change of the scalar
    derivative ``d/dt |arc.point(t) - z|^2`` and golden-section
    refinement brackets each local minimum.
    """
    # An ellipse-to-point distance has at most a handful of extrema, so
    # a moderate scan resolves every sign change; golden-section then
    # refines the brackets precisely.
    n = 33
    ts = np.linspace(0.0, 1.0, n)

    def derivative(t):
        delta = arc.point(t) - z
        tangent = arc.derivative(t)
        return 2.0 * (delta.real * tangent.real
                      + delta.imag * tangent.imag)

    roots = []
    previous_t, previous_g = ts[0], derivative(ts[0])
    for t in ts[1:]:
        g = derivative(t)
        if previous_g == 0.0:
            roots.append(float(previous_t))
        elif previous_g * g < 0.0:
            # Sign change: refine the stationary point.  Minimize the
            # distance itself on this bracket (a minimum of the squared
            # distance is also a minimum of the distance).
            root = _golden_section(
                lambda tau: abs(arc.point(tau) - z), previous_t, t)
            roots.append(float(root))
        previous_t, previous_g = t, g
    return roots


def _stationary_seeds(seg1, seg2):
    """Interior candidates on the stationarity locus of the distance.

    At an interior minimum the displacement vector is orthogonal to
    both tangent vectors, i.e. ``d_k/dt |p1(t1) - p2(t2)|^2 = 0``.  For
    polynomial segments each condition is a univariate polynomial whose
    roots give exact candidate parameters; for arcs a uniform scan plus
    stationary-root bracketing is used instead.
    """
    seeds = []

    # Arc participation: sweep the other segment and seed every local
    # minimum of the arc distance, so no arc basin can be missed.
    arc_sweep = 11
    if isinstance(seg1, Arc):
        arc, other = seg1, seg2
        for u in np.linspace(0.0, 1.0, arc_sweep):
            for t in _arc_stationary_roots(arc, other.point(u)):
                seeds.append((t, float(u)))
    if isinstance(seg2, Arc):
        arc, other = seg2, seg1
        for u in np.linspace(0.0, 1.0, arc_sweep):
            for t in _arc_stationary_roots(arc, other.point(u)):
                seeds.append((float(u), t))
    if isinstance(seg1, Arc) or isinstance(seg2, Arc):
        # Plus the symmetric radial projections for full coverage.
        grid = np.linspace(0.0, 1.0, arc_sweep)
        for t1 in grid:
            _, t2 = _radial_min(seg2, seg1.point(t1))
            seeds.append((float(t1), float(t2)))
        for t2 in grid:
            _, t1 = _radial_min(seg1, seg2.point(t2))
            seeds.append((float(t1), float(t2)))
        return seeds

    p1, p2 = seg1.poly(), seg2.poly()
    dp1, dp2 = p1.deriv(), p2.deriv()
    # (p1(t) - p2(u)) . p1'(t) = 0 as a polynomial in t with u fixed;
    # collect t-roots for a few fixed values of u ...
    for u in np.linspace(0.0, 1.0, 9):
        shift = p1 - p2(u)
        eq = (real(shift) * real(dp1) + imag(shift) * imag(dp1))
        for t in polyroots01(eq):
            seeds.append((_clip01(float(t)), float(u)))
    # ... and the symmetric u-roots for fixed t values.
    for t in np.linspace(0.0, 1.0, 9):
        shift = p2 - p1(t)
        eq = (real(shift) * real(dp2) + imag(shift) * imag(dp2))
        for u in polyroots01(eq):
            seeds.append((float(t), _clip01(float(u))))
    return seeds


def _refine_grid_basin(seg1, seg2, a0, b0, half):
    """Nested-grid refinement around the cell centered at (a0, b0).

    Returns the best (distance, t1, t2) pair found after
    ``_GRID_REFINE_LEVELS`` levels, each halving the search square.  This
    locates narrow minima (such as near-tangent contacts) that the coarse
    grid cannot resolve directly.
    """
    best = (abs(seg1.point(a0) - seg2.point(b0)), a0, b0)
    span = half
    for _ in range(_GRID_REFINE_LEVELS):
        lo_a, hi_a = max(0.0, a0 - span), min(1.0, a0 + span)
        lo_b, hi_b = max(0.0, b0 - span), min(1.0, b0 + span)
        for a in np.linspace(lo_a, hi_a, _GRID_REFINE_N):
            p1 = seg1.point(a)
            for b in np.linspace(lo_b, hi_b, _GRID_REFINE_N):
                d = abs(p1 - seg2.point(b))
                if d < best[0]:
                    best = (d, float(a), float(b))
        _, a0, b0 = best
        span *= 0.5
    return best


def _prune_seeds(seg1, seg2, seeds, min_sep=0.03, max_seeds=16):
    """Reduces seed redundancy before the expensive polishing stage.

    Exact duplicates are removed first, then seeds are visited in order
    of increasing distance and any seed lying within ``min_sep`` (in
    parameter space) of an already-kept seed is dropped, so each distinct
    basin is polished at most once.  Boundary parameters (exactly 0 or 1)
    are always retained.
    """
    unique = []
    seen = set()
    for a, b in seeds:
        key = (a, b)
        if key not in seen:
            seen.add(key)
            unique.append((a, b))

    ranked = sorted(
        unique, key=lambda ab: abs(seg1.point(ab[0]) - seg2.point(ab[1])))
    kept = []
    for a, b in ranked:
        on_boundary = a in (0.0, 1.0) or b in (0.0, 1.0)
        if not on_boundary and any(
                abs(a - ka) < min_sep and abs(b - kb) < min_sep
                for ka, kb in kept):
            continue
        kept.append((a, b))
        if len(kept) >= max_seeds:
            break
    return kept


def _line_line_distance(seg1, seg2):
    """Exact minimum distance between two non-degenerate Line segments.

    Closed-form solution of the two-segment problem: the minimum is at an
    endpoint projection or, when the supporting lines meet inside both
    segments, at their intersection.  Returns ``(dist, t1, t2)``.
    """
    p0, p1 = seg1.start, seg1.end
    q0, q1 = seg2.start, seg2.end
    u, v = p1 - p0, q1 - q0
    # Parameterize seg1 as p0 + s*u and seg2 as q0 + t*v; the separation
    # vector is w0 + t*v - s*u with w0 = q0 - p0.
    w0 = q0 - p0
    a = u.real * u.real + u.imag * u.imag          # |u|^2
    b = u.real * v.real + u.imag * v.imag          # u.v
    c = v.real * v.real + v.imag * v.imag          # |v|^2
    d = u.real * w0.real + u.imag * w0.imag        # u.w0
    e = v.real * w0.real + v.imag * w0.imag        # v.w0
    denom = a * c - b * b

    candidates = []
    if abs(denom) > 1e-15:
        # Interior stationary point of the supporting lines, solving
        # a*s - b*t = d and -b*s + c*t = -e (from the two orthogonality
        # conditions on the separation vector).
        s = (c * d - b * e) / denom
        t = (b * d - a * e) / denom
        if 0.0 <= s <= 1.0 and 0.0 <= t <= 1.0:
            candidates.append((s, t))
    # The global minimum of two compact segments always has at least one
    # parameter on a boundary, so check the four endpoint projections:
    # fixing t, minimize over s gives s = (b*t + d)/a; fixing s,
    # minimize over t gives t = (b*s - e)/c.
    candidates.append((_clip01(d / a), 0.0))
    candidates.append((_clip01((b + d) / a), 1.0))
    candidates.append((0.0, _clip01(-e / c)))
    candidates.append((1.0, _clip01((b - e) / c)))

    best = None
    for s, t in candidates:
        dist = abs(seg1.point(s) - seg2.point(t))
        if best is None or dist < best[0]:
            best = (dist, s, t)
    return best


def _collinear(seg1, seg2):
    """True when two Lines lie on the same infinite supporting line."""
    if not (isinstance(seg1, Line) and isinstance(seg2, Line)):
        return False
    direction = seg1.end - seg1.start
    if direction == 0:
        return False
    # A zero cross product means parallel; it only signals collinearity
    # when the segments' own endpoints also lie on that line.  The fixed
    # absolute epsilon is appropriate for ordinary SVG coordinates and
    # small enough not to swallow genuine gaps.
    eps = 1e-10
    offset1 = seg1.start - seg2.start
    offset2 = seg1.end - seg2.start
    cross1 = direction.real * offset1.imag - direction.imag * offset1.real
    cross2 = direction.real * offset2.imag - direction.imag * offset2.real
    return abs(cross1) <= eps and abs(cross2) <= eps


def _polish(seg1, seg2, t1, t2, full=True, cache=None):
    """Refine a seed pair (t1, t2) to a local minimum of the distance.

    With ``full=True`` a two-stage process runs: L-BFGS-B on the
    two-parameter squared-distance problem first locates the basin, then
    exact closest-point projection drives the pair to full convergence
    (important for flat near-tangent contacts where a gradient optimizer
    stops short of machine precision).  With ``full=False`` only the
    projection stage runs, which is much cheaper and adequate for seeds
    already sitting inside a basin.  Projection never increases the
    distance within a basin; if it would leave the optimizer's basin
    (distance grows) the optimizer result is kept instead.
    """
    t1, t2 = _clip01(t1), _clip01(t2)
    radial = cache.get if cache is not None else _radial_min

    if full and _scipy_minimize_available:
        def squared_distance(x):
            d = seg1.point(x[0]) - seg2.point(x[1])
            return d.real * d.real + d.imag * d.imag

        result = minimize(
            squared_distance, x0=np.array([t1, t2]),
            bounds=((0.0, 1.0), (0.0, 1.0)),
            method='L-BFGS-B',
            options={'ftol': 1e-15, 'gtol': 1e-13, 'maxiter': 500})
        opt_a, opt_b = _clip01(float(result.x[0])), _clip01(float(result.x[1]))
        opt_d = abs(seg1.point(opt_a) - seg2.point(opt_b))

        a, b = opt_a, opt_b
        previous = (a, b)
        for _ in range(_POLISH_ITERS):
            _, b = radial(seg2, seg1.point(a))
            _, a = radial(seg1, seg2.point(b))
            if (abs(a - previous[0]) < _POLISH_EPS
                    and abs(b - previous[1]) < _POLISH_EPS):
                break
            previous = (a, b)
        proj_d = abs(seg1.point(a) - seg2.point(b))
        if proj_d <= opt_d + 1e-12 * max(1.0, opt_d):
            return a, b
        return opt_a, opt_b

    previous = (t1, t2)
    for _ in range(_POLISH_ITERS):
        _, t2 = radial(seg2, seg1.point(t1))
        _, t1 = radial(seg1, seg2.point(t2))
        if (abs(t1 - previous[0]) < _POLISH_EPS
                and abs(t2 - previous[1]) < _POLISH_EPS):
            break
        previous = (t1, t2)
    return _clip01(t1), _clip01(t2)


def _select_candidate(seg1, seg2, candidates, tol):
    """Picks the best (t1, t2) candidate with deterministic tie-breaking."""
    scored = []
    seen = set()
    for t1, t2 in candidates:
        t1, t2 = _clip01(t1), _clip01(t2)
        key = (round(t1, 12), round(t2, 12))
        if key in seen:
            continue
        seen.add(key)
        d = abs(seg1.point(t1) - seg2.point(t2))
        scored.append((float(d), t1, t2))

    best_d = min(s[0] for s in scored)
    # Only candidates agreeing to floating-point precision count as the
    # same minimum; among them take the lexicographically smallest
    # parameter pair so repeated calls always return the same solution.
    # A relative threshold keeps this meaningful at large magnitudes,
    # while an absolute floor avoids collapsing distinct shallow minima
    # at small distances.
    tie_eps = max(1e-12, 1e-12 * best_d)
    tied = [s for s in scored if s[0] <= best_d + tie_eps]
    d, t1, t2 = min(tied, key=lambda s: (s[1], s[2]))

    if d <= tol:
        # Report a canonical zero: pick the lexicographically smallest
        # parameter pair within the tolerance so the returned points
        # themselves agree to the same tolerance.
        touching = [s for s in scored if s[0] <= tol]
        d, t1, t2 = min(touching, key=lambda s: (s[1], s[2]))
        d = 0.0
    return d, t1, t2


# Segment-to-segment distance ################################################

def segment_distance(seg1, seg2, tol=DEFAULT_DISTANCE_TOL):
    """Minimum distance between two path segments.

    Args:
        seg1 (Line | QuadraticBezier | CubicBezier | Arc): first segment.
        seg2 (Line | QuadraticBezier | CubicBezier | Arc): second segment.
        tol (float): non-negative geometric tolerance.  When the closest
            pair of points is no farther apart than ``tol`` the segments
            are treated as touching and the returned ``dist`` is exactly
            ``0.0``.

    Returns:
        SegmentDistance: the distance, both closest points and their
        segment parameters.

    Raises:
        TypeError: if an argument is not a path segment or ``tol`` is
            not numeric.
        ValueError: if ``tol`` is negative or non-finite.

    Notes:
        Zero-length (degenerate) segments are treated as the single point
        they collapse to.  Coincident segments return a zero distance at
        the shared start point.
    """
    tol = _validate_tol(tol)
    if not (isinstance(seg1, (Line, QuadraticBezier, CubicBezier, Arc))
            and isinstance(seg2, (Line, QuadraticBezier, CubicBezier, Arc))):
        raise TypeError(
            "segment_distance() expects two path segments (Line, "
            "QuadraticBezier, CubicBezier, or Arc).")

    # Degenerate segments collapse to a point.
    deg1, deg2 = _is_degenerate(seg1), _is_degenerate(seg2)
    if deg1 and deg2:
        d = abs(seg1.start - seg2.start)
        return SegmentDistance(0.0 if d <= tol else d,
                               seg1.start, seg2.start, 0.0, 0.0)
    if deg1:
        d, t2 = _radial_min(seg2, seg1.start)
        return SegmentDistance(0.0 if d <= tol else d,
                               seg1.start, seg2.point(t2), 0.0, t2)
    if deg2:
        d, t1 = _radial_min(seg1, seg2.start)
        return SegmentDistance(0.0 if d <= tol else d,
                               seg1.point(t1), seg2.start, t1, 0.0)

    # Coincident segments: every parameter pair with t1 == t2 is a zero.
    # Pick the lexicographically smallest one deterministically.
    if seg1 == seg2:
        return SegmentDistance(0.0, seg1.start, seg2.start, 0.0, 0.0)

    # Line-Line has an exact closed-form solution (also keeps the
    # dependency-free fallback accurate without any iterative solver).
    if isinstance(seg1, Line) and isinstance(seg2, Line):
        d, t1, t2 = _line_line_distance(seg1, seg2)
        if d <= tol:
            d = 0.0
        return SegmentDistance(d, seg1.point(t1), seg2.point(t2), t1, t2)

    # Exact intersections short-circuit to a zero distance.  The segment
    # intersection routines intentionally do not handle overlapping
    # geometry, and several Arc paths (notably Arc-Arc contacts and
    # collinear lines) return either spurious or mis-parameterized
    # contacts; those cases are guarded explicitly and otherwise left to
    # the general minimizer, which verifies distances geometrically.
    arc_arc = isinstance(seg1, Arc) and isinstance(seg2, Arc)
    if not _collinear(seg1, seg2) and not arc_arc:
        try:
            hits = seg1.intersect(seg2)
        except Exception:
            hits = []
        # A reported root is trusted as a real contact only when its
        # initial residual is already at root-solving accuracy.  A loose
        # threshold here would accept phantom roots (the Bezier/Line
        # solvers occasionally return parameters that do not lie on both
        # curves), and closest-point projection would then manufacture a
        # false contact.  Genuinely near-tangent roots that the solver
        # cannot resolve this tightly are left to the general minimizer.
        verified = []
        for a, b in hits:
            a, b = _clip01(a), _clip01(b)
            if abs(seg1.point(a) - seg2.point(b)) <= 1e-8:
                verified.append(
                    (abs(seg1.point(a) - seg2.point(b)), a, b))
        if verified:
            _, t1, t2 = min(verified)
            return SegmentDistance(0.0, seg1.point(t1), seg2.point(t2),
                                   t1, t2)

    seeds = []
    cache = _RadialCache()

    # Boundary candidates: closest point to each of the four endpoints.
    for tb in (0.0, 1.0):
        _, t2 = cache.get(seg2, seg1.point(tb))
        seeds.append((tb, t2))
        _, t1 = cache.get(seg1, seg2.point(tb))
        seeds.append((t1, tb))

    # Interior stationary-point candidates (exact roots for polynomial
    # segments, sampled brackets for arcs).
    seeds.extend(_stationary_seeds(seg1, seg2))

    # Interior seeds: uniform grid.  Every grid local minimum seeds a
    # polish run (the squared-distance surface can have several shallow
    # basins for wiggly curves), and each such minimum is first sharpened
    # by nested-grid refinement so near-tangent contacts narrower than a
    # coarse grid cell are still seeded in the correct basin.
    grid = np.linspace(0.0, 1.0, _GRID_N)
    half = 1.0 / (_GRID_N - 1)
    grid_vals = np.array(
        [[abs(seg1.point(a) - seg2.point(b)) for b in grid]
         for a in grid])
    local_minima = []
    for i in range(1, _GRID_N - 1):
        for j in range(1, _GRID_N - 1):
            v = grid_vals[i, j]
            if (v <= grid_vals[i - 1, j] and v <= grid_vals[i + 1, j]
                    and v <= grid_vals[i, j - 1]
                    and v <= grid_vals[i, j + 1]):
                local_minima.append((float(grid[i]), float(grid[j])))
    # Include the globally best coarse cells, which may lie on the
    # boundary that the interior local-minimum scan skips.
    flat_idx = np.argsort(grid_vals, axis=None)[:_GRID_SEEDS]
    for flat in flat_idx:
        i, j = np.unravel_index(flat, grid_vals.shape)
        local_minima.append((float(grid[i]), float(grid[j])))

    # Refine each basin on a nested grid before polishing.
    for a0, b0 in local_minima:
        _, a, b = _refine_grid_basin(seg1, seg2, a0, b0, half)
        seeds.append((a, b))

    # Prune before polishing: drop exact duplicates and cluster seeds in
    # the same basin, keeping the lowest-distance representative of each.
    seeds = _prune_seeds(seg1, seg2, seeds)

    # Cheap projection-only polish for every distinct basin guarantees a
    # local minimum everywhere; the few lowest-distance basins then get
    # the full L-BFGS-B treatment, which is what resolves shallow or
    # near-tangent minima to machine precision.
    seeds_ranked = sorted(
        seeds,
        key=lambda ab: abs(seg1.point(ab[0]) - seg2.point(ab[1])))
    full_set = set(seeds_ranked[:_FULL_POLISH_SEEDS])
    candidates = list(seeds)
    for a, b in seeds:
        candidates.append(
            _polish(seg1, seg2, a, b,
                    full=((a, b) in full_set), cache=cache))

    d, t1, t2 = _select_candidate(seg1, seg2, candidates, tol)
    return SegmentDistance(d, seg1.point(t1), seg2.point(t2), t1, t2)


# Path-to-path distance ######################################################

def path_distance(path1, path2, tol=DEFAULT_DISTANCE_TOL):
    """Minimum distance between two paths.

    Args:
        path1 (Path): first path.
        path2 (Path): second path.
        tol (float): non-negative geometric tolerance; see
            :func:`segment_distance` for its semantics.

    Returns:
        PathDistance: the distance, both closest points, the arc-length
        path parameters (``T1``, ``T2``) locating them, the winning
        segment indices and the local segment parameters.

    Raises:
        TypeError: if an argument is not a ``Path`` or ``tol`` is not
            numeric.
        ValueError: if either path contains no segments or ``tol`` is
            invalid.

    Notes:
        The inputs are never modified.  Paths containing zero-length
        segments are handled; those segments are treated as points.
        Identical (coincident) paths return a zero distance at their
        first segment's start point.
    """
    tol = _validate_tol(tol)
    if not (isinstance(path1, Path) and isinstance(path2, Path)):
        raise TypeError("path_distance() expects two Path objects.")
    if len(path1) == 0 or len(path2) == 0:
        raise ValueError(
            "Cannot compute the distance of an empty path (a path with "
            "no segments).")

    # Coincident paths: deterministic zero at the common start point.
    if path1 == path2:
        return PathDistance(0.0, path1.point(0.0), path2.point(0.0),
                            0.0, 0.0, 0, 0, 0.0, 0.0)

    # t2T() needs arc-length fractions; this only fills a read-only cache.
    path1._calc_lengths()
    path2._calc_lengths()

    candidates = []
    for i, seg1 in enumerate(path1):
        for j, seg2 in enumerate(path2):
            res = segment_distance(seg1, seg2, tol=tol)
            T1 = path1.t2T(i, res.t1)
            T2 = path2.t2T(j, res.t2)
            candidates.append(
                (res.dist, T1, T2, i, j, res.t1, res.t2))

    best_d = min(c[0] for c in candidates)
    # Numerically indistinguishable segment pairs (agreeing to
    # floating-point precision) count as ties; among them take the
    # lexicographically smallest (T1, T2) so the result is reproducible
    # regardless of segment enumeration order.
    tie_eps = max(1e-12, 1e-12 * best_d)
    best = min((c for c in candidates if c[0] <= best_d + tie_eps),
               key=lambda c: (c[1], c[2]))

    d, T1, T2, i, j, t1, t2 = best
    return PathDistance(d, path1[i].point(t1), path2[j].point(t2),
                        T1, T2, i, j, t1, t2)


# Dispatch ###################################################################

def distance(curve1, curve2, tol=DEFAULT_DISTANCE_TOL):
    """Minimum distance between two segments or paths.

    Dispatches to :func:`segment_distance` when both arguments are
    individual segments, or to :func:`path_distance` otherwise (a bare
    segment is wrapped in a one-segment :class:`~svgpathtools.path.Path`).

    See the two wrapped functions for arguments, return types and error
    handling.
    """
    segment_types = (Line, QuadraticBezier, CubicBezier, Arc)
    if isinstance(curve1, segment_types) and isinstance(curve2, segment_types):
        return segment_distance(curve1, curve2, tol=tol)

    path1 = curve1 if isinstance(curve1, Path) else Path(curve1)
    path2 = curve2 if isinstance(curve2, Path) else Path(curve2)
    return path_distance(path1, path2, tol=tol)
