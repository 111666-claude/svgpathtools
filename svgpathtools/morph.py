"""This submodule adds closed-path morphing (shape interpolation) to
svgpathtools.

The :func:`morph_paths` function takes two closed paths (each possibly made of
several closed subpaths, e.g. an outer contour with holes) and a progress
value ``t`` between 0 and 1, and returns a drawable :class:`Path` whose
geometry is an intermediate contour.

Why a dedicated module is needed
---------------------------------
Two poses of the same icon are generally described with different numbers of
segments (a rectangle is four lines, a circle is four arcs or many beziers),
so interpolating segments one-to-one is not even possible and, when the
segment counts happen to match, it tears the outline apart because segment
boundaries sit at different places on the two shapes.

The approach used here instead:

1. Every subpath is converted to cubic bezier segments (lines and quadratics
   are degree-elevated exactly; arcs are approximated to a controlled
   geometric tolerance).
2. Both subpaths of a matched pair are re-parameterized by *arc length* and
   split at the union of each other's normalized arc-length breakpoints, so
   the resulting piece lists have equal length and corresponding pieces
   cover equal fractions of the perimeter.
3. Corresponding cubic control points are linearly interpolated.

Closed loops with different start points or opposite orientations are aligned
first (cyclic shift / reversal chosen to minimize point-wise distortion), so
the contour does not twist during the morph.
"""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from itertools import permutations

import numpy as np

from .path import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                   concatpaths, is_path_segment)

__all__ = ['MorphError', 'morph_paths']


# Exceptions ##################################################################

class MorphError(ValueError):
    """Raised when two paths cannot be morphed into each other (incompatible
    subpath/hole topology, degenerate zero-length contours, non-finite
    geometry, or an invalid progress value)."""


# Default numerical parameters ###############################################

# Number of samples used when aligning the cyclic start points of two loops.
_DEFAULT_ALIGN_SAMPLES = 128

# Geometric tolerance (in the paths' own coordinate units) used when an Arc
# is approximated by cubic beziers.  A sampled error grid checks the actual
# approximation achieved.
_DEFAULT_ARC_TOLERANCE = 1e-6

# Relative tolerance used to decide that two normalized arc-length fractions
# describe the same point (guards against floating-point drift around
# segment joints).
_FRACTION_TOL = 1e-9


# Small helpers ##############################################################

def _finite(z):
    return np.isfinite(z.real) and np.isfinite(z.imag)


def _signed_area(subpath, samples=64):
    """Signed area of a closed subpath, positive for counter-clockwise
    parameterization (in the mathematical, y-up sense used by the complex
    plane).  Arcs are handled by point sampling."""
    total = 0.0
    for seg in subpath:
        if isinstance(seg, Arc):
            ts = np.linspace(0.0, 1.0, samples + 1)
            pts = [seg.point(float(t)) for t in ts]
            for p0, p1 in zip(pts[:-1], pts[1:]):
                total += (p0.real * p1.imag - p1.real * p0.imag) * 0.5
        else:
            # Integral of x*dy over the segment via its complex polynomial.
            poly = seg.poly()
            x = np.poly1d(poly.coeffs.real)
            dy = np.poly1d(poly.coeffs.imag).deriv()
            integrand = (x * dy).integ()
            total += float(integrand(1) - integrand(0))
    return total


def _centroid(subpath, samples=64):
    """Arc-length-weighted average point of the subpath, used only as a
    stable position summary for pairing up subpaths of two drawings."""
    total_len = 0.0
    acc = 0j
    for seg in subpath:
        length = seg.length()
        if length <= 0:
            continue
        if isinstance(seg, Arc):
            ts = np.linspace(0.0, 1.0, samples + 1)
        else:
            # Gauss-ish midpoint rule with a handful of samples is plenty for
            # a position summary of a line/bezier segment.
            ts = np.linspace(0.0, 1.0, 9)
        pts = np.array([seg.point(float(t)) for t in ts])
        mid = (pts[:-1] + pts[1:]) * 0.5
        weights = np.abs(np.diff(pts))
        wsum = weights.sum()
        if wsum > 0:
            acc += sum(m * w for m, w in zip(mid, weights))
            total_len += wsum
    if total_len <= 0:
        raise MorphError("Subpath has zero length and cannot be morphed.")
    return acc / total_len


def _as_cubic(seg, arc_tolerance=_DEFAULT_ARC_TOLERANCE):
    """Return a list of CubicBezier segments geometrically equivalent to
    `seg`.  Lines and quadratic beziers are converted exactly; arcs are
    approximated until the sampled error is below `arc_tolerance`."""
    if isinstance(seg, CubicBezier):
        return [CubicBezier(seg.start, seg.control1, seg.control2, seg.end)]
    if isinstance(seg, QuadraticBezier):
        p0, p1, p2 = seg.bpoints()
        c1 = p0 + (2.0 / 3.0) * (p1 - p0)
        c2 = p2 + (2.0 / 3.0) * (p1 - p2)
        return [CubicBezier(p0, c1, c2, p2)]
    if isinstance(seg, Line):
        p0, p1 = seg.bpoints()
        return [CubicBezier(p0, p0, p1, p1)]
    if isinstance(seg, Arc):
        return _arc_to_cubics(seg, arc_tolerance)
    raise MorphError("Unsupported segment type: {!r}".format(type(seg)))


def _arc_max_error(arc, cubics, check_samples=24):
    """Largest sampled distance between an Arc and the cubic-bezier pieces
    approximating it.  Both are evaluated at the same segment parameter."""
    worst = 0.0
    n = len(cubics)
    for k, cubic in enumerate(cubics):
        # Exclude the shared endpoints of adjacent pieces (error is zero
        # there by construction); sample the interior.
        ts = (np.arange(1, check_samples + 1) /
              (check_samples + 1.0) + k) / n
        for t in ts:
            exact = arc.point(float(t))
            approx = cubic.point(float(t * n - k))
            worst = max(worst, abs(exact - approx))
    return worst


def _arc_to_cubics(arc, tolerance):
    """Approximate an Arc with cubic beziers whose sampled geometric error
    does not exceed `tolerance`."""
    sweep_deg = abs(float(arc.delta))
    # 30 degrees per cubic is the usual safe starting point.
    n = max(1, int(np.ceil(sweep_deg / 30.0)))
    # Cubic arc error scales like O(1/n^4), so even very tight tolerances
    # need at most a few hundred pieces; this cap only guards against
    # non-termination.
    for _ in range(10000):
        cubics = list(arc.as_cubic_curves(n))
        # A safety factor of 2 accounts for the error between the finite
        # sample grid and the true maximum of the approximation error.
        if _arc_max_error(arc, cubics) <= max(tolerance * 0.5, 1e-12):
            return cubics
        n += 1
    raise MorphError("Could not approximate arc to the requested tolerance.")


def _to_cubic_path(subpath, arc_tolerance):
    """Degree-elevate / approximate every segment of `subpath` to cubic
    beziers, dropping interior zero-length pieces.

    Returns a tuple ``(path, joint_fractions)`` where ``joint_fractions`` are
    normalized arc-length fractions of the *geometrically meaningful*
    segment joints (joints of the original line/bezier segments).  Joints
    introduced only to approximate an arc are excluded, since they carry no
    corner information.
    """
    pieces = []
    meaningful_joints = [0.0]
    cumulative = 0.0
    for seg in subpath:
        for cubic in _as_cubic(seg, arc_tolerance):
            if abs(cubic.end - cubic.start) > 0.0:
                pieces.append(cubic)
                cumulative += cubic.length()
        if not isinstance(seg, Arc):
            meaningful_joints.append(cumulative)
    total = cumulative
    if not pieces or total <= 0.0:
        raise MorphError("Subpath has zero length and cannot be morphed.")
    fractions = sorted({f / total for f in meaningful_joints
                        if 0.0 <= f <= total})
    return Path(*pieces), fractions


# Closed-loop handling #######################################################

def _rotate_cubic_start(path, fraction):
    """Return a closed cubic path whose parameterization starts at arc-length
    fraction `fraction` of `path`.  The geometry is unchanged.  The two
    cropped halves are re-joined by snapping the shared junction, so the
    result stays exactly continuous despite floating-point drift."""
    if fraction <= 0.0 or fraction >= 1.0:
        return Path(*path)
    first = _crop_closed_loop(path, fraction, 1.0)
    rest = _crop_closed_loop(path, 0.0, fraction)
    new = Path(*[seg for seg in first])
    junction = new[-1].end
    for seg in rest:
        seg = CubicBezier(junction, seg.control1, seg.control2, seg.end)
        new.append(seg)
        junction = seg.end
    # Snap the final endpoint onto the new start to close the loop exactly.
    last = new[-1]
    new[-1] = CubicBezier(last.start, last.control1, last.control2,
                          new[0].start)
    return new


def _sampled_points(path, k):
    """k evenly arc-length-spaced points around the closed path."""
    return [path.point(i / k) for i in range(k)]


def _cyclic_cost(points_a, points_b, shift):
    """Sum of squared distances for a given cyclic offset of `points_b`."""
    k = len(points_a)
    total = 0.0
    for i in range(k):
        d = points_a[i] - points_b[(i + shift) % k]
        total += d.real * d.real + d.imag * d.imag
    return total


def _fraction_cost(points_a, points_b, fraction):
    """Cost when `points_b` is cyclically shifted by an arbitrary arc-length
    fraction; intermediate points of b are evaluated by linear interpolation
    on the periodic sample grid."""
    k = len(points_b)
    pos = fraction * k
    i0 = int(np.floor(pos))
    w = pos - i0
    total = 0.0
    for i in range(k):
        zb = ((1.0 - w) * points_b[(i + i0) % k] +
              w * points_b[(i + i0 + 1) % k])
        d = points_a[i] - zb
        total += d.real * d.real + d.imag * d.imag
    return total


def _best_cyclic_shift(points_a, points_b):
    """Find the cyclic offset of `points_b` (same-length periodic lists)
    minimizing the summed squared distance to `points_a`.  A coarse grid
    search is followed by a local refinement on a finer grid."""
    k = len(points_a)
    coarse_shift = min(range(k),
                       key=lambda s: _cyclic_cost(points_a, points_b, s))

    # Refine between the neighboring coarse fractions on a 10x finer grid.
    lo = (coarse_shift - 1) / k
    hi = (coarse_shift + 1) / k
    fine_k = 10
    best_fraction, best_cost = coarse_shift / k, np.inf
    for j in range(fine_k + 1):
        fraction = lo + (hi - lo) * j / fine_k
        cost = _fraction_cost(points_a, points_b, fraction)
        if cost < best_cost:
            best_cost, best_fraction = cost, fraction
    return best_fraction % 1.0, best_cost


def _snap_to_fraction(fraction, candidate_fractions, window):
    """If any fraction in `candidate_fractions` lies within `window` (in
    normalized arc length, on the periodic [0, 1) domain) of `fraction`,
    return the nearest one; otherwise return `fraction` unchanged.  This
    keeps corner-based shapes (e.g. a rectangle) registered
    corner-to-corner instead of landing the seam on an edge."""
    best, best_dist = fraction, window
    for v in candidate_fractions:
        dist = min(abs(v - fraction), abs(v - fraction + 1.0),
                   abs(v - fraction - 1.0))
        if dist < best_dist:
            best, best_dist = v, dist
    return best % 1.0


def _align_loops(sub_a, sub_b, joints_a, joints_b, ccw_a, ccw_b,
                 align_samples):
    """Make `sub_b` travel in the same direction as `sub_a` and make both
    loops start at the corresponding perimeter point.  Returns
    ``(aligned_a, aligned_b, joints_a', joints_b')`` where the last two
    items are the meaningful joint fractions mapped through the same cyclic
    shift as the paths.

    The orientation and cyclic offset are chosen to minimize the point-wise
    distance between equally arc-length-spaced samples, which prevents
    unnecessary twisting of the contour.  The chosen start is then snapped to
    a nearby geometrically meaningful joint (a real corner of either input)
    when one exists, so polygons stay registered on their corners.

    `joints_a`/`joints_b` are the normalized arc-length fractions of the
    original, pre-arc-approximation segment joints.  `ccw_a`/`ccw_b` report
    the counter-clockwise-ness of each input's signed area."""
    k = align_samples
    pts_a = _sampled_points(sub_a, k)

    # Opposite signed-area signs mean the loops run in opposite directions;
    # reverse b so the morph does not fold the contour over itself.
    do_reverse = (ccw_a != ccw_b)
    b = sub_b.reversed() if do_reverse else sub_b
    # Reversal maps arc-length fraction f to 1 - f (mod 1).
    joints_of_b = sorted({(1.0 - f) % 1.0 for f in joints_b}) \
        if do_reverse else sorted(set(joints_b))
    pts_b = _sampled_points(b, k)
    fraction, _ = _best_cyclic_shift(pts_a, pts_b)

    # Snap to a meaningful joint when one is within roughly one coarse sample
    # step.  Only joints that are real corners in at least one input count;
    # smooth loops (e.g. a circle approximated from arcs) contribute nothing
    # here so their seam stays wherever the best cyclic alignment put it.
    # Both loops must end up starting at the same chosen fraction, so any
    # snap is applied to both.
    window = 1.5 / k
    snapped = _snap_to_fraction(fraction, joints_of_b, window)
    if snapped != fraction:
        fraction = snapped
    snapped = _snap_to_fraction(fraction, joints_a, window)
    if snapped != fraction:
        fraction = snapped

    aligned_a = _rotate_cubic_start(sub_a, fraction)
    aligned_b = _rotate_cubic_start(b, fraction)

    # Map each loop's meaningful joints through the same cyclic shift, so
    # the downstream breakpoint computation can snap back to exact values.
    def shifted(joints):
        return sorted({(f - fraction) % 1.0 for f in joints})

    return aligned_a, aligned_b, shifted(joints_a), shifted(joints_of_b)


# Arc-length re-parameterization #############################################

def _arc_length_breakpoints(path, canonical=None):
    """Normalized cumulative arc-length breakpoints of the path:
    [0, l1/L, l2/L, ..., 1].

    If `canonical` is given, each computed fraction within a relative
    tolerance of a canonical value is replaced by that value.  This keeps the
    joints that were only re-derived numerically (e.g. after rotating the
    loop start) bit-identical to their known arc-length positions, so later
    cropping lands exactly on the segment boundary."""
    path._calc_lengths()
    fracs = [0.0]
    acc = 0.0
    for w in path._lengths:
        acc += w
        fracs.append(acc)
    fracs[-1] = 1.0
    if canonical:
        out = []
        canon = sorted(canonical)
        for f in fracs:
            tol = _FRACTION_TOL * max(1.0, abs(f))
            match = next((c for c in canon if abs(c - f) <= tol), None)
            out.append(match if match is not None else f)
        return out
    return fracs


def _merge_breakpoints(fracs_a, fracs_b):
    """Union of two breakpoint sets, with near-duplicates collapsed onto a
    single canonical value."""
    all_fracs = sorted(set([0.0, 1.0] +
                           [f for f in fracs_a if 0.0 < f < 1.0] +
                           [f for f in fracs_b if 0.0 < f < 1.0]))
    # Cluster fractions separated by no more than the relative tolerance and
    # keep each cluster's smallest member as its canonical value.
    clusters = []
    for f in all_fracs:
        if clusters and f - clusters[-1][-1] <= _FRACTION_TOL * max(1.0, abs(f)):
            clusters[-1].append(f)
        else:
            clusters.append([f])
    cleaned = [cluster[0] for cluster in clusters]
    cleaned[0] = 0.0
    cleaned[-1] = 1.0
    return cleaned


def _segment_and_t(path, f):
    """Like Path.T2t, but snaps results that land within a small absolute
    tolerance of a segment boundary onto that boundary (t = 0 or t = 1), so a
    breakpoint meant to sit on a joint cannot spill into the neighboring
    segment because of floating-point drift.

    The tolerance is absolute on the segment parameter (not relative to the
    arc-length fraction), otherwise a breakpoint genuinely 1e-7 inside a
    very short segment could be mistaken for a joint."""
    idx, t = path.T2t(f)
    n = len(path)
    tol = 1e-11
    if t <= tol:
        # Treat as the start of this segment (unless we are at the very
        # beginning of the whole loop).
        if not (idx == 0 and f < tol):
            return idx, 0.0
    if t >= 1.0 - tol:
        # Treat as the end of this segment (unless at the very end).
        if not (idx == n - 1 and f > 1.0 - tol):
            return idx, 1.0
    return idx, t


def _crop_closed_loop(path, f0, f1):
    """Crop a single closed subpath between two normalized arc-length
    fractions with 0 <= f0 < f1 <= 1.

    The caller guarantees every segment joint is a breakpoint, so in exact
    arithmetic each crop lies inside one segment.  On very short segments the
    inverse-arc-length lookup can drift across a joint by a hair; when that
    happens (the crop spans exactly one adjacent joint with both parameters
    near the boundary), the interval is collapsed onto the segment containing
    its start.  The sliver dropped is below floating-point resolution."""
    if f1 <= f0:
        return Path()
    if f0 == 0.0 and f1 == 1.0:
        return Path(*path)

    n = len(path)
    i0, t0 = _segment_and_t(path, f0)
    i1, t1 = _segment_and_t(path, f1)
    # A breakpoint snapped to a segment end is the start of the next segment
    # (the loop is closed); a breakpoint snapped to a start is the end of the
    # previous one.  Normalize so that t0 in [0, 1) and t1 in (0, 1].
    if t0 == 1.0:
        i0 = (i0 + 1) % n
        t0 = 0.0
    if t1 == 0.0 and f1 != 0.0:
        i1 = (i1 - 1) % n
        t1 = 1.0

    if i0 == i1:
        return Path(path[i0].cropped(t0, t1))

    # Drift signature: the only joint crossed is the immediate next one, with
    # both endpoints hugging it.  Collapse onto the segment of f0 (its tail).
    if i1 == (i0 + 1) % n and t0 > 0.5 and t1 < 0.5:
        return Path(path[i0].cropped(t0, 1.0))

    # Genuine multi-joint span (only possible for a full [0, 1]-style crop):
    # tail piece, whole middle segments, then head piece.
    out = Path(path[i0].cropped(t0, 1.0))
    idx = (i0 + 1) % n
    while idx != i1:
        out.append(path[idx])
        idx = (idx + 1) % n
    out.append(path[i1].cropped(0.0, t1))
    return out


def _subdivide(path, breakpoints):
    """Split a closed cubic path into per-global-interval piece lists.

    Every segment joint of both morphed paths is a global breakpoint, so in
    exact arithmetic each global interval lies inside one segment of each
    path and each cell gets exactly one piece.  Membership of a breakpoint in
    a segment is decided by comparing arc-length fractions (not by an
    inverse-arc-length lookup), which stays reliable even when the path has
    hundreds of very short segments.

    Returns a nested list ``cells[k]`` with one CubicBezier per global
    interval.  The pieces share every junction exactly and the last piece
    ends at the first piece's start, so the flattened result is exactly
    closed and continuous."""
    path._calc_lengths()
    joints = _arc_length_breakpoints(path)  # [0, l1/L, ..., 1]
    tol = _FRACTION_TOL

    # Mark which cell each segment joint corresponds to, by locating every
    # joint inside the global breakpoint list.  A joint that coincides with a
    # breakpoint is the boundary between two cells and produces no extra cut.
    n_cells = len(breakpoints) - 1
    cells = [None] * n_cells
    bp = breakpoints
    for seg_idx, seg in enumerate(path):
        j0, j1 = joints[seg_idx], joints[seg_idx + 1]
        # Global cells whose interior overlaps this segment's arc-length
        # span [j0, j1].  Find the first breakpoint > j0 and the last
        # breakpoint < j1 (within tolerance), then cut between them.
        lo = bisect_right(bp, j0 + tol * max(1.0, abs(j0))) - 1
        hi = bisect_left(bp, j1 - tol * max(1.0, abs(j1))) - 1
        # Cut fractions (global) inside (j0, j1).
        cuts = bp[lo + 1:hi + 1]
        edges_global = [j0] + cuts + [j1]
        span = j1 - j0
        for k, (g0, g1) in enumerate(zip(edges_global[:-1],
                                          edges_global[1:])):
            u0, u1 = (g0 - j0) / span, (g1 - j0) / span
            piece = seg.cropped(u0, u1)
            cell_index = lo + k
            if cells[cell_index] is not None:
                # A breakpoint that is a segment joint sits on the boundary,
                # so no cell should ever be filled twice.
                raise MorphError(
                    "Internal error: overlapping pieces in cell {}."
                    .format(cell_index))
            cells[cell_index] = piece

    if any(cell is None for cell in cells):
        raise MorphError(
            "Internal error: an interval received no piece during "
            "re-parameterization.")

    # Snap each junction to the canonical previous endpoint so floating-point
    # drift cannot open the contour, then close the loop.
    pieces = [CubicBezier(cells[0].start, cells[0].control1,
                          cells[0].control2, cells[0].end)]
    for seg in cells[1:]:
        pieces.append(CubicBezier(pieces[-1].end, seg.control1,
                                  seg.control2, seg.end))
    last = pieces[-1]
    pieces[-1] = CubicBezier(last.start, last.control1, last.control2,
                             pieces[0].start)
    return [[cubic] for cubic in pieces]


def _interpolate_cubics(cells_a, cells_b, t):
    """Linearly interpolate the control points of corresponding cubics.

    `cells_a`/`cells_b` are nested per-global-interval piece lists with
    matching per-cell counts (see :func:`_subdivide`); they are flattened
    before interpolation."""
    pieces_a = [cubic for cell in cells_a for cubic in cell]
    pieces_b = [cubic for cell in cells_b for cubic in cell]
    if len(pieces_a) != len(pieces_b):
        raise MorphError(
            "Internal error: piece counts differ after re-parameterization "
            "({} vs {}).".format(len(pieces_a), len(pieces_b)))
    out = []
    for ca, cb in zip(pieces_a, pieces_b):
        pa, pb = ca.bpoints(), cb.bpoints()
        blended = tuple((1.0 - t) * a + t * b for a, b in zip(pa, pb))
        if not all(_finite(z) for z in blended):
            raise MorphError(
                "Non-finite coordinate produced while morphing (t={})."
                .format(t))
        out.append(CubicBezier(*blended))
    return out


# Subpath pairing ############################################################

def _prepare_subpaths(path, where, arc_tolerance):
    """Validate a path and return (outer_subpath, hole_subpaths) as cubic
    paths, together with their position summaries."""
    if not isinstance(path, Path):
        raise MorphError(
            "Expected a Path object for {}, got {}.".format(
                where, type(path).__name__))
    if len(path) == 0:
        raise MorphError("The {} path is empty.".format(where))
    for seg in path:
        if not is_path_segment(seg):
            raise MorphError(
                "The {} path contains an object that is not a path "
                "segment: {!r}".format(where, seg))
        if not (_finite(seg.start) and _finite(seg.end)):
            raise MorphError(
                "The {} path contains non-finite coordinates.".format(where))

    subs = path.continuous_subpaths()
    for sub in subs:
        if not sub.iscontinuous() or not sub.isclosedac():
            raise MorphError(
                "The {} path contains a subpath that is not closed; "
                "morph_paths() only morphs closed contours.".format(where))

    converted = [_to_cubic_path(sub, arc_tolerance) for sub in subs]
    cubic_subs = [item[0] for item in converted]
    joint_fractions = [item[1] for item in converted]
    areas = [_signed_area(sub) for sub in cubic_subs]

    # The outer contour is the loop with the largest enclosing area; every
    # other loop is a hole (nested contour).
    outer_idx = int(np.argmax([abs(a) for a in areas]))
    info = []
    for idx, sub in enumerate(cubic_subs):
        info.append({
            'path': sub,
            'joints': joint_fractions[idx],
            'area': areas[idx],
            'centroid': _centroid(sub),
            'is_outer': idx == outer_idx,
        })
    outer = [item for item in info if item['is_outer']]
    holes = [item for item in info if not item['is_outer']]
    return outer, holes


def _pairing_cost(item_a, item_b):
    """Position-based cost of pairing two subpaths of the same kind."""
    return abs(item_a['centroid'] - item_b['centroid'])


def _best_pairing(group_a, group_b):
    """Pair two ordered groups of subpaths minimizing total centroid
    distance.  Exhaustive for small groups (typical icons have a handful of
    holes), greedy fallback for pathological inputs."""
    n = len(group_a)
    if n != len(group_b):
        raise MorphError(
            "Incompatible topology: the two paths do not have the same number "
            "of subpaths ({} vs {}).".format(n, len(group_b)))
    if n == 0:
        return []
    if n <= 6:
        best_perm, best_cost = None, np.inf
        for perm in permutations(range(n)):
            cost = sum(_pairing_cost(group_a[i], group_b[perm[i]])
                       for i in range(n))
            if cost < best_cost:
                best_cost, best_perm = cost, perm
        return [(group_a[i], group_b[best_perm[i]]) for i in range(n)]
    # Greedy nearest-neighbor pairing on the larger case.
    remaining = list(range(n))
    pairs = []
    for item_a in group_a:
        j = min(remaining,
                key=lambda idx: _pairing_cost(item_a, group_b[idx]))
        remaining.remove(j)
        pairs.append((item_a, group_b[j]))
    return pairs


def _check_topology(outer_a, holes_a, outer_b, holes_b):
    if len(outer_a) != len(outer_b):
        raise MorphError(
            "Incompatible topology: number of outer contours differs "
            "({} vs {}).".format(len(outer_a), len(outer_b)))
    if len(holes_a) != len(holes_b):
        raise MorphError(
            "Incompatible topology: number of holes differs ({} vs {}).".format(
                len(holes_a), len(holes_b)))


# Public API #################################################################

def morph_paths(path1, path2, t, arc_tolerance=_DEFAULT_ARC_TOLERANCE,
                align_samples=_DEFAULT_ALIGN_SAMPLES):
    """Morph between two closed paths at progress `t`.

    Parameters
    ----------
    path1, path2 : Path
        Closed paths.  Each may contain several continuous closed subpaths;
        the subpath enclosing the largest area is treated as the outer
        contour and the remaining ones as holes.  Both paths must have the
        same number of outer contours and holes.
    t : float
        Morph progress, with ``0 <= t <= 1``.  At ``t=0`` the returned path
        reproduces `path1` and at ``t=1`` it reproduces `path2` (exactly for
        line/bezier input, within `arc_tolerance` when arcs are involved).
    arc_tolerance : float, optional
        Maximum allowed geometric error when approximating Arc segments with
        cubic beziers, in the coordinate units of the input paths.
    align_samples : int, optional
        Number of equally arc-length-spaced samples used to decide the
        cyclic start point and orientation alignment of each loop pair.

    Returns
    -------
    Path
        A closed path of CubicBezier segments.  For multi-subpath input the
        returned path is the concatenation of the morphed subpaths, each
        individually closed, so subpath correspondence and closure are
        preserved.

    Raises
    ------
    MorphError
        If `t` is outside [0, 1], either path is empty/open/non-finite/
        zero-length, or the subpath/hole topologies do not match.
    """
    # Validate the progress first, before doing any geometry work.
    if not isinstance(t, (int, float, np.integer, np.floating)) or \
            not np.isfinite(t):
        raise MorphError(
            "Morph progress t must be a finite number, got {!r}.".format(t))
    t = float(t)
    if not 0.0 <= t <= 1.0:
        raise MorphError(
            "Morph progress t must be between 0 and 1 inclusive, got {}."
            .format(t))
    if arc_tolerance <= 0.0:
        raise MorphError(
            "arc_tolerance must be positive, got {}.".format(arc_tolerance))
    if align_samples < 8:
        raise MorphError(
            "align_samples must be at least 8, got {}.".format(align_samples))

    outer_a, holes_a = _prepare_subpaths(path1, 'first', arc_tolerance)
    outer_b, holes_b = _prepare_subpaths(path2, 'second', arc_tolerance)
    _check_topology(outer_a, holes_a, outer_b, holes_b)

    pairs = (_best_pairing(outer_a, outer_b) +
             _best_pairing(holes_a, holes_b))

    morphed_subpaths = []
    for item_a, item_b in pairs:
        # _align_loops reverses the second loop when its orientation
        # disagrees with the first, and picks cyclic starts so corresponding
        # perimeter points meet (no unnecessary twisting of the contour).
        sub_a, sub_b, joints_a, joints_b = _align_loops(
            item_a['path'], item_b['path'],
            item_a['joints'], item_b['joints'],
            item_a['area'] > 0, item_b['area'] > 0, align_samples)

        fracs_a = _arc_length_breakpoints(sub_a, canonical=joints_a)
        fracs_b = _arc_length_breakpoints(sub_b, canonical=joints_b)
        # The union contains every segment joint of BOTH loops.
        breakpoints = _merge_breakpoints(fracs_a, fracs_b)

        cells_a = _subdivide(sub_a, breakpoints)
        cells_b = _subdivide(sub_b, breakpoints)
        if [len(cell) for cell in cells_a] != \
                [len(cell) for cell in cells_b]:
            raise MorphError(
                "Internal error: sub-piece counts differ after "
                "re-parameterization.")
        morphed_subpaths.append(
            Path(*_interpolate_cubics(cells_a, cells_b, t)))

    result = concatpaths(morphed_subpaths)
    if len(result) == 0:
        raise MorphError("Morph produced an empty path.")
    # Final safety net: no NaN/inf may leave this function.
    for seg in result:
        if not all(_finite(z) for z in seg.bpoints()):
            raise MorphError("Morph produced non-finite coordinates.")
    return result
