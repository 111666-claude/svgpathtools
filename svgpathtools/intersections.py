"""Geometric classification of path intersections.

This module builds on top of the segment-level ``intersect`` routines in
``path.py`` and classifies every contact between two paths into one of four
geometric categories:

- ``CROSSING``   -- the two paths pass through each other; their tangents at
                    the contact point are not parallel.
- ``TANGENT``    -- the paths touch and share a common tangent line (this
                    includes contacts where one or both tangents reverse
                    direction, e.g. cusps).
- ``ENDPOINT``   -- the contact involves an endpoint of one of the (open)
                    paths, so the contact is one-sided by construction.
- ``COINCIDENT`` -- the two paths overlap along an interval of positive
                    length rather than at isolated points.

The main entry points are ``classify_intersections(path1, path2)`` and the
``Path.intersect_classified`` method.  Both return an ``IntersectionReport``.

Notes on scope and determinism
------------------------------
* The classic ``Path.intersect``/``Segment.intersect`` API is untouched; this
  module only adds new functionality.
* Coincidence detection is exact-by-construction: collinear ``Line`` pairs
  (partial overlaps included), straight (collinear) Beziers against lines,
  Beziers that are identical up to degree elevation and/or reversal, and
  ``Arc`` pairs lying on the same ellipse.  Other curve types cannot coincide
  over an interval without being one of these cases.
* Near-tangent contacts are decided by a fixed threshold on the sine of the
  angle between the tangents (``tangency_tol``), so results do not depend on
  segment order or floating point noise below the threshold.
* Zero-length (degenerate) segments never raise; they are treated as points.
* Duplicate contacts reported at joints shared by adjacent segments are
  merged, and the merged result is independent of the order in which segment
  pairs are visited.
"""

from collections import namedtuple

from .path import (Path, Line, QuadraticBezier, CubicBezier, Arc)
from .polytools import real, imag, polyroots01

# Intersection category labels (also available as IntersectionKind.CROSSING
# etc.).
CROSSING = 'crossing'
TANGENT = 'tangent'
ENDPOINT = 'endpoint'
COINCIDENT = 'coincident'


class IntersectionKind(object):
    """Namespace of the four intersection category labels."""

    CROSSING = CROSSING
    TANGENT = TANGENT
    ENDPOINT = ENDPOINT
    COINCIDENT = COINCIDENT

    ALL = (CROSSING, TANGENT, ENDPOINT, COINCIDENT)


class Intersection(namedtuple('Intersection',
                              ['kind', 'point', 'side1', 'side2'])):
    """A single classified point contact between two paths.

    Attributes:
        kind: one of ``CROSSING``, ``TANGENT`` or ``ENDPOINT``.
        point (complex): the contact point.
        side1 (tuple): ``(T1, seg1, t1)`` locating the contact on the first
            path -- the same format used by ``Path.intersect``.
        side2 (tuple): ``(T2, seg2, t2)`` locating the contact on the second
            path.
    """

    __slots__ = ()


class Coincidence(namedtuple('Coincidence',
                             ['kind', 'seg1', 'seg2',
                              't1_range', 't2_range',
                              'T1_range', 'T2_range',
                              'start_point', 'end_point'])):
    """An interval over which two paths coincide.

    Attributes:
        kind: always ``COINCIDENT``.
        seg1, seg2: the coinciding segments of path1 and path2.
        t1_range, t2_range: ``(t_start, t_end)`` parameter intervals on
            ``seg1``/``seg2`` (decreasing if the segments are oppositely
            oriented).
        T1_range, T2_range: the same interval expressed in the whole-path
            parameterizations of path1 and path2.
        start_point, end_point (complex): endpoints of the shared interval
            (ordered along path1).
    """

    __slots__ = ()


class IntersectionReport(object):
    """The classified contacts between two paths.

    Attributes:
        points (list[Intersection]): all isolated point contacts, sorted by
            their position along path1 then path2.
        coincidences (list[Coincidence]): all coincidence intervals, sorted
            by their position along path1 then path2.
    """

    def __init__(self, points, coincidences):
        """Create a report from classified points and coincidences."""
        self.points = list(points)
        self.coincidences = list(coincidences)

    @property
    def crossings(self):
        """Point contacts where the paths pass through each other."""
        return [i for i in self.points if i.kind == CROSSING]

    @property
    def tangencies(self):
        """Point contacts where the paths touch with a common tangent."""
        return [i for i in self.points if i.kind == TANGENT]

    @property
    def endpoint_contacts(self):
        """Point contacts involving an endpoint of either path."""
        return [i for i in self.points if i.kind == ENDPOINT]

    def kinds(self):
        """The set of categories present in this report."""
        return {i.kind for i in self.points} | {c.kind for c in
                                                self.coincidences}

    def legacy(self):
        """The point contacts in the classic ``Path.intersect`` format:
        a list of ``((T1, seg1, t1), (T2, seg2, t2))`` tuples."""
        return [(i.side1, i.side2) for i in self.points]

    def __len__(self):
        return len(self.points) + len(self.coincidences)

    def __iter__(self):
        return iter(self.points + self.coincidences)

    def __repr__(self):
        counts = {k: 0 for k in IntersectionKind.ALL}
        for i in self.points:
            counts[i.kind] += 1
        counts[COINCIDENT] += len(self.coincidences)
        return ("IntersectionReport(crossing={0}, tangent={1}, endpoint={2},"
                " coincident={3})".format(counts[CROSSING], counts[TANGENT],
                                          counts[ENDPOINT],
                                          counts[COINCIDENT]))


# -------------------------------------------------------------------------
# small geometry helpers
# -------------------------------------------------------------------------

def _cross2d(u, v):
    """The z-component of the 3d cross product of planar vectors u, v."""
    return u.real * v.imag - u.imag * v.real


def _seg_scale(seg):
    """A characteristic length of the segment (>= 0)."""
    if isinstance(seg, Arc):
        return max(abs(seg.radius), abs(seg.end - seg.start))
    return max([abs(c - seg.start) for c in seg.bpoints()] +
               [abs(seg.end - seg.start)])


def _is_degenerate(seg, tol):
    """True if the segment has zero length (within tol), i.e. is a point."""
    if isinstance(seg, Arc):
        # The Arc constructor forbids start == end and zero radii.
        return False
    if isinstance(seg, Line):
        return abs(seg.end - seg.start) <= tol
    if isinstance(seg, QuadraticBezier):
        return (abs(seg.control - seg.start) <= tol and
                abs(seg.end - seg.start) <= tol)
    if isinstance(seg, CubicBezier):
        return (abs(seg.control1 - seg.start) <= tol and
                abs(seg.control2 - seg.start) <= tol and
                abs(seg.end - seg.start) <= tol)
    raise TypeError("unknown segment type: %r" % (seg,))


def _point_on_segment_t(seg, p, tol):
    """If p lies on seg (within tol), return its parameter t, else None.

    Works for every segment type, unlike the ``point_to_t`` methods which
    only exist for Line and Arc.
    """
    if _is_degenerate(seg, tol):
        return 0.0 if abs(seg.start - p) <= tol else None
    if isinstance(seg, (Line, Arc)):
        t = seg.point_to_t(p)
        if t is None or not (0 <= t <= 1):
            return None
        return t
    # Bezier segment: minimize |seg.point(t) - p| over t in [0, 1] by
    # solving d/dt |P(t) - p|^2 = 0 and checking the endpoints.
    poly = seg.poly()
    px = real(poly) - p.real
    py = imag(poly) - p.imag
    dist2_deriv = px * px.deriv() + py * py.deriv()
    candidates = [0.0, 1.0]
    candidates += [t for t in polyroots01(dist2_deriv) if 0 <= t <= 1]
    best_t, best_d = None, None
    for t in candidates:
        d = abs(seg.point(t) - p)
        if best_d is None or d < best_d:
            best_t, best_d = t, d
    if best_d is not None and best_d <= tol:
        return best_t
    return None


# -------------------------------------------------------------------------
# coincidence (overlap interval) detection
# -------------------------------------------------------------------------

def _cubic_bpoints(seg):
    """Control points of the segment degree-elevated to a cubic Bezier."""
    if isinstance(seg, Line):
        p0, p1 = seg.start, seg.end
        d = (p1 - p0) / 3.0
        return [p0, p0 + d, p1 - d, p1]
    if isinstance(seg, QuadraticBezier):
        p0, c, p1 = seg.start, seg.control, seg.end
        return [p0, p0 + (2.0 / 3) * (c - p0), p1 + (2.0 / 3) * (c - p1), p1]
    if isinstance(seg, CubicBezier):
        return list(seg.bpoints())
    raise TypeError("not a line/bezier segment: %r" % (seg,))


def _project_t(line, p):
    """The parameter t such that line.point(t) is the orthogonal
    projection of p onto the (infinite) line."""
    d = line.end - line.start
    w = p - line.start
    return (w.real * d.real + w.imag * d.imag) / (abs(d) ** 2)


def _line_line_coincidence(l1, l2, tol):
    """Overlap interval of two Lines, or None if they are not collinear."""
    d1 = l1.end - l1.start
    d2 = l2.end - l2.start
    n1, n2 = abs(d1), abs(d2)
    if n1 <= tol or n2 <= tol:
        return None
    # parallel?
    if abs(_cross2d(d1, d2)) > tol * n1 * n2:
        return None
    # collinear?
    if abs(_cross2d(l2.start - l1.start, d1)) > tol * n1:
        return None
    ta = _project_t(l1, l2.start)
    tb = _project_t(l1, l2.end)
    lo = max(0.0, min(ta, tb))
    hi = min(1.0, max(ta, tb))
    if hi <= lo or (hi - lo) * n1 <= tol:
        return None
    # corresponding parameters on l2
    ua = _project_t(l2, l1.point(lo))
    ub = _project_t(l2, l1.point(hi))
    ua = min(1.0, max(0.0, ua))
    ub = min(1.0, max(0.0, ub))
    return (lo, hi), (ua, ub)


def _line_bezier_coincidence(line, bez, tol):
    """Overlap of a Line with a Bezier that is entirely collinear with it."""
    d = line.end - line.start
    n = abs(d)
    if n <= tol:
        return None
    for bp in bez.bpoints():
        if abs(_cross2d(bp - line.start, d)) > tol * n:
            return None
    # The bezier is a straight segment on the line; find the overlap of
    # the chord spanned by its endpoints with the line segment.
    ta = _project_t(line, bez.start)
    tb = _project_t(line, bez.end)
    lo = max(0.0, min(ta, tb))
    hi = min(1.0, max(ta, tb))
    if hi <= lo or (hi - lo) * n <= tol:
        return None
    # parameters on the bezier of the overlap endpoints
    ua = _point_on_segment_t(bez, line.point(lo), tol)
    ub = _point_on_segment_t(bez, line.point(hi), tol)
    if ua is None or ub is None:
        return None
    return (lo, hi), (ua, ub)


def _bezier_bezier_coincidence(b1, b2, tol):
    """Full-segment coincidence of two line/bezier segments, detected by
    degree-elevating both to cubics and comparing control points (in either
    orientation)."""
    c1 = _cubic_bpoints(b1)
    c2 = _cubic_bpoints(b2)
    scale = max(1.0, max(abs(c) for c in c1 + c2))
    if all(abs(a - b) <= tol * scale for a, b in zip(c1, c2)):
        return (0.0, 1.0), (0.0, 1.0)
    if all(abs(a - b) <= tol * scale for a, b in zip(c1, reversed(c2))):
        return (0.0, 1.0), (1.0, 0.0)
    return None


def _arc_arc_coincidence(a1, a2, tol):
    """Overlap interval of two Arcs lying on the same ellipse."""
    if abs(a1.center - a2.center) > tol:
        return None
    if abs(a1.radius - a2.radius) > tol:
        return None
    # the ellipse rotations must agree modulo 180 degrees
    d_rot = (a1.rotation - a2.rotation) % 180.0
    if min(d_rot, 180.0 - d_rot) > 1e-6:
        return None

    ang_tol = 1e-6  # degrees

    def covered(arc, angle):
        """Is `angle` (degrees) swept by `arc`?"""
        if arc.delta >= 0:
            return (angle - arc.theta) % 360.0 <= arc.delta + ang_tol
        return (arc.theta - angle) % 360.0 <= -arc.delta + ang_tol

    def directed_distance(arc, angle):
        """Signed angular distance from arc.theta to angle along the arc's
        sweep direction (same sign convention as arc.delta)."""
        if arc.delta >= 0:
            return (angle - arc.theta) % 360.0
        return -((arc.theta - angle) % 360.0)

    # candidate boundary angles of the overlap: endpoints of either arc
    # that are covered by the other
    ends1 = [a1.theta, a1.theta + a1.delta]
    ends2 = [a2.theta, a2.theta + a2.delta]
    candidates = [a for a in ends2 if covered(a1, a)]
    candidates += [a for a in ends1 if covered(a2, a)]
    if not candidates:
        return None
    dists = [directed_distance(a1, a) for a in candidates]
    d_lo, d_hi = min(dists), max(dists)
    if d_hi - d_lo <= ang_tol:
        return None  # overlap is a single point, not an interval
    t1a = d_lo / a1.delta
    t1b = d_hi / a1.delta
    if not (0 <= t1a <= 1 and 0 <= t1b <= 1):
        return None
    p_lo = a1.point(t1a)
    p_hi = a1.point(t1b)
    t2a = a2.point_to_t(p_lo)
    t2b = a2.point_to_t(p_hi)
    if t2a is None or t2b is None:
        return None
    return (t1a, t1b), (t2a, t2b)


def _segment_coincidence(seg1, seg2, tol):
    """If seg1 and seg2 coincide over an interval of positive length,
    return ((t1a, t1b), (t2a, t2b)); otherwise return None."""
    if _is_degenerate(seg1, tol) or _is_degenerate(seg2, tol):
        return None
    if isinstance(seg1, Arc) or isinstance(seg2, Arc):
        if isinstance(seg1, Arc) and isinstance(seg2, Arc):
            return _arc_arc_coincidence(seg1, seg2, tol)
        return None  # a straight segment cannot coincide with a true arc
    if isinstance(seg1, Line) and isinstance(seg2, Line):
        return _line_line_coincidence(seg1, seg2, tol)
    if isinstance(seg1, Line):
        r = _line_bezier_coincidence(seg1, seg2, tol)
        return r
    if isinstance(seg2, Line):
        r = _line_bezier_coincidence(seg2, seg1, tol)
        if r is None:
            return None
        (t2a, t2b), (t1a, t1b) = r
        return (t1a, t1b), (t2a, t2b)
    return _bezier_bezier_coincidence(seg1, seg2, tol)


# -------------------------------------------------------------------------
# point intersections between two segments
# -------------------------------------------------------------------------

def _segment_point_intersections(seg1, seg2, tol):
    """Isolated point intersections of two segments as (t1, t2) tuples.

    Unlike ``Segment.intersect`` this never raises on degenerate segments
    and never recurses forever on (near-)coincident ones -- the caller is
    expected to have filtered exact coincidences out already.
    """
    deg1 = _is_degenerate(seg1, tol)
    deg2 = _is_degenerate(seg2, tol)
    if deg1 and deg2:
        if abs(seg1.start - seg2.start) <= tol:
            return [(0.0, 0.0)]
        return []
    if deg1:
        t2 = _point_on_segment_t(seg2, seg1.start, tol)
        return [(0.0, t2)] if t2 is not None else []
    if deg2:
        t1 = _point_on_segment_t(seg1, seg2.start, tol)
        return [(t1, 0.0)] if t1 is not None else []
    try:
        # always solve with a tight tolerance so the classified points match
        # the precision of the classic `intersect` API; `tol` only governs
        # classification (coincidence, clustering, degeneracy)
        return list(seg1.intersect(seg2, tol=min(tol, 1e-12)))
    except RecursionError:
        # near-coincident curved segments can drive bezier_intersections
        # to the recursion limit; treat deterministically as no isolated
        # intersections
        return []


# -------------------------------------------------------------------------
# tangent directions (with cusp handling)
# -------------------------------------------------------------------------

def _seg_directions(seg, t, tol):
    """Unit direction(s) of travel of seg at parameter t.

    Returns one direction for regular points.  At a cusp (vanishing
    derivative) the direction of the lowest non-vanishing higher derivative
    at the cusp is returned instead, so a tangent reversal is still
    recognized as a single tangent line -- even when the intersection
    solver's t is slightly off the exact cusp parameter.
    """
    eps_t = 1e-9
    scale = max(_seg_scale(seg), 1.0)
    try:
        d = seg.derivative(t)
    except Exception:
        d = 0j
    if abs(d) > 1e-4 * scale:
        return [d / abs(d)]
    # (near-)vanishing derivative: if there is an exact stationary point
    # of the curve close to t, the contact is at a cusp; classify using
    # the cusp axis (lowest non-vanishing higher derivative there)
    if isinstance(seg, (QuadraticBezier, CubicBezier)):
        for ts in polyroots01(seg.poly().deriv()):
            if abs(ts - t) > 1e-3:
                continue
            for n in (2, 3, 4):
                try:
                    dn = seg.derivative(ts, n)
                except Exception:
                    break
                if abs(dn) > 1e-9 * scale:
                    return [dn / abs(dn)]
    # fall back to one-sided secants
    dirs = []
    p = seg.point(t)
    h = 1e-6
    if t < 1.0 - eps_t:
        q = seg.point(min(t + h, 1.0))
        v = q - p
        if abs(v) > 0:
            dirs.append(v / abs(v))
    if t > eps_t:
        q = seg.point(max(t - h, 0.0))
        v = p - q
        if abs(v) > 0:
            dirs.append(v / abs(v))
    return dirs


def _path_directions_at(path, occurrences, tol):
    """All unit travel directions of `path` at a contact point.

    `occurrences` is a list of (segment_index, t) pairs at which the point
    is realized.  At joints shared by adjacent segments, the directions of
    both neighboring segments are included.
    """
    eps_t = 1e-9
    dirs = []
    n = len(path)
    for i, t in occurrences:
        seg = path[i]
        if not _is_degenerate(seg, tol):
            dirs += _seg_directions(seg, t, tol)
        # continuity with neighbors across a joint
        if t <= eps_t and i > 0:
            prev = path[i - 1]
            if (abs(prev.end - seg.start) <= tol and
                    not _is_degenerate(prev, tol)):
                dirs += _seg_directions(prev, 1.0, tol)
        if t >= 1.0 - eps_t and i < n - 1:
            nxt = path[i + 1]
            if (abs(nxt.start - seg.end) <= tol and
                    not _is_degenerate(nxt, tol)):
                dirs += _seg_directions(nxt, 0.0, tol)
    return dirs


def _at_path_endpoint(path, T, point, tol):
    """True if the contact at whole-path parameter T is at an endpoint of
    an (open) path."""
    if len(path) == 0:
        return False
    eps_T = 1e-9
    ctol = max(tol, 1e-9) * 10
    if abs(path[0].start - path[-1].end) <= ctol:
        return False  # closed path: no endpoints
    if T <= eps_T and abs(point - path[0].start) <= ctol:
        return True
    if T >= 1.0 - eps_T and abs(point - path[-1].end) <= ctol:
        return True
    return False


def _classify_point(dirs1, dirs2, at_endpoint, tangency_tol):
    """Classify a point contact from the tangent directions of both paths."""
    if at_endpoint or not dirs1 or not dirs2:
        return ENDPOINT
    for u in dirs1:
        for v in dirs2:
            if abs(_cross2d(u, v)) > tangency_tol:
                return CROSSING
    return TANGENT


# -------------------------------------------------------------------------
# main entry point
# -------------------------------------------------------------------------

def classify_intersections(path1, path2, tol=1e-9, tangency_tol=1e-6):
    """Find and geometrically classify all contacts between two paths.

    Args:
        path1, path2: ``Path`` objects (a single segment is also accepted
            and is wrapped in a ``Path``).
        tol (float): geometric tolerance used for coincidence detection,
            point clustering and degenerate-segment detection.
        tangency_tol (float): threshold on |sin(angle)| between the tangent
            directions of the two paths at a contact point; contacts with
            smaller values are classified ``TANGENT``, larger ones
            ``CROSSING``.  This is the near-tangent error threshold.

    Returns:
        IntersectionReport: classified point contacts and coincidence
        intervals.  Deterministic: identical inputs always produce
        identical output, independent of segment visitation order, and
        duplicate contacts at joints shared by adjacent segments are
        merged into one.
    """
    if not isinstance(path1, Path):
        path1 = Path(path1)
    if not isinstance(path2, Path):
        path2 = Path(path2)

    ctol = max(tol, 1e-9) * 10  # clustering tolerance for duplicate points

    # 1. coincidence intervals -------------------------------------------
    coincidences = []
    coincident_pairs = set()  # (index in path1, index in path2)
    for i, seg1 in enumerate(path1):
        for j, seg2 in enumerate(path2):
            co = _segment_coincidence(seg1, seg2, tol)
            if co is None:
                continue
            (t1a, t1b), (t2a, t2b) = co
            coincident_pairs.add((i, j))
            coincidences.append(Coincidence(
                kind=COINCIDENT,
                seg1=seg1, seg2=seg2,
                t1_range=(t1a, t1b), t2_range=(t2a, t2b),
                T1_range=(path1.t2T(seg1, t1a), path1.t2T(seg1, t1b)),
                T2_range=(path2.t2T(seg2, t2a), path2.t2T(seg2, t2b)),
                start_point=seg1.point(t1a), end_point=seg1.point(t1b)))
    coincidences.sort(key=lambda c: (c.T1_range[0], c.T2_range[0]))

    # 2. isolated point intersections ------------------------------------
    raw = []  # (T1, i1, t1, T2, i2, t2, point)
    for i, seg1 in enumerate(path1):
        for j, seg2 in enumerate(path2):
            if (i, j) in coincident_pairs:
                continue
            for t1, t2 in _segment_point_intersections(seg1, seg2, tol):
                T1 = path1.t2T(seg1, t1)
                T2 = path2.t2T(seg2, t2)
                raw.append((T1, i, t1, T2, j, t2, seg1.point(t1)))

    # 3. merge duplicates (e.g. contacts at joints shared by adjacent
    #    segments) via union-find on point proximity
    n = len(raw)
    parent = list(range(n))

    def find(a):
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a, b):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    for a in range(n):
        for b in range(a + 1, n):
            if abs(raw[a][6] - raw[b][6]) <= ctol:
                union(a, b)

    clusters = {}
    for k in range(n):
        clusters.setdefault(find(k), []).append(k)

    # 4. classify each cluster -------------------------------------------
    points = []
    for members in clusters.values():
        members.sort(key=lambda k: (raw[k][0], raw[k][3]))
        first = raw[members[0]]
        point = first[6]

        # skip contacts already covered by a coincidence interval
        idx1 = {raw[k][1] for k in members}
        idx2 = {raw[k][4] for k in members}
        if any(i in idx1 and j in idx2 for i, j in coincident_pairs):
            continue

        T1 = min(raw[k][0] for k in members)
        T2 = min(raw[k][3] for k in members)
        m1 = min(members, key=lambda k: raw[k][0])
        m2 = min(members, key=lambda k: raw[k][3])
        side1 = (raw[m1][0], path1[raw[m1][1]], raw[m1][2])
        side2 = (raw[m2][3], path2[raw[m2][4]], raw[m2][5])

        occ1 = [(raw[k][1], raw[k][2]) for k in members]
        occ2 = [(raw[k][4], raw[k][5]) for k in members]
        dirs1 = _path_directions_at(path1, occ1, tol)
        dirs2 = _path_directions_at(path2, occ2, tol)
        at_endpoint = (_at_path_endpoint(path1, T1, point, tol) or
                       _at_path_endpoint(path2, T2, point, tol))
        kind = _classify_point(dirs1, dirs2, at_endpoint, tangency_tol)
        points.append(Intersection(kind=kind, point=point,
                                   side1=side1, side2=side2))

    points.sort(key=lambda i: (i.side1[0], i.side2[0],
                               i.point.real, i.point.imag))
    return IntersectionReport(points, coincidences)
