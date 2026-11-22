"""Minimum-area rotated bounding boxes for paths and path segments.

The axis-aligned :meth:`Path.bbox` can waste a great deal of space on a
slender, tilted shape (e.g. a diagonal badge): a rotated rectangle whose
sides follow the shape's principal directions may have a much smaller
area.  This submodule computes that minimum-area rectangle for any
:class:`~svgpathtools.path.Path`, :class:`~svgpathtools.path.Line`,
:class:`~svgpathtools.path.QuadraticBezier`,
:class:`~svgpathtools.path.CubicBezier` or :class:`~svgpathtools.path.Arc`.

Algorithm
---------

Every curve is first *flattened* into a polyline whose maximum
deviation from the true curve never exceeds the geometric tolerance
``tol``: elliptical arcs use an analytic, curvature-based piece count,
while Bezier segments size their uniform subdivision from the measured
chord sagitta.  The convex hull of the resulting vertices is then built
with the monotone-chain algorithm.

For a convex polygon, Freeman & Shapira (1975) show that a minimum-area
enclosing rectangle has a side parallel to one of the hull's edges, so
only the edge directions need to be examined.  A rotating-calipers
sweep scores every edge direction in linear time; the best handful of
directions are then re-evaluated on the *exact* analytic segment
extrema so the returned box contains every point of every curve,
including curved portions that bulge beyond their control points.
"""

from __future__ import annotations

from math import atan2, cos, pi, radians, sin, sqrt

import numpy as np

from .path import Arc, Line, Path, is_path_segment, rotate

# Default geometric tolerance (in the user's coordinate units).  The
# flattening error never exceeds this value.
DEFAULT_TOL = 1e-8

# Relative area slack used to recognize "equal area" orientations.  Two
# directions whose exact areas differ by no more than this fraction are
# considered tied, which makes the near-square (and exactly square)
# orientation choice deterministic.
AREA_TIE_REL_TOL = 1e-8

# Upper bound on how many tied directions receive the (more expensive)
# exact analytic re-evaluation.  A flattened circle can present tens of
# thousands of numerically identical hull edges; the smallest-angle
# members of that tie band carry the same optimum, so a fixed budget
# keeps the cost bounded without changing the result.
_MAX_EXACT_CONTENDERS = 64

# Angular tolerance (radians) at which two candidate directions are
# treated as the same direction.
_ANGLE_EPS = 1e-12

# Safety cap on the number of flattening pieces per segment.
_MAX_SUBDIV_DEPTH = 64


class RotatedBoundingBox:
    """A minimum-area axis-free enclosing rectangle of a path.

    Attributes
    ----------
    center : complex
        The center of the rectangle.
    theta : float
        Direction angle in radians of the ``width`` side, measured
        counter-clockwise from the positive real (x) axis.  Angles are
        unoriented, so the value lies in ``0 <= theta < pi``; the
        width side is always the longer side (``height <= width``).
    width, height : float
        Non-negative side lengths measured along ``theta`` and the
        perpendicular direction ``theta + pi/2`` respectively, with
        ``height <= width``.
    corners : tuple of complex
        The four corners, in counter-clockwise order starting at the
        corner with minimum coordinate along ``theta`` and, among those
        (the height-zero case), minimum coordinate along the normal.
    tol : float
        Geometric tolerance used when computing the box.
    """

    def __init__(self, center, theta, width, height, corners, tol):
        """Initialize a rotated bounding box from its geometric data."""
        self.center = center
        self.theta = theta
        self.width = width
        self.height = height
        self.corners = tuple(corners)
        self.tol = tol

    @property
    def direction(self):
        """Unit complex number ``exp(i*theta)`` along the width side."""
        return cos(self.theta) + 1j * sin(self.theta)

    @property
    def normal(self):
        """Unit normal to the width side (counter-clockwise)."""
        u = self.direction
        return -1j * u

    @property
    def area(self):
        """Area of the rectangle (zero for point/line degenerate boxes)."""
        return self.width * self.height

    @property
    def half_width(self):
        """Half of :attr:`width`."""
        return self.width / 2.0

    @property
    def half_height(self):
        """Half of :attr:`height`."""
        return self.height / 2.0

    def contains(self, point, eps=None):
        """Return ``True`` if ``point`` lies in or on the rectangle.

        Parameters
        ----------
        point : complex
            The point to test.
        eps : float, optional
            Non-negative inclusive slack added on all four sides.  It
            defaults to :attr:`tol`, matching the geometric tolerance
            the box was built with.
        """
        if eps is None:
            eps = self.tol
        delta = point - self.center
        u = self.direction
        n = self.normal
        return (abs((delta * np.conj(u)).real) <= self.half_width + eps and
                abs((delta * np.conj(n)).real) <= self.half_height + eps)

    def to_path(self):
        """Return the rectangle as a closed four-segment :class:`Path`.

        The path consists of four :class:`Line` segments.
        """
        c0, c1, c2, c3 = self.corners
        return Path(Line(c0, c1), Line(c1, c2), Line(c2, c3),
                    Line(c3, c0))

    def __repr__(self):
        """Return the unambiguous string representation of the box."""
        return ("RotatedBoundingBox(center={}, theta={:.10g}, "
                "width={:.10g}, height={:.10g})").format(
                    self.center, self.theta, self.width, self.height)

    def __eq__(self, other):
        """Compare two boxes by all their geometric attributes."""
        if not isinstance(other, RotatedBoundingBox):
            return NotImplemented
        return (self.center == other.center and
                self.theta == other.theta and
                self.width == other.width and
                self.height == other.height and
                self.corners == other.corners)

    def __ne__(self, other):
        """Negation of :meth:`__eq__`."""
        result = self.__eq__(other)
        if result is NotImplemented:
            return result
        return not result


# Flattening ##################################################################

def _max_sagitta(seg):
    """Measure the maximum interior chord sagitta of a segment.

    The distance from the curve at ``t`` in (0.25, 0.5, 0.75) to the
    chord joining the segment's endpoints is returned.  A single
    midpoint probe would miss S-shaped curves (whose midpoint lies on
    the chord while both halves bulge away), so three interior probes
    are used.  For a Line the value is exactly zero.  Uniform
    subdivision quarters the sagitta of every subpiece at each halving
    level for both Bezier and Arc segments.
    """
    start, end = seg.start, seg.end
    chord = end - start
    chord_len2 = chord.real ** 2 + chord.imag ** 2
    if chord_len2 == 0.0:
        # Degenerate (zero-length) segment: measure against the point.
        return max(abs(seg.point(t) - start)
                   for t in (0.25, 0.5, 0.75))
    out = 0.0
    for t in (0.25, 0.5, 0.75):
        # Scalar cross product |(p-start) x chord| / |chord|.
        delta = seg.point(t) - start
        cross = abs(delta.real * chord.imag - delta.imag * chord.real)
        out = max(out, cross / sqrt(chord_len2))
    return out


def _uniform_flatten(seg, n):
    """Sample ``seg`` uniformly at ``n + 1`` parameters.

    The parameters are evaluated in one vectorized call.
    """
    ts = np.linspace(0.0, 1.0, n + 1)
    if isinstance(seg, Arc):
        # Vectorized form of Arc.point: the scalar Arc.point() is far
        # too slow for the tens of thousands of samples tight tolerances
        # require on long arcs.
        angle = np.radians(seg.theta + ts * seg.delta)
        cosphi = seg.rot_matrix.real
        sinphi = seg.rot_matrix.imag
        rx, ry = seg.radius.real, seg.radius.imag
        x = rx * cosphi * np.cos(angle) - ry * sinphi * np.sin(angle) \
            + seg.center.real
        y = rx * sinphi * np.cos(angle) + ry * cosphi * np.sin(angle) \
            + seg.center.imag
        return list(x + 1j * y)
    if hasattr(seg, "points"):
        return list(seg.points(ts))
    return [seg.point(t) for t in ts]


def _arc_piece_count(arc, tol):
    """Return the polyline piece count that flattens an Arc to ``tol``.

    The smallest number of uniform polyline pieces keeping every point
    of the elliptical :class:`Arc` within ``tol`` of the polyline is
    returned.  The worst deviation of a chord of a circular arc of
    radius ``r`` spanning angle ``dalpha`` is
    ``r*(1 - cos(dalpha/2))``.  For an ellipse the curvature radius is
    bounded by the larger semi-axis, which gives the conservative bound
    used here.
    """
    rx, ry = arc.radius.real, arc.radius.imag
    r = max(rx, ry)
    span = abs(radians(arc.delta))
    if span == 0.0 or r == 0.0:
        return 1
    # Solve r*(1-cos(span/(2n))) <= tol for n.
    target = min(1.0, tol / r)
    half_step = np.arccos(1.0 - target)
    n = int(np.ceil(span / (2.0 * half_step)))
    return max(1, min(n, 2 ** _MAX_SUBDIV_DEPTH))


def _flatten_segment(seg, tol):
    """Flatten one segment to a polyline within deviation ``tol``.

    The vertices of a polyline approximating ``seg`` with deviation at
    most ``tol`` from the true curve are returned.  Arcs use an
    analytic, curvature-based piece count.  Bezier segments use the
    measured interior chord sagitta ``s``: with ``n`` uniform pieces the
    per-piece deviation shrinks as ``K*s/n**2`` where the measured
    worst-case value of ``K`` is about 11 for quadratic and cubic
    Beziers (including S-shapes whose midpoint lies on the chord); the
    conservative constant 64 below leaves a wide margin for round-off
    and the direct (non-power-of-two) piece count.  All vertices are
    produced in one vectorized evaluation.
    """
    if isinstance(seg, Line):
        return [seg.start, seg.end]
    if isinstance(seg, Arc):
        n = _arc_piece_count(seg, tol)
        vertices = _uniform_flatten(seg, n)
        vertices[0] = seg.start
        vertices[-1] = seg.end
        return vertices
    sagitta = _max_sagitta(seg)
    if sagitta <= tol:
        return [seg.start, seg.end]
    n = int(np.ceil(np.sqrt(64.0 * sagitta / tol)))
    n = max(1, min(n, 2 ** _MAX_SUBDIV_DEPTH))
    vertices = _uniform_flatten(seg, n)
    vertices[0] = seg.start
    vertices[-1] = seg.end
    return vertices


def flatten_path(path, tol=DEFAULT_TOL):
    """Approximate a path by a tolerance-bounded polyline.

    The returned vertices are guaranteed to lie within distance ``tol``
    of every curve.

    Parameters
    ----------
    path : Path or path segment
        The curve(s) to flatten.  A bare segment is treated as a
        one-segment path.
    tol : float, optional
        Maximum allowed deviation, in coordinate units.  Must be
        non-negative.

    Returns
    -------
    list of complex
        The polyline vertices, with consecutive duplicates removed.
    """
    if tol < 0:
        raise ValueError("tol must be non-negative, got {}".format(tol))
    segments = [path] if is_path_segment(path) else list(path)
    vertices = []
    for seg in segments:
        if tol == 0.0 and not isinstance(seg, Line):
            raise ValueError(
                "tol=0 cannot flatten a curved segment; a polyline "
                "cannot represent a Bezier or Arc with zero error.")
        pts = _flatten_segment(seg, tol)
        if vertices:
            pts = pts[1:]
        vertices.extend(pts)
    # Drop consecutive duplicates (zero-length or point segments).
    deduped = []
    for z in vertices:
        if not deduped or abs(z - deduped[-1]) > 0:
            deduped.append(z)
    return deduped


# Convex hull #################################################################

def convex_hull(points):
    """Return the counter-clockwise convex hull of ``points``.

    The monotone-chain (Andrew's) algorithm is used.  Collinear boundary
    points are reduced to the segment endpoints.  The implementation is
    vectorized with NumPy so it stays fast on the tens of thousands of
    vertices produced by tight flattening tolerances.
    """
    pts = np.array(sorted(set(points), key=lambda z: (z.real, z.imag)))
    if pts.size <= 1:
        return list(pts)
    xs, ys = pts.real, pts.imag

    def build(indices):
        hull = []
        for i in indices:
            while len(hull) >= 2:
                o, a = hull[-2], hull[-1]
                cross = ((xs[a] - xs[o]) * (ys[i] - ys[o]) -
                         (ys[a] - ys[o]) * (xs[i] - xs[o]))
                if cross <= 0:
                    hull.pop()
                else:
                    break
            hull.append(i)
        return hull

    order = np.arange(len(pts))
    lower = build(order)
    upper = build(order[::-1])
    hull_idx = lower[:-1] + upper[:-1]
    return list(pts[hull_idx])


# Angle bookkeeping ###########################################################

def _canonical_theta(theta):
    """Map any angle to the unoriented range ``[0, pi)``."""
    return theta % pi


def _raw_hull_edge_angles(hull):
    """Return the monotonically unwrapped edge directions of a hull.

    The direction of edge ``i`` (from ``hull[i]`` to
    ``hull[(i+1) % n]``) is returned for a counter-clockwise convex
    hull.  The angles are unwrapped into a strictly increasing sequence
    whose total turn is ``pi``: a CCW convex polygon's edge directions
    rotate monotonically, and unoriented directions identify ``theta``
    and ``theta + pi``.  Near-duplicate consecutive directions
    (parallel hull edges, e.g. on an axis-aligned polygon) are nudged
    apart by :data:`_ANGLE_EPS` so the sequence stays strictly
    increasing, which is the ordering the rotating-calipers sweep
    requires.
    """
    n = len(hull)
    angles = []
    for i in range(n):
        p, q = hull[i], hull[(i + 1) % n]
        angles.append(atan2(q.imag - p.imag, q.real - p.real))
    unwrapped = [angles[0] % pi]
    for a in angles[1:]:
        a = a % pi
        while a <= unwrapped[-1]:
            a += pi
        if a - unwrapped[-1] < _ANGLE_EPS:
            a = unwrapped[-1] + _ANGLE_EPS
        unwrapped.append(a)
    return unwrapped


# Extents #####################################################################

def _all_approx_boxes(hull, edge_angles):
    """Width, height and area of the box at every hull-edge direction.

    The hull vertices are counter-clockwise, and each entry of
    ``edge_angles`` gives the direction of the corresponding hull edge
    in ``[0, pi)``.  As the edges are walked in hull order their
    directions rotate monotonically (by a total of ``pi``); the
    supporting vertices for the edge direction and its perpendicular
    likewise advance monotonically around the hull (rotating-calipers
    property), so the four support points together make at most
    ``O(hull)`` steps and every box is found in linear time.

    Returns
    -------
    widths, heights, areas : numpy.ndarray
        Arrays indexed like ``edge_angles``.
    """
    h = len(hull)
    xs = np.array([p.real for p in hull])
    ys = np.array([p.imag for p in hull])

    def dot(i, ux, uy):
        return xs[i] * ux + ys[i] * uy

    def move(pointer, ux, uy, maximize):
        value = dot(pointer, ux, uy)
        while True:
            nxt = (pointer + 1) % h
            nxt_value = dot(nxt, ux, uy)
            if (maximize and nxt_value >= value) or \
                    (not maximize and nxt_value <= value):
                pointer, value = nxt, nxt_value
            else:
                return pointer

    widths = np.empty(h)
    heights = np.empty(h)
    # Initialize all four supports for the first edge direction.
    theta0 = edge_angles[0]
    ux0, uy0 = cos(theta0), sin(theta0)
    vx0, vy0 = -uy0, ux0
    max_u = int(np.argmax(xs * ux0 + ys * uy0))
    min_u = int(np.argmin(xs * ux0 + ys * uy0))
    max_v = int(np.argmax(xs * vx0 + ys * vy0))
    min_v = int(np.argmin(xs * vx0 + ys * vy0))

    for i in range(h):
        theta = edge_angles[i]
        ux, uy = cos(theta), sin(theta)
        vx, vy = -uy, ux
        max_u = move(max_u, ux, uy, True)
        min_u = move(min_u, ux, uy, False)
        max_v = move(max_v, vx, vy, True)
        min_v = move(min_v, vx, vy, False)
        widths[i] = dot(max_u, ux, uy) - dot(min_u, ux, uy)
        heights[i] = dot(max_v, vx, vy) - dot(min_v, vx, vy)
    return widths, heights, widths * heights


def _exact_extents(path, theta):
    """Return the exact analytic extents of a curve at orientation.

    The extents of the true curve along the two box axes at orientation
    ``theta`` are returned as ``(xmin, xmax, ymin, ymax)``.  Rotating a
    curve by ``-theta`` about the coordinate origin makes its box axes
    axis-aligned, so each segment's own exact :meth:`bbox`
    (root-finding for Bezier derivatives, trigonometric extrema for
    Arc) gives the extrema with no sampling error.  The pivot point is
    irrelevant to axis extents, and using the origin keeps the inverse
    frame change a pure multiplication by ``exp(i*theta)``.
    """
    rotated = rotate(path, -theta * 180.0 / pi, origin=0j)
    xmin, xmax, ymin, ymax = rotated.bbox()
    return xmin, xmax, ymin, ymax


def _exact_box(path, theta):
    """Exact width, height and area at orientation ``theta``."""
    xmin, xmax, ymin, ymax = _exact_extents(path, theta)
    width = xmax - xmin
    height = ymax - ymin
    return width, height, width * height


# Degenerate handling #########################################################

def _point_box(point, tol):
    """Deterministic zero-size box around a single point."""
    corners = (point, point, point, point)
    return RotatedBoundingBox(center=point, theta=0.0, width=0.0,
                              height=0.0, corners=corners, tol=tol)


def _collinear_box(points, tol):
    """Deterministic zero-height box along the line through ``points``.

    The caller guarantees the convex hull has exactly two vertices,
    i.e. the two extreme points of the line.
    """
    start, end = points[0], points[-1]
    # Unoriented direction in [0, pi); independent of endpoint order.
    theta = _canonical_theta(atan2(end.imag - start.imag,
                                   end.real - start.real))
    u = cos(theta) + 1j * sin(theta)
    center = (start + end) / 2.0
    half_len = abs(end - start) / 2.0
    c0 = center - half_len * u
    c1 = center + half_len * u
    corners = (c0, c1, c1, c0)
    return RotatedBoundingBox(center=center, theta=theta,
                              width=2.0 * half_len, height=0.0,
                              corners=corners, tol=tol)


# Main entry point ############################################################

def min_area_rotated_bbox(path, tol=DEFAULT_TOL):
    """Compute the minimum-area rotated bounding box of a path.

    The box encloses every point of every segment of ``path`` (not just
    its control points), including :class:`Arc` and Bezier bulges.  Its
    area is near-optimal within the geometric tolerance ``tol``: the
    flattening approximation error never exceeds ``tol``, while the
    returned extents themselves are computed from exact analytic
    segment bounds.

    Parameters
    ----------
    path : Path, Line, QuadraticBezier, CubicBezier or Arc
        The curve to enclose.  A bare segment is treated as a
        one-segment path.
    tol : float, optional
        Geometric tolerance in coordinate units (default
        :data:`DEFAULT_TOL`).  Must be non-negative.  Smaller values
        examine more candidate directions and take longer.

    Returns
    -------
    RotatedBoundingBox
        The enclosing rectangle, with deterministic semantics for
        degenerate and tied inputs:

        * an empty :class:`Path` raises :class:`ValueError`;
        * a single-point path yields a zero-width, zero-height box
          centered on the point with ``theta = 0``;
        * a collinear path yields a zero-height box aligned with the
          line, whose direction is given in ``0 <= theta < pi``;
        * when several directions have (numerically) equal area, as for
          a square, the smallest angle in ``[0, pi)`` is chosen, giving
          a stable, translation- and rotation-consistent answer;
        * self-intersecting paths need no special treatment: only their
          convex hull matters.

    Raises
    ------
    ValueError
        If ``path`` contains no geometry, or ``tol`` is negative.
    TypeError
        If ``path`` is not a path or path segment.
    """
    if is_path_segment(path):
        path = Path(path)
    elif not isinstance(path, Path):
        raise TypeError(
            "path must be a Path, Line, QuadraticBezier, CubicBezier "
            "or Arc object, got {}".format(type(path).__name__))
    if len(path) == 0:
        raise ValueError("Cannot compute a rotated bounding box of an "
                         "empty path.")

    vertices = flatten_path(path, tol)
    if not vertices:
        raise ValueError("Cannot compute a rotated bounding box of an "
                         "empty path.")

    # Degenerate geometry: one distinct point.  Complex numbers sort
    # exactly by (real, imag), so a constant path collapses to a single
    # value without a quadratic scan.
    if np.all(np.asarray(vertices) == vertices[0]):
        return _point_box(vertices[0], tol)

    hull = convex_hull(vertices)

    # Degenerate geometry: all points on one line.
    if len(hull) == 2:
        return _collinear_box(hull, tol)

    # Candidate orientations from hull edges (Freeman-Shapira theorem:
    # a minimum-area enclosing rectangle of a convex polygon has a side
    # parallel to one of its edges).  A rotating-calipers sweep scores
    # every edge direction in linear time, which makes pre-simplifying
    # the dense hull of a flattened curve unnecessary.
    edge_angles = _raw_hull_edge_angles(hull)
    _w, _h, approx_areas = _all_approx_boxes(hull, edge_angles)
    order = np.argsort(approx_areas, kind="stable")
    best_approx_area = float(approx_areas[order[0]])
    # Everything within the relative tie band competes on an equal
    # footing: a flattened circle or near-square presents many hull
    # edges whose sampled areas differ only by flattening noise, and the
    # smallest-angle direction must win those ties deterministically.
    tie_cutoff = best_approx_area * (1.0 + AREA_TIE_REL_TOL) + tol * tol
    tied = sorted({_canonical_theta(edge_angles[i])
                   for i in order
                   if float(approx_areas[i]) <= tie_cutoff})
    # The axis-aligned direction is always offered as a candidate.  It
    # only wins when its area lies in the optimum tie band (e.g. a
    # circle or square), where it gives the deterministic theta = 0
    # answer that a discretized hull with no horizontal edge could not.
    axis_area = _exact_box(path, 0.0)[2]
    if axis_area <= tie_cutoff:
        tied = sorted(set(tied) | {0.0})
    # Keep the smallest angles; the cap bounds exact-evaluation cost on
    # shapes with huge exactly-tied families (e.g. a flattened circle).
    contenders = tied[:_MAX_EXACT_CONTENDERS]

    # Re-evaluate the contending directions against exact analytic
    # extents.  Each contender is an unoriented angle in [0, pi).  The
    # deterministic ordering prefers (1) area within the relative tie
    # band of the best exact area, then (2) a frame that already
    # satisfies the width >= height convention, then (3) the smallest
    # angle.  Grouping numerically equal areas together (rule 1) is what
    # makes a circle or square report theta = 0 rather than an
    # arbitrarily rotated near-tied direction.
    exact = [_exact_box(path, theta) + (theta,)
             for theta in contenders]
    best_exact_area = min(item[2] for item in exact)
    exact_tie = best_exact_area * (1.0 + AREA_TIE_REL_TOL) + tol * tol

    def ranking(item):
        width, height, area, theta = item
        area_band = 0 if area <= exact_tie else 1
        return (area_band, 0 if width >= height else 1, theta)

    _width, _height, _area, theta = sorted(exact, key=ranking)[0]

    # Rebuild the final geometry from exact extents.  theta is the
    # unoriented box direction in [0, pi); enforce width >= height by
    # rotating the frame 90 degrees when needed.
    xmin, xmax, ymin, ymax = _exact_extents(path, theta)
    width, height = xmax - xmin, ymax - ymin
    if height > width:
        theta = (theta + pi / 2.0) % pi
        xmin, xmax, ymin, ymax = _exact_extents(path, theta)
        width, height = xmax - xmin, ymax - ymin
    # Undo the -theta rotation used to make the box axis-aligned.
    center_rotated = complex((xmin + xmax) / 2.0,
                             (ymin + ymax) / 2.0)
    u = cos(theta) + 1j * sin(theta)
    center = u * center_rotated
    n = -1j * u
    c = center
    hw, hh = width / 2.0, height / 2.0
    # Counter-clockwise, starting at the minimum-u corner.
    c0 = c - hw * u - hh * n
    c1 = c + hw * u - hh * n
    c2 = c + hw * u + hh * n
    c3 = c - hw * u + hh * n
    return RotatedBoundingBox(center=center, theta=theta, width=width,
                              height=height, corners=(c0, c1, c2, c3),
                              tol=tol)
