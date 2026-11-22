"""Grid coverage queries for path center lines.

This module implements :func:`grid_coverage`, which returns the grid cells
actually intersected by the *center line* of a :class:`~svgpathtools.path.Path`
(or of an individual path segment).  Unlike the cells intersected by the
path's overall bounding box -- which can contain large numbers of empty
cells for long curving features such as rivers -- every reported cell is
guaranteed to contain at least one point of the stroke center line.

The algorithm is exact rather than a rasterization: for each segment the
parameter values where the curve crosses a vertical or horizontal grid line
are found analytically (linear solving for :class:`Line`, polynomial roots
for the Bezier segments, trigonometric roots for :class:`Arc`), and the
curve is walked through those crossing events, assigning every interval to
its containing cell via its midpoint.  Its cost is proportional to the
number of *occupied* cells, never to the number of cells covered by the
bounding box.

Boundary convention
-------------------
Cells are half-open: cell ``(i, j)`` covers

    ``origin_x + i*cell_w <= x < origin_x + (i+1)*cell_w``
    ``origin_y + j*cell_h <= y < origin_y + (j+1)*cell_h``

so a curve running along a grid line is assigned to the cell on its
lower/left side.  This gives tangencies, corners, points lying exactly on
grid lines and negative cell coordinates one deterministic owner.  The
``tol`` argument snaps points that land within tolerance of a grid line
onto that line (to neutralize floating-point noise) before the cell index
is computed.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import floor, pi, radians

import numpy as np

from .path import Path, Line, QuadraticBezier, CubicBezier, Arc, is_path_segment

__all__ = ['GridCell', 'GridCoverageLimit', 'grid_coverage']


class GridCoverageLimit(Exception):
    """Raised when a grid coverage query would produce more than the
    configured maximum number of cells.

    The limit exists to make queries on degenerate inputs (zero-sized cells
    or extremely long curves) fail deterministically instead of exhausting
    available memory.  Raise ``max_cells`` (and, usually, fix the cell size)
    if the query was legitimate.
    """


class GridCell(tuple):
    """A ``(i, j, segments)`` triple of two integer grid coordinates and the
    indices of the path segments whose center line intersects the cell.

    The coordinates are available through the :attr:`i` and :attr:`j`
    properties, and a cell compares equal to the plain coordinate tuple
    ``(i, j)`` so it can be used directly as a set/dict key of coordinates.
    The segment indices are available through :attr:`segments` or ``cell[2]``.

    Attributes:
        i (int): column index, measured from ``origin_x``.
        j (int): row index, measured from ``origin_y``.
        segments (tuple[int, ...]): sorted tuple of segment indices.
    """

    __slots__ = ()

    def __new__(cls, i, j, segments):
        return super().__new__(
            cls, (int(i), int(j), tuple(sorted(segments))))

    def __repr__(self):
        return 'GridCell(i={}, j={}, segments={})'.format(
            self[0], self[1], self[2])

    @property
    def i(self):
        return self[0]

    @property
    def j(self):
        return self[1]

    @property
    def segments(self):
        return self[2]

    def __eq__(self, other):
        if isinstance(other, GridCell):
            return tuple.__eq__(self, other)
        if isinstance(other, tuple) and len(other) == 2:
            # compare against a plain (i, j) coordinate
            return (self[0], self[1]) == other
        return tuple.__eq__(self, other)

    def __hash__(self):
        return hash((self[0], self[1]))


# Numerical guards ############################################################

# Parameter values closer than this are treated as the same crossing event.
_T_EPS = 1e-10

# Root solvers (np.roots on a multiple root, trigonometric solves) can return
# a boundary root displaced by a larger numerical error than _T_EPS, so
# accept roots drifting past an endpoint up to this looser bound.
_ROOT_ENDPOINT_EPS = 1e-7


@dataclass(frozen=True)
class _Grid:
    """The immutable grid definition for one query, threaded explicitly
    through the per-segment solvers (no module-global mutable state)."""
    origin_x: float
    origin_y: float
    cell_w: float
    cell_h: float
    tol_x: float
    tol_y: float

    def candidate_indices(self, lo, hi, vertical):
        """Integer grid-line offsets whose line is crossed by a coordinate
        spanning ``[lo, hi]``.

        Covers every strictly interior grid line and, when the upper bound
        lies (numerically) on a grid line, that boundary line too -- a curve
        ending on a grid line must place its endpoint in the cell on the high
        side of the half-open convention.  A lower-boundary line is excluded,
        since it can only reproduce the start cell rather than mark a crossing.
        """
        if hi <= lo:
            return range(0, 0)
        origin = self.origin_x if vertical else self.origin_y
        size = self.cell_w if vertical else self.cell_h
        k_lo = floor((lo - origin) / size) + 1
        k_hi = floor((hi - origin) / size)
        if k_hi < k_lo:
            return range(0, 0)
        return range(k_lo, k_hi + 1)

    def line_value(self, k, vertical):
        if vertical:
            return self.origin_x + k * self.cell_w
        return self.origin_y + k * self.cell_h

    def cell_of(self, point):
        """Half-open cell index of a point, with one-directional tolerance
        snapping: only points whose evaluation noise pushed them just *below*
        their cell's upper grid line are snapped up onto it.  Points on the
        low side and points exactly on a line are left to the floor rule, so
        every boundary has a single deterministic owner."""
        x, y = point.real, point.imag
        i0 = floor((x - self.origin_x) / self.cell_w)
        j0 = floor((y - self.origin_y) / self.cell_h)
        upper_x = self.origin_x + (i0 + 1) * self.cell_w
        upper_y = self.origin_y + (j0 + 1) * self.cell_h
        if 0.0 <= upper_x - x <= self.tol_x:
            x = upper_x
        if 0.0 <= upper_y - y <= self.tol_y:
            y = upper_y
        return (int(floor((x - self.origin_x) / self.cell_w)),
                int(floor((y - self.origin_y) / self.cell_h)))


def _accept_root(t):
    """Normalize a crossing parameter found by a root solver.

    Returns the parameter of a strictly interior crossing, or ``None`` when
    the root lies exactly on a segment endpoint.  Two near-endpoint cases are
    distinguished:

    * A root numerically equal to 0 or 1 (within ``_T_EPS``) is an endpoint
      contact.  It is not an event because the endpoint's cell is assigned
      directly in :func:`_segment_cells`.
    * A root displaced slightly past an endpoint but outside ``_T_EPS`` (a
      multiple root from :func:`numpy.roots` can drift by ~1e-8) is a genuine
      final crossing into the endpoint's neighborhood.  It is pulled back to a
      point just inside the parameter range so the terminal interval keeps a
      midpoint.
    """
    if t < -_ROOT_ENDPOINT_EPS or t > 1.0 + _ROOT_ENDPOINT_EPS:
        return None
    if -_T_EPS <= t <= _T_EPS or 1.0 - _T_EPS <= t <= 1.0 + _T_EPS:
        return None
    if t < 0.0:
        return _T_EPS
    if t > 1.0:
        return 1.0 - _T_EPS
    return float(t)


def _merge_events(events):
    """Sort crossing parameters and collapse solver noise (duplicated roots
    of transverse crossings and tangent contacts) onto a single event value.
    Tangent cells are not lost by this collapse because :func:`_segment_cells`
    registers every event point's cell directly, not only interval midpoints.
    """
    events = sorted(t for t in events if 0.0 <= t <= 1.0)
    out = []
    for t in events:
        if not out or abs(t - out[-1]) > _T_EPS:
            out.append(t)
    return out


# Per-segment crossing roots ##################################################

def _line_crossings(seg, grid):
    """Yield ``(t, axis)`` events where a Line meets a grid line.  ``axis`` is
    0 for a vertical line (an x crossing), 1 for horizontal (a y crossing)."""
    x0, y0 = seg.start.real, seg.start.imag
    x1, y1 = seg.end.real, seg.end.imag
    dx, dy = x1 - x0, y1 - y0
    if dx != 0.0:
        for k in grid.candidate_indices(min(x0, x1), max(x0, x1), True):
            t = _accept_root((grid.line_value(k, True) - x0) / dx)
            if t is not None:
                yield t, 0
    if dy != 0.0:
        for k in grid.candidate_indices(min(y0, y1), max(y0, y1), False):
            t = _accept_root((grid.line_value(k, False) - y0) / dy)
            if t is not None:
                yield t, 1


def _bernstein_poly1d(control_values):
    """np.poly1d for one coordinate of a 1st/2nd/3rd degree Bezier curve,
    given that coordinate's control values in Bezier order."""
    n = len(control_values) - 1
    p0, p1 = control_values[0], control_values[-1]
    if n == 1:
        return np.poly1d([p1 - p0, p0])
    if n == 2:
        c = control_values[1]
        return np.poly1d([p0 - 2 * c + p1, 2 * (c - p0), p0])
    if n == 3:
        c1, c2 = control_values[1], control_values[2]
        return np.poly1d([
            -p0 + 3 * (c1 - c2) + p1,
            3 * (p0 - 2 * c1 + c2),
            3 * (c1 - p0),
            p0])
    raise ValueError("unsupported Bezier degree: {}".format(n))


def _bezier_crossings(seg, grid):
    """Yield ``(t, axis)`` interior grid-line crossings of a Bezier segment
    by solving its coordinate polynomials."""
    xmin, xmax, ymin, ymax = seg.bbox()
    bpoints = seg.bpoints()
    polys = (
        (0, _bernstein_poly1d([p.real for p in bpoints]), True, xmin, xmax),
        (1, _bernstein_poly1d([p.imag for p in bpoints]), False, ymin, ymax),
    )
    for axis, poly, vertical, lo, hi in polys:
        for k in grid.candidate_indices(lo, hi, vertical):
            roots = np.roots((poly - grid.line_value(k, vertical)).coeffs)
            for r in roots:
                if abs(r.imag) <= 1e-10:
                    t = _accept_root(float(r.real))
                    if t is not None:
                        yield t, axis


def _arc_crossings(seg, grid):
    """Yield ``(t, axis)`` interior grid-line crossings of an elliptical
    Arc.  With ``a = radians(theta + delta*t)`` the point decomposes into two
    independent sinusoids,

        x(a) = cx + Ax*cos(a + px)
        y(a) = cy + Ay*cos(a + py)

    so each grid line is solved with one arccos and a period scan."""
    xmin, xmax, ymin, ymax = seg.bbox()
    theta = radians(seg.theta)
    delta = radians(seg.delta)
    if delta == 0.0:
        return
    phi = seg.phi
    rx, ry = seg.radius.real, seg.radius.imag
    cos_phi, sin_phi = seg.rot_matrix.real, seg.rot_matrix.imag

    ax = np.hypot(rx * cos_phi, ry * sin_phi)
    px = np.arctan2(ry * sin_phi, rx * cos_phi)
    ay = np.hypot(rx * sin_phi, ry * cos_phi)
    py = np.arctan2(-ry * cos_phi, rx * sin_phi)

    def solve(offset, base, amp, phase):
        if amp == 0.0:
            return
        q = (offset - base) / amp
        if q < -1.0 - 1e-12 or q > 1.0 + 1e-12:
            return
        q = min(1.0, max(-1.0, q))
        beta = np.arccos(q)
        # |delta| <= 2*pi, so a spans at most one full turn; scanning a few
        # periods on each side covers every root in the parameter range.
        for k in range(-3, 4):
            for root_angle in (beta + 2 * pi * k, -beta + 2 * pi * k):
                t = _accept_root((root_angle - phase - theta) / delta)
                if t is not None:
                    yield t

    specs = (
        (0, True, xmin, xmax, seg.center.real, ax, px),
        (1, False, ymin, ymax, seg.center.imag, ay, py),
    )
    for axis, vertical, lo, hi, base, amp, phase in specs:
        for k in grid.candidate_indices(lo, hi, vertical):
            for t in solve(grid.line_value(k, vertical), base, amp, phase):
                yield t, axis


def _segment_cells(seg, grid):
    """Return the set of ``(i, j)`` cells intersected by one segment's
    center line."""
    if isinstance(seg, Line):
        events = list(_line_crossings(seg, grid))
    elif isinstance(seg, (QuadraticBezier, CubicBezier)):
        events = list(_bezier_crossings(seg, grid))
    elif isinstance(seg, Arc):
        events = list(_arc_crossings(seg, grid))
    else:
        raise TypeError(
            "unsupported path segment type: {}".format(type(seg).__name__))

    ts = _merge_events([t for t, _axis in events])
    if not ts:
        ts = [0.0, 1.0]
    else:
        if ts[0] != 0.0:
            ts.insert(0, 0.0)
        if ts[-1] != 1.0:
            ts.append(1.0)

    cells = set()
    # Every crossing/contact point owns its cell directly.  Registering event
    # points themselves (in addition to interval midpoints) makes tangencies
    # robust: a curve that touches a grid line and turns back occupies the
    # high-side cell at the contact point even though both surrounding
    # intervals lie in a different cell.
    for t in ts:
        cells.add(grid.cell_of(seg.point(t)))
    for t0, t1 in zip(ts[:-1], ts[1:]):
        if t1 - t0 <= _T_EPS:
            continue
        cells.add(grid.cell_of(seg.point(0.5 * (t0 + t1))))
    # Explicit endpoints cover the degenerate single-point segment case.
    cells.add(grid.cell_of(seg.point(0.0)))
    cells.add(grid.cell_of(seg.point(1.0)))
    return cells


def _is_finite_real(value, name):
    if not np.isfinite(value):
        raise ValueError("{} must be finite, got {}".format(name, value))


def grid_coverage(path, origin=0j, cell_size=1 + 1j, *,
                  tol=1e-8, max_cells=10000):
    """Find the grid cells intersected by a path's stroke center line.

    Args:
        path: a :class:`~svgpathtools.path.Path` or a single path segment
            (:class:`Line`, :class:`QuadraticBezier`, :class:`CubicBezier`
            or :class:`Arc`).
        origin (complex): lower-left corner of cell ``(0, 0)`` as
            ``origin_x + 1j*origin_y``.  Cell coordinates may be negative.
        cell_size (complex): cell width and height as ``cell_w + 1j*cell_h``.
            Both must be strictly positive.
        tol (float): non-negative geometric tolerance in path-coordinate
            units.  A point within ``tol`` below its cell's upper grid line is
            treated as lying on that line, so floating-point noise cannot
            scatter boundary points across neighboring cells.  The effective
            tolerance is widened automatically for very large coordinates to
            stay above machine precision.
        max_cells (int): safety cap on the number of distinct cells
            returned.  Exceeding it raises :class:`GridCoverageLimit`
            instead of returning partial results or rasterizing the grid.

    Returns:
        list[GridCell]: cells actually intersected by the center line,
        sorted by ``(j, i)`` (row then column) for a stable order.  Each
        cell's :attr:`~GridCell.segments` attribute lists the path segment
        indices whose center line passes through it.  Cells only touched by
        the path's bounding box (and not by the curve itself) are never
        included.

    Raises:
        ValueError: non-finite coordinates, non-positive cell size, negative
            tolerance, an invalid ``max_cells``, or a path containing
            non-finite control points.
        TypeError: ``path`` is neither a Path nor a supported segment.
        GridCoverageLimit: more than ``max_cells`` distinct cells would be
            produced.
    """
    if isinstance(path, Path):
        segments = list(path)
    elif is_path_segment(path):
        segments = [path]
    else:
        raise TypeError(
            "path must be a Path or path segment, got {}".format(
                type(path).__name__))

    origin_x, origin_y = origin.real, origin.imag
    cell_w, cell_h = cell_size.real, cell_size.imag
    _is_finite_real(origin_x, 'origin_x')
    _is_finite_real(origin_y, 'origin_y')
    _is_finite_real(cell_w, 'cell_w')
    _is_finite_real(cell_h, 'cell_h')
    if cell_w <= 0.0:
        raise ValueError(
            "cell_w must be strictly positive, got {}".format(cell_w))
    if cell_h <= 0.0:
        raise ValueError(
            "cell_h must be strictly positive, got {}".format(cell_h))
    _is_finite_real(tol, 'tol')
    if tol < 0.0:
        raise ValueError("tol must be non-negative, got {}".format(tol))
    if not isinstance(max_cells, (int, np.integer)) or max_cells <= 0:
        raise ValueError(
            "max_cells must be a positive integer, got {}".format(max_cells))

    if not segments:
        return []

    # Reject non-finite geometry up front rather than producing NaN indices.
    for idx, seg in enumerate(segments):
        points = seg.apoints() if isinstance(seg, Arc) else seg.bpoints()
        for z in points:
            if isinstance(z, complex) and not (
                    np.isfinite(z.real) and np.isfinite(z.imag)):
                raise ValueError(
                    "segment {} contains non-finite coordinates".format(idx))

    # A tolerance below roughly a few ULPs of the involved coordinates cannot
    # distinguish anything; floor it deterministically.
    span = max(1.0, abs(origin_x), abs(origin_y))
    tol_x = max(tol, span * 1e-12, cell_w * 1e-12)
    tol_y = max(tol, span * 1e-12, cell_h * 1e-12)
    grid = _Grid(origin_x, origin_y, cell_w, cell_h, tol_x, tol_y)

    # Up-front work bound.  A curve inside a tight bbox spanning n_col columns
    # and n_row rows can touch at most n_col * n_row cells (the bbox's cell
    # count).  Evaluating this before any root solving means an extremely long
    # curve or a vanishingly small cell size fails immediately and
    # deterministically instead of allocating billions of crossing events.
    budget = 0
    for seg in segments:
        xmin, xmax, ymin, ymax = seg.bbox()
        n_col = floor((xmax - origin_x) / cell_w) - \
            floor((xmin - origin_x) / cell_w) + 1
        n_row = floor((ymax - origin_y) / cell_h) - \
            floor((ymin - origin_y) / cell_h) + 1
        budget += int(max(1, n_col) * max(1, n_row))
        if budget > max_cells:
            raise GridCoverageLimit(
                "grid coverage would exceed max_cells={}: a segment's "
                "tight bounding box alone spans {} candidate cells (check "
                "cell size and curve length)".format(max_cells, budget))

    cell_to_segments = {}
    for idx, seg in enumerate(segments):
        for cell in _segment_cells(seg, grid):
            cell_to_segments.setdefault(cell, []).append(idx)
            if len(cell_to_segments) > max_cells:
                raise GridCoverageLimit(
                    "grid coverage exceeds max_cells={}; the curve crosses "
                    "more than {} cells (check cell size and curve length)"
                    .format(max_cells, max_cells))

    return [GridCell(i, j, cell_to_segments[(i, j)])
            for (i, j) in sorted(cell_to_segments, key=lambda c: (c[1], c[0]))]
