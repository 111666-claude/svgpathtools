"""Conservative spatial indexing for path segment intersection queries.

The module provides :class:`SegmentSpatialIndex`, a reusable uniform-grid
("spatial hash") index over the axis-aligned bounding boxes of a sequence of
path segments.  The index is *conservative*: every pair of segments whose
bounding boxes overlap (within a numerical padding) is guaranteed to be
returned by :meth:`SegmentSpatialIndex.candidate_pairs`, so it can only
reduce the number of pairs that reach the exact intersection routine -- it
can never drop a true intersection.

Degenerate geometries (empty inputs, zero-size bounding boxes, infinite
coordinates, every box overlapping every other box) are handled by falling
back to a brute-force enumeration, so the caller always receives correct
(if unscreened) candidates instead of a memory blow-up.
"""

from math import isfinite

__all__ = ['SegmentSpatialIndex', 'segment_candidate_pairs',
           'MAX_GRID_OCCUPANCY', 'GRID_MEMBERSHIP_FACTOR']

# A single grid cell containing more than this many segments means the
# distribution is too clustered for a uniform grid to pay off; the index
# degrades to brute force rather than building a pathological hash table.
MAX_GRID_OCCUPANCY = 64

# If the total number of segment-to-cell memberships exceeds this factor
# times the number of segments, the grid is considered too dense (segments
# span too many cells) and the index degrades to brute force.
GRID_MEMBERSHIP_FACTOR = 8

# A single segment may never occupy more than this many grid cells per
# axis.  Without a cap, a thin segment (e.g. a horizontal line whose bbox
# height is only the numerical padding) combined with a similarly small
# median cell size would span billions of cells.  When the cap is reached
# the index degrades instead of looping for an unbounded time.
MAX_CELLS_PER_SEGMENT_AXIS = 128


def _padded_bbox(seg, tol=1e-12):
    """Return ``(xmin, xmax, ymin, ymax)`` bbox of ``seg``.

    The box is padded by a conservative amount so that two segments only
    touching at an endpoint, lying on a shared boundary, or sitting within
    ``tol`` of each other cannot be separated by the index.  The padding
    also absorbs ordinary floating-point error in bbox computations
    (notably the trigonometric :meth:`Arc.bbox`).
    """
    xmin, xmax, ymin, ymax = seg.bbox()
    pad = tol + 1e-12 * max(1.0,
                            abs(xmin), abs(xmax), abs(ymin), abs(ymax))
    return xmin - pad, xmax + pad, ymin - pad, ymax + pad


def _bboxes_overlap(box1, box2):
    """Return whether two ``(xmin, xmax, ymin, ymax)`` boxes intersect.

    Boundary contact counts as an overlap (inclusive comparisons), which
    is essential for tangencies and endpoint-on-endpoint intersections.
    """
    return (box1[0] <= box2[1] and box2[0] <= box1[1] and
            box1[2] <= box2[3] and box2[2] <= box1[3])


class SegmentSpatialIndex:
    """A reusable conservative spatial index over path segments.

    The index stores the (padded) axis-aligned bounding box of every
    segment and, when the geometry is suitable, a uniform grid mapping
    grid cells to the segments overlapping them.

    Args:
        segments: an iterable of path segments (objects exposing
            ``bbox()``).  May be empty.
        tol (float): tolerance used both to pad the bounding boxes (so
            near-boundary contacts are never missed) and as the reference
            scale for the floating-point safety margin.

    Degenerate inputs (no segments, non-finite coordinates, a
    zero-extent world, or an overly dense grid) result in an index in
    "degraded" mode, whose :meth:`candidate_pairs` method enumerates all
    pairs after a direct bbox overlap check.  This keeps worst-case memory
    linear in the number of segments instead of quadratic.
    """

    def __init__(self, segments, tol=1e-12):
        """Build a conservative spatial index over ``segments``.

        Args:
            segments: an iterable of path segments (objects exposing
                ``bbox()``).  May be empty.
            tol (float): tolerance used both to pad the bounding boxes (so
                near-boundary contacts are never missed) and as the
                reference scale for the floating-point safety margin.

        Degenerate inputs (no segments, non-finite coordinates, a
        zero-extent world, or an overly dense grid) result in an index in
        "degraded" mode, whose :meth:`candidate_pairs` method enumerates
        all pairs after a direct bbox overlap check.  This keeps
        worst-case memory linear in the number of segments instead of
        quadratic.
        """
        self.tol = tol
        self.bboxes = [_padded_bbox(seg, tol) for seg in segments]
        self.n = len(self.bboxes)
        self.degraded = False
        self.degrade_reason = None
        # Maps (ix, iy) grid cell -> list of segment indices.  Remains
        # empty in degraded mode.
        self._cells = {}
        self._cell_w = 1.0
        self._cell_h = 1.0
        self._x0 = 0.0
        self._y0 = 0.0
        self._build()

    # ------------------------------------------------------------------ #
    # Construction
    # ------------------------------------------------------------------ #
    def _degrade(self, reason):
        """Switch to brute-force mode and record why."""
        self.degraded = True
        self.degrade_reason = reason
        self._cells = {}

    def _build(self):
        """Build the uniform grid, or degrade when it cannot help."""
        if self.n == 0:
            self._degrade('empty path')
            return

        boxes = self.bboxes
        xmin = min(b[0] for b in boxes)
        xmax = max(b[1] for b in boxes)
        ymin = min(b[2] for b in boxes)
        ymax = max(b[3] for b in boxes)

        # Non-finite coordinates cannot be placed on a finite grid.
        if not all(map(isfinite, (xmin, xmax, ymin, ymax))):
            self._degrade('non-finite bounding-box coordinates')
            return

        # Pick a cell size near the median segment extent so that typical
        # segments occupy only one or a few cells.
        widths = sorted(b[1] - b[0] for b in boxes)
        heights = sorted(b[3] - b[2] for b in boxes)
        mid = self.n // 2
        cell_w = widths[mid]
        cell_h = heights[mid]

        world_w = xmax - xmin
        world_h = ymax - ymin
        if cell_w <= 0.0 or cell_h <= 0.0 or world_w <= 0.0 or \
                world_h <= 0.0:
            # Every segment is a point (or collinear on one axis): a grid
            # has no resolving power on that axis.
            self._degrade('zero-extent geometry')
            return

        # Reject grid dimensions that would overflow or iterate an
        # absurd number of cells (extreme coordinates, or a median cell
        # size vanishingly small relative to the world extent).
        grid_nx = int(world_w / cell_w) + 1
        grid_ny = int(world_h / cell_h) + 1
        if grid_nx > MAX_CELLS_PER_SEGMENT_AXIS * self.n or \
                grid_ny > MAX_CELLS_PER_SEGMENT_AXIS * self.n:
            self._degrade('grid resolution too fine for coordinate range')
            return

        self._cell_w = cell_w
        self._cell_h = cell_h
        self._x0 = xmin
        self._y0 = ymin

        cells = self._cells
        memberships = 0
        max_ix = int(world_w / cell_w)
        max_iy = int(world_h / cell_h)
        for i, (bx0, bx1, by0, by1) in enumerate(boxes):
            ix0 = int((bx0 - xmin) / cell_w)
            iy0 = int((by0 - ymin) / cell_h)
            ix1 = min(int((bx1 - xmin) / cell_w), max_ix)
            iy1 = min(int((by1 - ymin) / cell_h), max_iy)
            # Thin segments (zero bbox extent plus numerical padding) can
            # still span far more cells than the median segment: cap the
            # per-axis span rather than materializing a huge cell range.
            if ix1 - ix0 > MAX_CELLS_PER_SEGMENT_AXIS or \
                    iy1 - iy0 > MAX_CELLS_PER_SEGMENT_AXIS:
                self._degrade('segment spans too many grid cells')
                return
            for ix in range(ix0, ix1 + 1):
                for iy in range(iy0, iy1 + 1):
                    cells.setdefault((ix, iy), []).append(i)
                    memberships += 1
            if memberships > GRID_MEMBERSHIP_FACTOR * self.n:
                self._degrade('segments span too many grid cells')
                return

        # Heavy bbox overlap means screening would not save work.  Two
        # shapes of the pathology are handled:
        #   * some cell holds nearly the whole population (all boxes
        #     share one region; grid lookups degenerate into one huge
        #     bucket);
        #   * a fine grid still has cells holding half the population
        #     (genuinely clustered data).
        # A coarse-but-spread grid is kept: with few cells, high average
        # occupancy is a consequence of the cell count, not of overlap,
        # and the per-pair bbox re-check still screens efficiently.
        n_cells = len(cells)
        max_occupancy = max(len(v) for v in cells.values())
        if max_occupancy * 10 >= self.n * 9 or \
                (n_cells >= MAX_GRID_OCCUPANCY and
                 max_occupancy > MAX_GRID_OCCUPANCY and
                 max_occupancy >= self.n // 2):
            self._degrade('grid cell occupancy too high')
            return

    # ------------------------------------------------------------------ #
    # Queries
    # ------------------------------------------------------------------ #
    def candidate_pairs(self, other):
        """Return conservatively overlapping segment index pairs.

        Args:
            other (SegmentSpatialIndex): the index of the other segment
                sequence.  It may be this same index (self-intersection
                queries) or an index built for a different path.

        Returns:
            list[tuple[int, int]]: pairs ``(i, j)`` where ``i`` indexes a
            segment in this index and ``j`` indexes a segment in ``other``
            and the two (padded) bounding boxes overlap.  Pairs are
            produced in ascending lexicographic order ``(i, j)`` so that
            callers iterating them preserve a brute-force-compatible
            processing order.
        """
        pairs = []
        if self.n == 0 or other.n == 0:
            return pairs

        if self.degraded or other.degraded:
            return self._brute_force_pairs(other)

        same = other is self
        own_boxes = self.bboxes
        other_boxes = other.bboxes
        seen = set()
        for i, box in enumerate(own_boxes):
            # Self-intersection: only look at strictly later segments so
            # each unordered pair is generated once.
            j_start = i + 1 if same else 0
            jcand = self._overlapping_indices(box, other)
            if jcand is None:
                # The box would span an absurd number of cells of the
                # other grid: fall back to a direct bbox comparison for
                # this segment instead of materializing the range.
                jcand = range(j_start, other.n)
            for j in jcand:
                if same and j < j_start:
                    continue
                # Re-check the actual boxes: grid co-membership is a
                # necessary but not sufficient condition, and inclusive
                # comparisons keep tangencies/endpoints as candidates.
                if _bboxes_overlap(box, other_boxes[j]):
                    key = (i, j)
                    if key not in seen:
                        seen.add(key)
                        pairs.append(key)
        pairs.sort()
        return pairs

    def _brute_force_pairs(self, other):
        """All bbox-overlapping pairs found by direct comparison."""
        pairs = []
        own_boxes = self.bboxes
        other_boxes = other.bboxes
        same = other is self
        for i, box in enumerate(own_boxes):
            j_start = i + 1 if same else 0
            for j in range(j_start, other.n):
                if _bboxes_overlap(box, other_boxes[j]):
                    pairs.append((i, j))
        return pairs

    def _overlapping_indices(self, box, other):
        """Return segment indices in ``other`` sharing a grid cell.

        Considers segments whose cell overlaps the given (already padded)
        bounding box.  Returns ``None`` when the box would span an
        excessive number of the other grid's cells, signalling the caller
        to fall back to a direct bounding-box comparison.
        """
        bx0, bx1, by0, by1 = box
        # Clamp to the other grid's cell domain: a box outside that world
        # yields negative (or beyond-end) indices, which match no cells and
        # must not be mistaken for an excessive span.
        ix0 = max(0, int((bx0 - other._x0) / other._cell_w))
        iy0 = max(0, int((by0 - other._y0) / other._cell_h))
        ix1 = int((bx1 - other._x0) / other._cell_w)
        iy1 = int((by1 - other._y0) / other._cell_h)
        if ix1 < 0 or iy1 < 0:
            return []
        if ix1 - ix0 > MAX_CELLS_PER_SEGMENT_AXIS or \
                iy1 - iy0 > MAX_CELLS_PER_SEGMENT_AXIS:
            return None
        found = []
        cells = other._cells
        for ix in range(ix0, ix1 + 1):
            for iy in range(iy0, iy1 + 1):
                bucket = cells.get((ix, iy))
                if bucket is not None:
                    found.extend(bucket)
        return found


def segment_candidate_pairs(segments1, segments2, tol=1e-12):
    """Build ad-hoc indices and return conservative candidate pairs.

    Convenience wrapper used by callers that do not keep a cached index
    around.  Returns a list of ``(i, j)`` index pairs in ascending
    lexicographic order.
    """
    index1 = SegmentSpatialIndex(segments1, tol=tol)
    index2 = SegmentSpatialIndex(segments2, tol=tol)
    return index1.candidate_pairs(index2)
