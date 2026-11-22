"""Tools for welding (joining) endpoints of open paths across small gaps.

Such gaps are commonly produced by CAD/laser-cutting exports whose
segments miss each other by a fraction of a pixel.

The main entry point is :func:`weld_paths`.  It connects path endpoints
whose separation is within a user-supplied distance tolerance, reversing
fragments where necessary, and reports every connection it makes.

Welding rules (deterministic)
-----------------------------
1.  Only path *endpoints* are ever connected -- never a point in the
    interior of a curve, so T-junctions whose stem ends on the middle of
    another contour are left alone, and distinct contours are never
    bridged except endpoint-to-endpoint.
2.  All candidate endpoint pairs within ``tolerance`` compete globally;
    the smallest gap is welded first.  Ties are broken by input order,
    so the result is deterministic for a given input ordering (and fully
    order-independent whenever no two candidate gaps are exactly equal).
3.  Each endpoint takes part in at most one weld.  At a fork (one
    endpoint within tolerance of several others) the nearest candidate
    wins and the losers stay open.
4.  A chain whose own two ends come within tolerance is closed (a
    self-weld), competing with external candidates by the same distance
    rule.  Single-segment chains are never self-welded (that would
    collapse the segment to a point); already-closed paths are passed
    through untouched.
5.  Gaps larger than ``tolerance`` are never bridged; the affected paths
    are returned open and unmodified (except for the copy).
6.  Welding moves endpoints only: each of the two joined endpoints is
    shifted to their midpoint, i.e. by at most ``tolerance/2``.  Control
    points and every interior segment are left bit-for-bit intact.
"""

# External Dependencies
from __future__ import division, absolute_import, print_function
from collections import namedtuple
from copy import deepcopy
from math import isfinite

# Internal Dependencies
from .path import Path


class Weld(namedtuple('Weld', ['kind', 'path_indices', 'ends', 'gap',
                               'moved', 'point'])):
    """Record of a single weld performed by :func:`weld_paths`.

    fields
    ------
    kind : str
        ``'join'`` if two different chains were connected, ``'close'`` if
        a chain's own two ends were welded to close a loop.
    path_indices : tuple
        Indices (into the input list) of the original paths owning the
        two welded endpoints (the same index twice if a chain of
        fragments grew back to its own first path).
    ends : tuple
        Which endpoint of each original path was welded, each entry
        being ``'start'`` or ``'end'`` (relative to the path's original
        orientation, before any reversal).
    gap : float
        Distance between the two endpoints before welding.
    moved : float
        Distance each endpoint was shifted to the common joint point
        (always ``gap / 2``).
    point : complex
        The joint point both endpoints were moved to.
    """

    __slots__ = ()


# sides of a chain/endpoint
_START = 0
_END = 1
_SIDE_NAMES = {_START: 'start', _END: 'end'}


class _Chain(object):
    """A growable sequence of (copied) segments with two exposed ends.

    ``start_src``/``end_src`` record which endpoint of which original
    path is currently exposed at each end, as ``(path_index, side)``.
    """

    __slots__ = ('segments', 'start_src', 'end_src')

    def __init__(self, segments, start_src, end_src):
        self.segments = segments
        self.start_src = start_src
        self.end_src = end_src

    @property
    def start(self):
        return self.segments[0].start

    @property
    def end(self):
        return self.segments[-1].end

    def point_at(self, side):
        return self.start if side == _START else self.end

    def reversed(self):
        """Return a new _Chain with opposite orientation."""
        segs = [seg.reversed() for seg in reversed(self.segments)]
        return _Chain(segs, self.end_src, self.start_src)


def _check_finite(path, index):
    """Raise ValueError if any segment endpoint of `path` is non-finite."""
    for seg in path:
        for pt in (seg.start, seg.end):
            if not (isfinite(pt.real) and isfinite(pt.imag)):
                raise ValueError(
                    "weld_paths: path %d has a non-finite endpoint %r; "
                    "refusing to weld non-finite geometry." % (index, pt))


def weld_paths(paths, tolerance):
    """Weld the endpoints of open paths whose gaps are within `tolerance`.

    Input
    -----
    paths : iterable of Path
        The paths to weld.  They are never modified; welding operates on
        deep copies.
    tolerance : float
        Non-negative, finite distance tolerance.  Endpoint pairs at most
        this far apart may be welded.  ``tolerance=0`` welds only
        endpoints that coincide exactly.

    Output
    ------
    (welded_paths, welds) : (list of Path, list of Weld)
        ``welded_paths`` are new Path objects; the input paths are left
        untouched.  ``welds`` lists every connection made, in the order
        it was made (see the Weld docstring and the module docstring for
        the exact rules).

    Notes
    -----
    * Paths with no segments are passed through unchanged and never
      welded.
    * Zero-length segments follow the same endpoint rules as any other
      segment: an exposed endpoint may be welded (moving it by at most
      ``tolerance / 2``).  A path consisting solely of zero-length
      segments has coincident ends and is therefore already closed, so
      it is passed through without a self-weld.
    * A ValueError is raised if `tolerance` is negative or non-finite,
      or if any segment endpoint has a non-finite coordinate.
    * Metadata (``transform``, ``meta``, etc.) is not carried over to
      the output paths.
    """
    if not (isinstance(tolerance, (int, float)) and isfinite(tolerance)
            and tolerance >= 0):
        raise ValueError("tolerance must be a finite, non-negative number, "
                         "got %r" % (tolerance,))

    paths = list(paths)
    chains = []
    passthrough = []  # (input_index, Path) for empty paths
    for idx, path in enumerate(paths):
        if not isinstance(path, Path):
            raise TypeError("weld_paths expects Path objects, got %r"
                            % type(path))
        _check_finite(path, idx)
        if len(path) == 0:
            passthrough.append((idx, Path()))
            continue
        segments = [deepcopy(seg) for seg in path]
        chains.append(_Chain(segments, (idx, _START), (idx, _END)))

    welds = []
    closed_chains = []  # (min input index, segments)

    def try_weld(a, side_a, b, side_b):
        """Weld `side_a` of chain a to `side_b` of chain b (a is not b)."""
        pa, pb = a.point_at(side_a), b.point_at(side_b)
        gap = abs(pa - pb)
        joint = (pa + pb) / 2
        # Orient both chains so the welded ends face each other:
        # a contributes its END, b contributes its START.
        if side_a == _START:
            a = a.reversed()
        if side_b == _END:
            b = b.reversed()
        a.segments[-1].end = joint
        b.segments[0].start = joint
        welds.append(Weld(
            kind='join',
            path_indices=(a.end_src[0], b.start_src[0]),
            ends=(_SIDE_NAMES[a.end_src[1]], _SIDE_NAMES[b.start_src[1]]),
            gap=gap, moved=gap / 2, point=joint))
        return _Chain(a.segments + b.segments, a.start_src, b.end_src)

    def try_close(a):
        """Weld chain a's own two ends, closing the loop."""
        pa, pb = a.start, a.end
        gap = abs(pa - pb)
        joint = (pa + pb) / 2
        a.segments[0].start = joint
        a.segments[-1].end = joint
        welds.append(Weld(
            kind='close',
            path_indices=(a.start_src[0], a.end_src[0]),
            ends=(_SIDE_NAMES[a.start_src[1]], _SIDE_NAMES[a.end_src[1]]),
            gap=gap, moved=gap / 2, point=joint))
        return a.segments

    while True:
        # Find the best (smallest-gap) candidate among all exposed ends.
        best = None  # (sort_key, kind, i, side_i, j, side_j)
        for i in range(len(chains)):
            for j in range(i, len(chains)):
                if i == j:
                    # Self-closure: needs >= 2 segments and not already
                    # closed; competes by the same distance rule.
                    if len(chains[i].segments) < 2:
                        continue
                    gap = abs(chains[i].start - chains[i].end)
                    if gap == 0 or gap > tolerance:
                        continue
                    key = (gap, chains[i].start_src[0], _START, _END)
                    if best is None or key < best[0]:
                        best = (key, 'close', i, _START, i, _END)
                else:
                    for side_i in (_START, _END):
                        for side_j in (_START, _END):
                            gap = abs(chains[i].point_at(side_i)
                                      - chains[j].point_at(side_j))
                            if gap > tolerance:
                                continue
                            key = (gap, chains[i].start_src[0],
                                   chains[j].start_src[0], side_i, side_j)
                            if best is None or key < best[0]:
                                best = (key, 'join', i, side_i, j, side_j)
        if best is None:
            break
        _, kind, i, side_i, j, side_j = best
        if kind == 'close':
            segments = try_close(chains[i])
            closed_chains.append((chains[i].start_src[0], segments))
            del chains[i]
        else:
            merged = try_weld(chains[i], side_i, chains[j], side_j)
            del chains[j]
            chains[i] = merged

    # Emit results in a deterministic order: by the smallest input index
    # contributing to each output path.
    results = []
    for first_idx, segments in closed_chains:
        results.append((first_idx, Path(*segments)))
    for chain in chains:
        results.append((chain.start_src[0], Path(*chain.segments)))
    results.extend(passthrough)
    results.sort(key=lambda item: item[0])
    return [path for _, path in results], welds
