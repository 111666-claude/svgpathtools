"""Tests for svgpathtools.weld (endpoint welding of open paths)."""

from __future__ import division, absolute_import, print_function

import unittest
from itertools import permutations

from svgpathtools import (Path, Line, CubicBezier, QuadraticBezier,
                          weld_paths, Weld)


def micro_gap_square(gap=1e-4):
    """Return a unit square of 4 lines, corner endpoints off by `gap`.

    Endpoints are offset symmetrically, so each welded joint lands near
    the ideal corner.
    """
    return [
        Path(Line(0 + 0j, 1 + gap * 1j)),          # bottom edge
        Path(Line(1 - gap * 1j, 1 + 1j + gap)),    # right edge
        Path(Line(1 + 1j - gap, 1j + gap * 1j)),   # top edge
        Path(Line(1j - gap * 1j, 0 + gap)),        # left edge
    ]


class TestWeldPaths(unittest.TestCase):
    """Endpoint-welding behavior, priority rules, and edge cases."""

    def test_micro_gap_contour_closes(self):
        """A laser-cut square with sub-pixel gaps welds into one loop."""
        gap = 1e-4
        paths = micro_gap_square(gap)
        originals = [p.d() for p in paths]

        welded, welds = weld_paths(paths, tolerance=1e-3)

        # topology: 4 open fragments -> 1 closed loop
        self.assertEqual(len(welded), 1)
        loop = welded[0]
        self.assertTrue(loop.iscontinuous())
        self.assertTrue(loop.isclosedac())
        self.assertEqual(len(loop), 4)

        # 3 joins between fragments + 1 closure
        self.assertEqual(len(welds), 4)
        self.assertEqual(sum(1 for w in welds if w.kind == 'join'), 3)
        self.assertEqual(sum(1 for w in welds if w.kind == 'close'), 1)

        # error bounds: every weld respected the tolerance, and each
        # endpoint moved by at most tolerance / 2
        for w in welds:
            self.assertIsInstance(w, Weld)
            self.assertLessEqual(w.gap, 1e-3 + 1e-12)
            self.assertAlmostEqual(w.moved, w.gap / 2)
            self.assertLessEqual(w.moved, 1e-3 / 2 + 1e-12)

        # every original path contributed exactly one endpoint record
        # (3 joins touch 3 pairs + closure touches the remaining pair)
        touched = sorted(i for w in welds for i in w.path_indices)
        self.assertEqual(touched, [0, 0, 1, 1, 2, 2, 3, 3])

        # originals untouched
        self.assertEqual([p.d() for p in paths], originals)

    def test_welded_loop_geometry_stays_put(self):
        """Welding moves endpoints by <= tol/2; interior stays exact."""
        gap = 2e-4
        paths = micro_gap_square(gap)
        welded, welds = weld_paths(paths, tolerance=1e-3)
        loop = welded[0]
        # every vertex of the welded loop is within gap/2 of a vertex of
        # the ideal unit square
        ideal = {0j, 1 + 0j, 1 + 1j, 1j}
        for seg in loop:
            self.assertTrue(any(abs(seg.start - v) <= gap / 2 + 1e-12
                                for v in ideal))

    def test_interior_control_points_untouched(self):
        """Mid-curve control points must survive welding bit-for-bit."""
        c1, c2 = 0.3 + 0.9j, 0.7 + 0.8j
        a = Path(CubicBezier(0j, 0.1 + 0.5j, c1, 1 + 0.5j))
        b = Path(CubicBezier(1 + 0.5j + 1e-4, c2, 0.9 + 0.2j, 2 + 0j))
        welded, welds = weld_paths([a, b], tolerance=1e-3)
        self.assertEqual(len(welded), 1)
        self.assertEqual(len(welds), 1)
        seg0, seg1 = welded[0]
        self.assertEqual(seg0.control1, 0.1 + 0.5j)
        self.assertEqual(seg0.control2, c1)   # interior control point
        self.assertEqual(seg1.control1, c2)   # interior control point
        self.assertEqual(seg1.control2, 0.9 + 0.2j)
        # only the two welded endpoints moved, to their midpoint
        joint = (1 + 0.5j + (1 + 0.5j + 1e-4)) / 2
        self.assertEqual(seg0.end, joint)
        self.assertEqual(seg1.start, joint)

    def test_reversal_when_needed(self):
        """Fragments facing the wrong way are reversed, not distorted."""
        a = Path(Line(0j, 1 + 0j))
        b = Path(Line(2 + 0j, 1 + 1e-4 + 0j))  # runs right-to-left
        welded, welds = weld_paths([a, b], tolerance=1e-3)
        self.assertEqual(len(welded), 1)
        self.assertTrue(welded[0].iscontinuous())
        self.assertEqual(welded[0].start, 0j)
        self.assertEqual(welded[0].end, 2 + 0j)
        # source records the original orientation of the welded ends
        self.assertEqual(welds[0].path_indices, (0, 1))
        self.assertEqual(welds[0].ends, ('end', 'end'))

    def test_t_junction_fork_nearest_wins(self):
        """Forks resolve to the nearest endpoint; interiors never weld."""
        stem = Path(Line(0.5 + 0.5j, 0.5 + 1e-4 + 0j))   # ends near bar
        bar = Path(Line(0j, 1 + 0j))                      # stem hits its
        # interior at 0.5+0j -- interior points must never weld
        welded, welds = weld_paths([stem, bar], tolerance=1e-3)
        self.assertEqual(len(welded), 2)   # no endpoint pair within tol
        self.assertEqual(len(welds), 0)

        # True endpoint fork: end of `a` is within tol of both ends of
        # `b` and `c`; the nearer one wins.
        a = Path(Line(0j, 1 + 0j))
        b = Path(Line(1 + 1e-4 + 0j, 2 + 0j))   # gap 1e-4 from a.end
        c = Path(Line(1 + 5e-4 + 0j, 3 + 0j))   # gap 5e-4 from a.end
        welded, welds = weld_paths([a, b, c], tolerance=1e-3)
        self.assertEqual(len(welds), 1)
        self.assertEqual(welds[0].path_indices, (0, 1))
        self.assertAlmostEqual(welds[0].gap, 1e-4)
        # a+b merged; c left open and geometrically unchanged
        self.assertEqual(len(welded), 2)
        merged = welded[0]
        self.assertTrue(merged.iscontinuous())
        self.assertEqual(merged.end, 2 + 0j)
        loner = welded[1]
        self.assertEqual(loner.start, 1 + 5e-4 + 0j)
        self.assertEqual(loner.end, 3 + 0j)

    def test_shuffled_input_same_topology(self):
        """Input order must not change the welded topology."""
        base = micro_gap_square(1e-4) + [
            Path(Line(5 + 0j, 6 + 0j)),            # isolated, stays open
            Path(QuadraticBezier(5 + 1e-4 + 0j, 5.5 + 0.5j, 6 + 1e-4 + 0j)),
        ]
        ref_paths, ref_welds = weld_paths(base, tolerance=1e-3)
        ref_sig = sorted((len(p), p.isclosedac(),
                          round(p.length(), 9)) for p in ref_paths)
        for perm in list(permutations(range(len(base))))[1:]:
            shuffled = [base[i] for i in perm]
            out_paths, out_welds = weld_paths(shuffled, tolerance=1e-3)
            sig = sorted((len(p), p.isclosedac(),
                          round(p.length(), 9)) for p in out_paths)
            self.assertEqual(sig, ref_sig)
            self.assertEqual(len(out_welds), len(ref_welds))
            for p in out_paths:
                self.assertTrue(p.iscontinuous())

    def test_over_tolerance_gap_not_bridged(self):
        a = Path(Line(0j, 1 + 0j))
        b = Path(Line(1.5 + 0j, 2 + 0j))
        welded, welds = weld_paths([a, b], tolerance=1e-3)
        self.assertEqual(len(welded), 2)
        self.assertEqual(len(welds), 0)
        self.assertEqual(welded[0], a)
        self.assertEqual(welded[1], b)

    def test_zero_tolerance_exact_only(self):
        a = Path(Line(0j, 1 + 0j))
        b = Path(Line(1 + 0j, 2 + 0j))          # exact touch
        c = Path(Line(2 + 1e-12 + 0j, 3 + 0j))  # inexact touch
        welded, welds = weld_paths([a, b, c], tolerance=0)
        self.assertEqual(len(welded), 2)
        self.assertEqual(len(welds), 1)
        self.assertEqual(welds[0].gap, 0)
        self.assertEqual(welds[0].moved, 0)
        self.assertEqual(welds[0].path_indices, (0, 1))
        self.assertTrue(welded[0].iscontinuous())
        self.assertEqual(welded[1], c)

    def test_zero_length_segments_preserved(self):
        """Zero-length segments weld by the same endpoint rules: the
        exposed endpoint moves by at most tolerance/2, nothing else."""
        z = Path(Line(1 + 1j, 1 + 1j))           # zero-length segment
        a = Path(Line(0j, 1 + 1j - 1e-4))
        welded, welds = weld_paths([z, a], tolerance=1e-3)
        self.assertEqual(len(welded), 1)
        self.assertEqual(len(welded[0]), 2)      # segment kept, not dropped
        self.assertTrue(welded[0].iscontinuous())
        self.assertEqual(len(welds), 1)
        self.assertAlmostEqual(welds[0].gap, 1e-4)
        # far ends of the chain did not move at all
        self.assertEqual(welded[0].start, 1 + 1j)
        self.assertEqual(welded[0].end, 0j)
        # the welded joint moved each endpoint by exactly gap/2
        self.assertAlmostEqual(
            abs(welded[0][0].end - (1 + 1j - 1e-4)), 1e-4 / 2)

    def test_zero_length_only_path_passes_through(self):
        z = Path(Line(2 + 2j, 2 + 2j), Line(2 + 2j, 2 + 2j))
        welded, welds = weld_paths([z], tolerance=1e-3)
        self.assertEqual(len(welded), 1)
        self.assertEqual(welded[0], z)   # coincident ends: already closed
        self.assertEqual(len(welds), 0)

    def test_single_segment_not_self_closed(self):
        """A lone segment near-closing on itself must not collapse."""
        a = Path(Line(0j, 1e-4 + 0j))
        welded, welds = weld_paths([a], tolerance=1e-3)
        self.assertEqual(len(welded), 1)
        self.assertEqual(welded[0], a)
        self.assertEqual(len(welds), 0)

    def test_already_closed_path_untouched(self):
        loop = Path(Line(0j, 1 + 0j), Line(1 + 0j, 1 + 1j),
                    Line(1 + 1j, 0j))
        other = Path(Line(5 + 0j, 5 + 1e-4 + 0j))  # within tol of nothing
        welded, welds = weld_paths([loop, other], tolerance=1e-3)
        self.assertEqual(len(welded), 2)
        self.assertEqual(welded[0], loop)
        self.assertEqual(len(welds), 0)

    def test_non_finite_coordinates_raise(self):
        for bad in (float('nan'), float('inf'), float('-inf')):
            p = Path(Line(0j, 1 + 0j), Line(1 + 0j, bad + 0j))
            with self.assertRaises(ValueError):
                weld_paths([p], tolerance=1e-3)
        with self.assertRaises(ValueError):
            weld_paths([Path(Line(0j, 1 + 0j))], tolerance=float('nan'))
        with self.assertRaises(ValueError):
            weld_paths([Path(Line(0j, 1 + 0j))], tolerance=-1)

    def test_empty_and_mixed_input(self):
        welded, welds = weld_paths([], tolerance=1e-3)
        self.assertEqual(welded, [])
        self.assertEqual(welds, [])
        welded, welds = weld_paths([Path()], tolerance=1e-3)
        self.assertEqual(len(welded), 1)
        self.assertEqual(len(welded[0]), 0)

    def test_originals_never_modified(self):
        paths = micro_gap_square(1e-4)
        snapshots = [(p.d(), [s.start for s in p], [s.end for s in p])
                     for p in paths]
        weld_paths(paths, tolerance=1e-3)
        for p, (d, starts, ends) in zip(paths, snapshots):
            self.assertEqual(p.d(), d)
            self.assertEqual([s.start for s in p], starts)
            self.assertEqual([s.end for s in p], ends)

    def test_multi_fragment_chain_and_record_order(self):
        """Welds happen smallest-gap-first; records carry sources."""
        a = Path(Line(0j, 1 + 0j))
        b = Path(Line(1 + 3e-4 + 0j, 2 + 0j))
        c = Path(Line(2 + 1e-4 + 0j, 3 + 0j))
        welded, welds = weld_paths([a, b, c], tolerance=1e-3)
        self.assertEqual(len(welded), 1)
        self.assertTrue(welded[0].iscontinuous())
        self.assertEqual(len(welds), 2)
        # smaller gap (b-c, 1e-4) welded before a-b (3e-4)
        self.assertLess(welds[0].gap, welds[1].gap)
        self.assertEqual(welds[0].path_indices, (1, 2))
        self.assertEqual(welds[1].path_indices, (0, 1))


if __name__ == '__main__':
    unittest.main()
