# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
from math import sqrt, pi, cos, sin
import numpy as np
from scipy.optimize import differential_evolution

# Internal dependencies
from svgpathtools import (
    Line, QuadraticBezier, CubicBezier, Arc, Path,
    distance, segment_distance, path_distance,
    SegmentDistance, PathDistance, DEFAULT_DISTANCE_TOL)
from svgpathtools.distance import _radial_min


TOL = 1e-9
ASSERT_TOL = 1e-8


def brute_force_seg_distance(seg1, seg2, n=400):
    """Independent reference: dense grid over the parameter square."""
    ts = np.linspace(0.0, 1.0, n)
    best = (np.inf, None, None)
    p1 = [seg1.point(t) for t in ts]
    p2 = [seg2.point(t) for t in ts]
    for i, a in enumerate(ts):
        for j, b in enumerate(ts):
            d = abs(p1[i] - p2[j])
            if d < best[0]:
                best = (d, a, b)
    return best


def scipy_seg_distance(seg1, seg2):
    """Independent global optimizer (used only as a fuzz-test oracle)."""
    def f(x):
        return abs(seg1.point(x[0]) - seg2.point(x[1]))
    res = differential_evolution(
        f, bounds=[(0, 1), (0, 1)], tol=1e-12, seed=0, polish=True,
        maxiter=400, popsize=20)
    return res.fun, res.x[0], res.x[1]


def make_arc(cx, cy, r, start_deg, end_deg, sweep=True, rx=None, ry=None):
    """Circular/elliptic arc from center and angular limits (degrees)."""
    rx = r if rx is None else rx
    ry = r if ry is None else ry
    start = complex(cx + rx * cos(radians(start_deg)),
                    cy + ry * sin(radians(start_deg)))
    end = complex(cx + rx * cos(radians(end_deg)),
                  cy + ry * sin(radians(end_deg)))
    large = abs(end_deg - start_deg) > 180
    return Arc(start, complex(rx, ry), 0, large, sweep, end)


def radians(deg):
    return deg * pi / 180.0


class TestSegmentDistance(unittest.TestCase):

    def test_returns_namedtuple_with_all_fields(self):
        l1 = Line(0, 1)
        l2 = Line(2j, 1 + 2j)
        res = segment_distance(l1, l2)
        self.assertIsInstance(res, SegmentDistance)
        self.assertEqual(res._fields,
                         ('dist', 'point1', 'point2', 't1', 't2'))
        for name in res._fields:
            self.assertTrue(np.isfinite(getattr(res, name)).all()
                            if isinstance(getattr(res, name), complex)
                            else np.isfinite(getattr(res, name)))

    # -- Line / parallel geometry ------------------------------------------

    def test_parallel_axis_aligned_lines(self):
        # Horizontal lines two units apart (the second sits at y = 2).
        l1 = Line(0, 10)
        l2 = Line(2j, 10 + 2j)
        res = segment_distance(l1, l2)
        self.assertAlmostEqual(res.dist, 2.0, places=10)
        self.assertAlmostEqual(abs(res.point1 - res.point2), 2.0, places=10)
        self.assertAlmostEqual(res.point1.imag, 0.0, places=10)
        self.assertAlmostEqual(res.point2.imag, 2.0, places=10)
        self.assertAlmostEqual(res.point1.real, res.point2.real, places=10)
        self.assertTrue(0.0 <= res.t1 <= 1.0 and 0.0 <= res.t2 <= 1.0)

    def test_parallel_slanted_lines(self):
        # Two lines parallel to y = x, separated by a perpendicular unit
        # vector (-1 + i)/sqrt(2) so the gap is exactly 1 and the
        # supporting lines are distinct.
        l1 = Line(0, 3 + 3j)
        off = (-1 + 1j) / sqrt(2)
        l2 = Line(off, 3 + 3j + off)
        res = segment_distance(l1, l2)
        self.assertAlmostEqual(res.dist, 1.0, places=8)
        # Closest segment connects perpendicular points: the displacement
        # vector is orthogonal to the line's tangent.
        delta = res.point2 - res.point1
        tangent = (l1.end - l1.start) / abs(l1.end - l1.start)
        self.assertAlmostEqual(
            delta.real * tangent.real + delta.imag * tangent.imag,
            0.0, places=8)
        self.assertAlmostEqual(abs(delta), 1.0, places=8)

    def test_collinear_overlapping_lines(self):
        # Collinear segments that overlap over an interval: the whole
        # overlap is a zero-distance set, and the lexicographic tie rule
        # picks the earliest shared point deterministically.
        l1 = Line(0, 4)
        l2 = Line(2, 6)
        r1 = segment_distance(l1, l2)
        r2 = segment_distance(l1, l2)
        self.assertEqual(r1.dist, 0.0)
        self.assertEqual(tuple(r1), tuple(r2))
        self.assertAlmostEqual(r1.point1, r1.point2, places=10)
        # Disjoint collinear segments: ordinary positive endpoint gap.
        gap = segment_distance(Line(0, 1), Line(3, 4))
        self.assertAlmostEqual(gap.dist, 2.0, places=10)

    def test_skew_lines_endpoint_minimum(self):
        # Two disjoint non-parallel segments whose closest pair is an
        # endpoint-to-interior-point pair.
        l1 = Line(0, 1)                      # on the real axis
        l2 = Line(2 + 1j, 2 - 1j)            # vertical line through x=2
        res = segment_distance(l1, l2)
        self.assertAlmostEqual(res.dist, 1.0, places=10)
        self.assertAlmostEqual(res.point1, 1.0, places=10)
        self.assertEqual(res.t1, 1.0)
        self.assertAlmostEqual(res.point2, 2.0, places=10)
        self.assertAlmostEqual(res.t2, 0.5, places=8)

    def test_intersecting_lines_zero_distance(self):
        l1 = Line(0, 2)
        l2 = Line(1 - 1j, 1 + 1j)
        res = segment_distance(l1, l2)
        self.assertEqual(res.dist, 0.0)
        self.assertAlmostEqual(res.point1, res.point2, places=10)
        self.assertAlmostEqual(res.point1, 1.0, places=10)
        self.assertAlmostEqual(res.t1, 0.5, places=8)
        self.assertAlmostEqual(res.t2, 0.5, places=8)

    # -- Bezier curves ------------------------------------------------------

    def test_translated_cubics_constant_gap(self):
        c1 = CubicBezier(0, 1 + 2j, 3 - 2j, 4)
        shift = 0.5 + 2j
        c2 = CubicBezier(*(bp + shift for bp in c1.bpoints()))
        res = segment_distance(c1, c2, tol=1e-12)
        # Independent global optimizer as the reference answer.
        d_ref, _, _ = scipy_seg_distance(c1, c2)
        self.assertAlmostEqual(res.dist, d_ref, places=9)
        self.assertTrue(res.dist <= abs(shift) + ASSERT_TOL)
        self.assertAlmostEqual(
            abs(c1.point(res.t1) - c2.point(res.t2)), res.dist, places=10)

    def test_intersecting_cubics_zero(self):
        c1 = CubicBezier(0, 1 + 1j, 2 - 1j, 3)
        c2 = CubicBezier(0 - 1j, 1, 2, 3 + 1j)
        res = segment_distance(c1, c2)
        self.assertEqual(res.dist, 0.0)
        self.assertAlmostEqual(abs(res.point1 - res.point2), 0.0,
                               places=8)

    def test_quadratic_to_line_vertex_gap(self):
        # Parabola y = (x-1)^2 on x in [0, 2], minimum vertex at (1, 0).
        q = QuadraticBezier(0 + 1j, 1 - 1j, 2 + 1j)
        line = Line(0 - 0.25j, 2 - 0.25j)
        res = segment_distance(q, line)
        self.assertAlmostEqual(res.dist, 0.25, places=7)
        self.assertAlmostEqual(res.t1, 0.5, places=6)
        self.assertAlmostEqual(res.point1, 1.0, places=6)
        # With a tolerance larger than the gap it is treated as touching.
        res_touch = segment_distance(q, line, tol=0.5)
        self.assertEqual(res_touch.dist, 0.0)
        # With a small tolerance the true positive gap survives.
        res_gap = segment_distance(q, line, tol=1e-12)
        self.assertAlmostEqual(res_gap.dist, 0.25, places=7)

    def test_coincident_curves_zero_and_deterministic(self):
        c = CubicBezier(0, 1 + 2j, 3 - 2j, 4)
        q = QuadraticBezier(0, 1 + 3j, 2)
        for seg in (c, q, Line(0, 1)):
            r1 = segment_distance(seg, seg)
            r2 = segment_distance(seg, seg)
            self.assertEqual(r1.dist, 0.0)
            self.assertEqual(r1.t1, 0.0)
            self.assertEqual(r1.t2, 0.0)
            self.assertEqual(tuple(r1), tuple(r2))

    # -- Arcs ---------------------------------------------------------------

    def test_concentric_circular_arcs(self):
        outer = make_arc(0, 0, 3, 0, 90)
        inner = make_arc(0, 0, 2, 0, 90)
        res = segment_distance(outer, inner)
        self.assertAlmostEqual(res.dist, 1.0, places=7)
        # Closest points lie on the same ray (their polar angles agree);
        # the lexicographic tie rule picks the shared start at angle 0.
        ang1 = np.angle(res.point1)
        ang2 = np.angle(res.point2)
        self.assertAlmostEqual(ang1, ang2, places=6)
        self.assertAlmostEqual(abs(res.point1), 3.0, places=6)
        self.assertAlmostEqual(abs(res.point2), 2.0, places=6)
        d_ref, _, _ = brute_force_seg_distance(outer, inner, n=300)
        self.assertAlmostEqual(res.dist, d_ref, places=5)

    def test_tangent_arcs_zero_within_tolerance(self):
        # Two circles of radius 1 with centers at (-1, 0) and (1, 0).
        # The two facing semicircles meet at their midpoints, the
        # external tangent point at the origin.
        left = Arc(-1 - 1j, 1 + 1j, 0, False, True, -1 + 1j)
        right = Arc(1 + 1j, 1 + 1j, 0, False, True, 1 - 1j)
        self.assertAlmostEqual(left.point(0.5), 0.0, places=10)
        self.assertAlmostEqual(right.point(0.5), 0.0, places=10)
        res = segment_distance(left, right)
        self.assertEqual(res.dist, 0.0)
        self.assertAlmostEqual(res.point1, 0.0, places=7)
        self.assertAlmostEqual(res.point2, 0.0, places=7)
        self.assertAlmostEqual(res.t1, 0.5, places=6)
        self.assertAlmostEqual(res.t2, 0.5, places=6)

    def test_arc_line_closest_point(self):
        # Quarter circle radius 2 in the first quadrant; horizontal line
        # y = 3 above it.  Closest point is the top of the circle (0, 2).
        arc = make_arc(0, 0, 2, 0, 90)
        line = Line(-3 + 3j, 3 + 3j)
        res = segment_distance(arc, line)
        self.assertAlmostEqual(res.dist, 1.0, places=6)
        self.assertAlmostEqual(res.point1, 2j, places=5)
        self.assertAlmostEqual(res.t1, 1.0, places=6)

    def test_arc_intersects_line(self):
        arc = make_arc(0, 0, 2, 0, 90)
        line = Line(-3 + 1j, 3 + 1j)
        res = segment_distance(arc, line)
        self.assertEqual(res.dist, 0.0)
        self.assertAlmostEqual(abs(res.point1 - res.point2), 0.0, places=8)
        self.assertAlmostEqual(res.point1.imag, 1.0, places=6)

    def test_elliptic_arc_matches_brute_force(self):
        arc = make_arc(1, -1, 3, 20, 200, rx=3, ry=1.5)
        c = CubicBezier(4 - 3j, 5 + 2j, 6 - 4j, 7 + 1j)
        res = segment_distance(arc, c)
        d_ref, _, _ = brute_force_seg_distance(arc, c, n=350)
        self.assertAlmostEqual(res.dist, d_ref, places=4)
        self.assertAlmostEqual(
            abs(arc.point(res.t1) - c.point(res.t2)), res.dist, places=10)

    # -- Degenerate segments ------------------------------------------------

    def test_zero_length_segments(self):
        point_seg = Line(2 + 2j, 2 + 2j)
        line = Line(0, 4)
        res = segment_distance(point_seg, line)
        # Distance from (2, 2) to the real axis is 2.
        self.assertAlmostEqual(res.dist, 2.0, places=10)
        self.assertEqual(res.t1, 0.0)
        self.assertAlmostEqual(res.point1, 2 + 2j, places=10)
        self.assertAlmostEqual(res.point2, 2.0, places=8)

        two_points = segment_distance(
            Line(1 + 1j, 1 + 1j), Line(4 + 5j, 4 + 5j))
        self.assertAlmostEqual(two_points.dist, 5.0, places=10)

        # A zero-length segment sitting on another curve is touching it.
        on_curve = Line(1.0, 1.0)
        self.assertEqual(segment_distance(on_curve, line).dist, 0.0)

    def test_fully_degenerate_bezier_segments(self):
        # All control points identical: the whole Bezier collapses to a
        # point and must be handled without invoking its derivative.
        point_quad = QuadraticBezier(3 + 4j, 3 + 4j, 3 + 4j)
        through = Line(0, 6 + 8j)
        r = segment_distance(point_quad, through)
        self.assertEqual(r.dist, 0.0)
        self.assertEqual(r.t1, 0.0)
        self.assertAlmostEqual(r.t2, 0.5, places=8)

        point_cubic = CubicBezier(*(1 + 1j for _ in range(4)))
        gap = segment_distance(point_cubic, Line(2j, 2 + 2j))
        self.assertAlmostEqual(gap.dist, 1.0, places=10)

    def test_strict_zero_tolerance_never_fabricates_contact(self):
        # A genuine 1e-11 gap must stay positive when tol is exactly zero.
        q = QuadraticBezier(0 + 1j, 1 - 1j, 2 + 1j)
        line = Line(0 - 1e-11j, 2 - 1e-11j)
        res = segment_distance(q, line, tol=0.0)
        self.assertAlmostEqual(res.dist, 1e-11, places=11)

    # -- Tolerance validation and near-tangent handling --------------------

    def test_invalid_tolerance(self):
        l1 = Line(0, 1)
        l2 = Line(2j, 1 + 2j)
        for bad in (-1e-9, -1.0, float('nan'), float('inf'),
                    '0.001', None, True):
            with self.assertRaises((ValueError, TypeError)):
                segment_distance(l1, l2, tol=bad)
            with self.assertRaises((ValueError, TypeError)):
                path_distance(Path(l1), Path(l2), tol=bad)

    def test_near_tangent_tolerance_behavior(self):
        # Parabola vertex sits 1e-10 below the line: geometrically a
        # positive gap, but inside the default tolerance it is "touching".
        gap = 1e-10
        q = QuadraticBezier(0 + 1j, 1 - 1j, 2 + 1j)
        line = Line(0 - gap * 1j, 2 - gap * 1j)
        default_res = segment_distance(q, line)
        self.assertEqual(default_res.dist, 0.0)
        strict_res = segment_distance(q, line, tol=1e-12)
        self.assertAlmostEqual(strict_res.dist, gap, places=11)

    def test_inputs_are_not_modified(self):
        c1 = CubicBezier(0, 1 + 2j, 3 - 2j, 4)
        c2 = c1.translated(2j)
        snap1 = (c1.bpoints(), c2.bpoints())
        segment_distance(c1, c2)
        segment_distance(c1, c2, tol=1e-12)
        self.assertEqual(c1.bpoints(), snap1[0])
        self.assertEqual(c2.bpoints(), snap1[1])

    def test_deterministic_tie_breaking(self):
        # A symmetric setup admits two equidistant minima; repeated calls
        # must return bit-identical parameters.
        q = QuadraticBezier(-1 + 1j, 0 + 0j, 1 + 1j)
        line = Line(-1 - 1j, 1 - 1j)
        results = [segment_distance(q, line) for _ in range(5)]
        for r in results[1:]:
            self.assertEqual(r.t1, results[0].t1)
            self.assertEqual(r.t2, results[0].t2)
            self.assertEqual(r.dist, results[0].dist)

    # -- Fuzz cross-check against an independent optimizer -----------------

    def test_random_segments_agree_with_scipy(self):
        rng = np.random.RandomState(42)

        def random_seg():
            kind = rng.randint(0, 4)
            pts = [complex(rng.uniform(-3, 3), rng.uniform(-3, 3))
                   for _ in range(4)]
            if kind == 0:
                return Line(pts[0], pts[1])
            if kind == 1:
                return QuadraticBezier(pts[0], pts[1], pts[2])
            if kind == 2:
                return CubicBezier(*pts)
            r = complex(abs(rng.uniform(0.5, 2.0)),
                        abs(rng.uniform(0.5, 2.0)))
            return Arc(pts[0], r, rng.uniform(-45, 45),
                       rng.randint(0, 2), rng.randint(0, 2), pts[1])

        for _ in range(12):
            s1, s2 = random_seg(), random_seg()
            if _is_point_degenerate(s1) or _is_point_degenerate(s2):
                continue
            # A strict tolerance keeps reported-zero cases genuine.
            res = segment_distance(s1, s2, tol=1e-12)
            d_ref, _, _ = scipy_seg_distance(s1, s2)
            self.assertAlmostEqual(res.dist, d_ref, delta=2e-7)
            # Returned parameters must actually reproduce the distance.
            residual = abs(s1.point(res.t1) - s2.point(res.t2))
            if res.dist == 0.0:
                self.assertLessEqual(residual, 1e-12)
            else:
                self.assertAlmostEqual(residual, res.dist, places=9)
            self.assertTrue(0.0 <= res.t1 <= 1.0)
            self.assertTrue(0.0 <= res.t2 <= 1.0)


def _is_point_degenerate(seg):
    if isinstance(seg, Line):
        return seg.start == seg.end
    if isinstance(seg, (QuadraticBezier, CubicBezier)):
        return all(bp == seg.start for bp in seg.bpoints())
    return False


class TestPathDistance(unittest.TestCase):

    def _rectangle(self, x0, y0, x1, y1):
        return Path(Line(complex(x0, y0), complex(x1, y0)),
                    Line(complex(x1, y0), complex(x1, y1)),
                    Line(complex(x1, y1), complex(x0, y1)),
                    Line(complex(x0, y1), complex(x0, y0)))

    def test_returns_namedtuple_with_all_fields(self):
        p1 = self._rectangle(0, 0, 1, 1)
        p2 = self._rectangle(3, 0, 4, 1)
        res = path_distance(p1, p2)
        self.assertIsInstance(res, PathDistance)
        self.assertEqual(res._fields,
                         ('dist', 'point1', 'point2', 'T1', 'T2',
                          'seg1_idx', 'seg2_idx', 't1', 't2'))

    def test_parallel_rectangles_known_gap(self):
        p1 = self._rectangle(0, 0, 1, 1)
        p2 = self._rectangle(3, 0, 4, 1)
        res = path_distance(p1, p2)
        self.assertAlmostEqual(res.dist, 2.0, places=8)
        # Several facing-side pairs are exactly equidistant; the
        # documented lexicographic tie rule deterministically picks the
        # earliest path parameter pair: the bottom-right corner of the
        # first box facing the bottom-left corner of the second.
        self.assertEqual((res.seg1_idx, res.seg2_idx), (0, 0))
        self.assertAlmostEqual(res.point1, 1.0, places=8)
        self.assertAlmostEqual(res.point2, 3.0, places=8)
        # Repeated calls return bit-identical results.
        self.assertEqual(tuple(path_distance(p1, p2)), tuple(res))
        # Path parameters are arc-length fractions in [0, 1] and locate
        # the same points as the local segment parameters.
        for T, p, seg_idx, t in ((res.T1, p1, res.seg1_idx, res.t1),
                                 (res.T2, p2, res.seg2_idx, res.t2)):
            self.assertTrue(0.0 <= T <= 1.0)
            self.assertAlmostEqual(p.point(T), p[seg_idx].point(t),
                                   places=8)

    def test_intersecting_paths_zero(self):
        p1 = self._rectangle(0, 0, 2, 2)
        p2 = self._rectangle(1, 1, 3, 3)
        res = path_distance(p1, p2)
        self.assertEqual(res.dist, 0.0)
        self.assertAlmostEqual(abs(res.point1 - res.point2), 0.0, places=8)

    def test_mixed_segment_path(self):
        path = Path(Line(0, 2),
                    CubicBezier(2, 2 + 2j, 4 - 2j, 4),
                    make_arc(6, 0, 2, 180, 0))
        line = Line(0 + 5j, 8 + 5j)
        res = path_distance(path, Path(line))
        d_ref = min(segment_distance(seg, line).dist for seg in path)
        self.assertAlmostEqual(res.dist, d_ref, places=9)
        self.assertAlmostEqual(
            abs(path[res.seg1_idx].point(res.t1)
                - line.point(res.t2)), res.dist, places=10)

    def test_empty_paths_raise(self):
        with self.assertRaises(ValueError):
            path_distance(Path(), Path(Line(0, 1)))
        with self.assertRaises(ValueError):
            path_distance(Path(Line(0, 1)), Path())

    def test_path_with_zero_length_segment(self):
        p1 = Path(Line(0, 2), Line(2, 2), Line(2, 5))
        p2 = Path(Line(5 + 1j, 5 - 1j))
        res = path_distance(p1, p2)
        # Horizontal distance from x=5 endpoint to the line x=5 is zero;
        # the closest point is the joint at (5, 0).
        self.assertAlmostEqual(res.dist, 0.0, places=8)
        self.assertAlmostEqual(res.point1, 5.0, places=7)

    def test_coincident_paths(self):
        p = self._rectangle(0, 0, 2, 3)
        res1 = path_distance(p, p)
        res2 = path_distance(p, p)
        self.assertEqual(res1.dist, 0.0)
        self.assertEqual(res1.T1, 0.0)
        self.assertEqual(res1.T2, 0.0)
        self.assertEqual(tuple(res1), tuple(res2))

    def test_path_not_mutated(self):
        p1 = self._rectangle(0, 0, 1, 1)
        p2 = self._rectangle(3, 0, 4, 1)
        before = [(type(s).__name__, s.bpoints()
                   if hasattr(s, 'bpoints') else s.apoints()) for s in p1]
        path_distance(p1, p2)
        after = [(type(s).__name__, s.bpoints()
                  if hasattr(s, 'bpoints') else s.apoints()) for s in p1]
        self.assertEqual(before, after)
        self.assertEqual(len(p1), 4)
        self.assertEqual(len(p2), 4)


class TestDispatchAndMethods(unittest.TestCase):

    def test_dispatch_returns_right_type(self):
        seg_res = distance(Line(0, 1), Line(2j, 1 + 2j))
        self.assertIsInstance(seg_res, SegmentDistance)
        path_res = distance(Path(Line(0, 1)), Path(Line(2j, 1 + 2j)))
        self.assertIsInstance(path_res, PathDistance)
        mixed = distance(Line(0, 1), Path(Line(2j, 1 + 2j)))
        self.assertIsInstance(mixed, PathDistance)

    def test_segment_and_path_methods(self):
        l1 = Line(0, 1)
        l2 = Line(2j, 1 + 2j)
        self.assertAlmostEqual(l1.distance(l2).dist,
                               segment_distance(l1, l2).dist, places=12)
        p = Path(l1)
        self.assertAlmostEqual(p.distance(l2).dist,
                               path_distance(p, Path(l2)).dist, places=12)

    def test_default_tolerance_constant(self):
        self.assertIsInstance(DEFAULT_DISTANCE_TOL, float)
        self.assertGreater(DEFAULT_DISTANCE_TOL, 0.0)

    def test_wrong_types_raise(self):
        with self.assertRaises(TypeError):
            segment_distance(Line(0, 1), Path(Line(0, 1)))
        with self.assertRaises(TypeError):
            segment_distance(Line(0, 1), 5)
        with self.assertRaises(TypeError):
            path_distance(Line(0, 1), Line(2, 3))


if __name__ == '__main__':
    unittest.main()
