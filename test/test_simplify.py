# External dependencies
from __future__ import division, absolute_import, print_function
import math
import unittest

# Internal dependencies
from svgpathtools import (Path, Line, CubicBezier, QuadraticBezier,
                          simplify_path, max_deviation, SimplificationError)
from svgpathtools.simplify import _self_intersection_count


def hand_drawn_path(n=1500):
    """A dense polyline of nearly-collinear short segments, mimicking a
    hand-drawn stroke (slowly varying curve plus deterministic jitter)."""
    pts = []
    for k in range(n + 1):
        x = k * 0.5
        y = 40.0 * math.sin(x / 60.0) + 0.3 * math.sin(x * 2.7)
        pts.append(complex(x, y))
    return Path(*[Line(pts[i], pts[i + 1]) for i in range(n)])


def closed_ngon_path(n=360, radius=50.0, center=100 + 100j):
    """A closed regular n-gon (many nearly-collinear segments)."""
    pts = [center + radius * complex(math.cos(2 * math.pi * k / n),
                                     math.sin(2 * math.pi * k / n))
           for k in range(n)]
    return Path(*[Line(pts[i], pts[(i + 1) % n]) for i in range(n)])


def star_path(radius=50.0, center=0j):
    """A self-intersecting closed {5/2} star polygon (5 segments)."""
    pts = [center + radius * complex(math.cos(math.pi / 2 + 4 * math.pi * k / 5),
                                     math.sin(math.pi / 2 + 4 * math.pi * k / 5))
           for k in range(5)]
    return Path(*[Line(pts[i], pts[(i + 1) % 5]) for i in range(5)])


class TestSimplifyDensePolyline(unittest.TestCase):

    def test_segment_count_and_deviation(self):
        path = hand_drawn_path()
        tol = 1.0
        result = simplify_path(path, tol)
        # massive reduction of the redundant short segments
        self.assertLess(len(result), len(path) // 5)
        self.assertGreater(len(result), 1)
        # verified maximum geometric deviation within tolerance
        dev = max_deviation(path, result, sample_tol=tol * 0.05)
        self.assertLessEqual(dev, tol)

    def test_endpoints_preserved(self):
        path = hand_drawn_path()
        result = simplify_path(path, 1.0)
        self.assertEqual(result[0].start, path[0].start)
        self.assertEqual(result[-1].end, path[-1].end)

    def test_input_not_modified(self):
        path = hand_drawn_path(200)
        d_before = path.d()
        simplify_path(path, 1.0)
        self.assertEqual(path.d(), d_before)

    def test_deterministic(self):
        path = hand_drawn_path(300)
        self.assertEqual(simplify_path(path, 1.0), simplify_path(path, 1.0))

    def test_straight_soup_merges_to_one_line(self):
        # many nearly collinear segments along one straight stroke
        pts = [complex(k * 0.7, 0.01 * math.sin(k)) for k in range(501)]
        path = Path(*[Line(pts[i], pts[i + 1]) for i in range(500)])
        result = simplify_path(path, 0.5)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], Line)
        self.assertEqual(result[0].start, path[0].start)
        self.assertEqual(result[0].end, path[-1].end)


class TestSimplifyCurves(unittest.TestCase):

    def test_split_cubic_merges_back(self):
        # a cubic split into three adjacent pieces should be merged
        base = CubicBezier(0 + 0j, 50 + 80j, 150 + 80j, 200 + 0j)
        a, bc = base.split(1.0 / 3.0)
        b, c = bc.split(0.5)
        path = Path(a, b, c)
        tol = 0.1
        result = simplify_path(path, tol)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], CubicBezier)
        dev = max_deviation(path, result, sample_tol=tol * 0.05)
        self.assertLessEqual(dev, tol)

    def test_curve_chain_deviation(self):
        # a chain of smooth cubics and quadratics
        segs = [
            CubicBezier(0 + 0j, 30 + 60j, 70 + 60j, 100 + 0j),
            CubicBezier(100 + 0j, 130 - 60j, 170 - 60j, 200 + 0j),
            QuadraticBezier(200 + 0j, 250 + 50j, 300 + 0j),
            CubicBezier(300 + 0j, 340 + 10j, 360 - 10j, 400 + 0j),
        ]
        path = Path(*segs)
        tol = 0.5
        result = simplify_path(path, tol)
        self.assertLess(len(result), 20)
        dev = max_deviation(path, result, sample_tol=tol * 0.05)
        self.assertLessEqual(dev, tol)
        self.assertEqual(result[0].start, path[0].start)
        self.assertEqual(result[-1].end, path[-1].end)

    def test_sharp_corners_preserved(self):
        # rectangle: all four corners are significant and must survive
        path = Path(Line(0 + 0j, 100 + 0j), Line(100 + 0j, 100 + 50j),
                    Line(100 + 50j, 0 + 50j), Line(0 + 50j, 0 + 0j))
        result = simplify_path(path, 2.0)
        corners = {0 + 0j, 100 + 0j, 100 + 50j, 0 + 50j}
        out_points = {seg.start for seg in result} | {seg.end for seg in result}
        self.assertTrue(corners <= out_points)
        self.assertEqual(len(result), 4)


class TestSimplifyClosedLoop(unittest.TestCase):

    def test_closed_ngon(self):
        path = closed_ngon_path()
        tol = 1.0
        result = simplify_path(path, tol)
        # topology: still one continuous, closed subpath
        subs = result.continuous_subpaths()
        self.assertEqual(len(subs), 1)
        self.assertTrue(subs[0].isclosed())
        # seam preserved
        self.assertEqual(result[0].start, path[0].start)
        # strong reduction and bounded deviation
        self.assertLess(len(result), len(path) // 4)
        dev = max_deviation(path, result, sample_tol=tol * 0.05)
        self.assertLessEqual(dev, tol)
        # no self-intersections introduced
        self.assertEqual(_self_intersection_count(result), 0)

    def test_closed_seam_corner_preserved(self):
        # closed triangle: seam is a significant corner
        path = Path(Line(0 + 0j, 100 + 0j), Line(100 + 0j, 50 + 80j),
                    Line(50 + 80j, 0 + 0j))
        result = simplify_path(path, 1.0)
        self.assertEqual(len(result), 3)
        self.assertEqual(result[0].start, 0 + 0j)
        self.assertTrue(result.isclosed())


class TestSimplifyTopology(unittest.TestCase):

    def test_self_intersecting_star_preserved(self):
        path = star_path()
        self.assertEqual(_self_intersection_count(path), 5)
        # even with a tolerance far too large for the star, the topology
        # must be preserved (the working tolerance is reduced internally)
        result = simplify_path(path, 40.0, preserve_topology=True)
        self.assertEqual(_self_intersection_count(result), 5)
        self.assertTrue(result.isclosed())
        dev = max_deviation(path, result, sample_tol=0.05)
        self.assertLessEqual(dev, 40.0)

    def test_topology_change_raises(self):
        # a tiny self-intersecting loop: no allowed simplification can
        # keep the intersection, so an explicit error must be raised
        pts = [0 + 0j, 2 + 0j, 2 + 1j, 1 - 1j, 3 + 0j]
        path = Path(*[Line(pts[i], pts[i + 1]) for i in range(4)])
        self.assertEqual(_self_intersection_count(path), 1)
        with self.assertRaises(SimplificationError):
            simplify_path(path, 100.0, corner_angle=math.pi,
                          preserve_topology=True)
        # ... while without the topology guard it simplifies fine
        result = simplify_path(path, 100.0, corner_angle=math.pi)
        self.assertEqual(len(result), 1)


class TestSimplifyEdgeCases(unittest.TestCase):

    def test_invalid_tolerance(self):
        path = Path(Line(0, 1 + 1j))
        for bad in (-1.0, -0.5, float('nan'), float('inf'), float('-inf')):
            with self.assertRaises(ValueError):
                simplify_path(path, bad)

    def test_zero_tolerance_returns_copy(self):
        path = Path(Line(0, 1 + 1j), CubicBezier(1 + 1j, 2 + 2j, 3, 4 - 1j))
        result = simplify_path(path, 0.0)
        self.assertEqual(result, path)
        self.assertIsNot(result, path)

    def test_empty_path(self):
        result = simplify_path(Path(), 1.0)
        self.assertIsInstance(result, Path)
        self.assertEqual(len(result), 0)

    def test_wrong_type(self):
        with self.assertRaises(TypeError):
            simplify_path("M 0 0 L 1 1", 1.0)

    def test_subpath_boundaries_preserved(self):
        # two disjoint continuous subpaths
        path = Path(Line(0 + 0j, 10 + 0j), Line(10 + 0j, 10 + 10j),
                    Line(50 + 50j, 60 + 50j))
        result = simplify_path(path, 1.0)
        subs = result.continuous_subpaths()
        self.assertEqual(len(subs), 2)
        self.assertEqual(subs[0][0].start, 0 + 0j)
        self.assertEqual(subs[0][-1].end, 10 + 10j)
        self.assertEqual(subs[1][0].start, 50 + 50j)
        self.assertEqual(subs[1][-1].end, 60 + 50j)

    def test_max_deviation_empty(self):
        self.assertEqual(max_deviation(Path(), Path()), 0.0)
        nonempty = Path(Line(0, 1))
        self.assertEqual(max_deviation(Path(), nonempty), float('inf'))


if __name__ == '__main__':
    unittest.main()
