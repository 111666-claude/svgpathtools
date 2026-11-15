"""Tests for Path.clipped (rectangular viewport clipping of stroked paths)."""
from __future__ import absolute_import
import unittest

from svgpathtools import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                          parse_path)

# viewport used by most tests: [0, 100] x [0, 100]
XMIN, XMAX, YMIN, YMAX = 0.0, 100.0, 0.0, 100.0


def clip(path, tol=1e-8):
    return path.clipped(XMIN, XMAX, YMIN, YMAX, tol=tol)


def assert_no_points_outside(testcase, fragments, rect=None, tol=1e-6):
    """Sample every fragment densely; all points must lie in the viewport."""
    if rect is None:
        rect = (XMIN, XMAX, YMIN, YMAX)
    xmin, xmax, ymin, ymax = rect
    for frag in fragments:
        for seg in frag:
            for k in range(41):
                z = seg.point(k/40.0)
                testcase.assertGreaterEqual(z.real, xmin - tol)
                testcase.assertLessEqual(z.real, xmax + tol)
                testcase.assertGreaterEqual(z.imag, ymin - tol)
                testcase.assertLessEqual(z.imag, ymax + tol)


class TestLineClipping(unittest.TestCase):
    def test_line_fully_inside(self):
        path = Path(Line(10 + 10j, 90 + 90j))
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        # untouched segment is reused as-is
        self.assertIs(frags[0][0], path[0])

    def test_line_fully_outside(self):
        path = Path(Line(200 + 200j, 300 + 300j))
        self.assertEqual(clip(path), [])

    def test_line_crossing(self):
        path = Path(Line(-50 + 50j, 150 + 50j))
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        self.assertEqual(len(frags[0]), 1)
        seg = frags[0][0]
        self.assertIsInstance(seg, Line)
        self.assertAlmostEqual(seg.start.real, 0.0)
        self.assertAlmostEqual(seg.end.real, 100.0)
        self.assertAlmostEqual(seg.start.imag, 50.0)

    def test_line_through_corner(self):
        # passes exactly through the corner (0, 0)
        path = Path(Line(-50 - 50j, 50 + 50j))
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        self.assertAlmostEqual(abs(frags[0][0].start - 0j), 0.0, places=6)
        self.assertAlmostEqual(abs(frags[0][0].end - (50 + 50j)), 0.0,
                               places=6)

    def test_line_along_edge_is_kept(self):
        path = Path(Line(0 + 10j, 0 + 90j))
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        self.assertIs(frags[0][0], path[0])

    def test_line_along_edge_outside_extent(self):
        # collinear with the edge x=0 but beyond its y-extent
        path = Path(Line(0 + 200j, 0 + 300j))
        self.assertEqual(clip(path), [])

    def test_multiple_entries_stable_order(self):
        # zig-zag entering and leaving the viewport three times
        path = Path(
            Line(-10 + 10j, 50 + 10j),     # in
            Line(50 + 10j, 50 + 200j),     # out (top)
            Line(50 + 200j, 60 + 50j),     # in again
            Line(60 + 50j, 200 + 50j),     # out (right)
            Line(200 + 50j, 70 + 60j),     # in again
        )
        frags = clip(path)
        self.assertEqual(len(frags), 3)
        # fragments appear in path order
        self.assertAlmostEqual(frags[0].start.imag, 10.0)
        # second fragment enters through the top edge y=100 at x=56.666...
        self.assertAlmostEqual(frags[1].start.real, 50 + 100/15.0, places=6)
        self.assertAlmostEqual(frags[1].start.imag, 100.0, places=6)
        self.assertAlmostEqual(frags[2].end.real, 70.0)
        assert_no_points_outside(self, frags)


class TestBezierClipping(unittest.TestCase):
    def test_cubic_crossing_multiple_times(self):
        # S-shaped cubic weaving out through the top and bottom edges:
        # inside -> outside -> inside -> outside -> inside
        path = Path(CubicBezier(20 + 50j, 40 + 300j, 60 - 200j, 80 + 50j))
        frags = clip(path)
        self.assertEqual(len(frags), 3)
        for frag in frags:
            self.assertIsInstance(frag[0], CubicBezier)
        # stable order: x increases along the path
        self.assertLess(frags[0].start.real, frags[1].start.real)
        self.assertLess(frags[1].start.real, frags[2].start.real)
        assert_no_points_outside(self, frags)

    def test_cubic_fragment_matches_original_curve(self):
        # the clipped fragment must lie on the original cubic, not on a
        # polyline approximation of it
        cubic = CubicBezier(-20 + 50j, 20 + 20j, 80 + 80j, 120 + 50j)
        frags = clip(Path(cubic))
        self.assertEqual(len(frags), 1)
        frag = frags[0][0]
        self.assertIsInstance(frag, CubicBezier)
        # a point halfway along the fragment lies exactly on the original
        # curve (find its parameter by proximity)
        z = frag.point(0.5)
        best = min((abs(cubic.point(t) - z), t)
                   for t in [i/1000.0 for i in range(1001)])
        self.assertLess(best[0], 1e-6)

    def test_quadratic_tangent_outside(self):
        # parabola touching y=0 from below (outside) at one point
        path = Path(QuadraticBezier(20 - 60j, 50 + 0j, 80 - 60j))
        # control point on the edge pulls the curve below it; the curve
        # itself stays strictly outside except nothing at all
        frags = clip(path)
        # curve peaks at y = -30 + ... -- entirely below y=0
        self.assertEqual(frags, [])

    def test_quadratic_tangent_inside(self):
        # parabola inside the viewport, tangent to the edge y=100
        path = Path(QuadraticBezier(10 + 60j, 50 + 180j, 90 + 60j))
        # apex at t=0.5: y = 0.25*60 + 0.5*180 + 0.25*60 = 120 -> outside.
        # adjust: choose control so apex is exactly 100
        path = Path(QuadraticBezier(10 + 40j, 50 + 160j, 90 + 40j))
        # apex y = 0.25*40 + 0.5*160 + 0.25*40 = 100 -> tangent from inside
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        assert_no_points_outside(self, frags)

    def test_quadratic_enter_exit(self):
        path = Path(QuadraticBezier(-50 + 50j, 50 + 50j, 150 + 50j))
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        self.assertIsInstance(frags[0][0], QuadraticBezier)
        self.assertAlmostEqual(frags[0].start.real, 0.0, places=6)
        self.assertAlmostEqual(frags[0].end.real, 100.0, places=6)
        assert_no_points_outside(self, frags)


class TestArcClipping(unittest.TestCase):
    def test_arc_crossing(self):
        # quarter circle of radius 60 centered at (50, 50), from (-10, 50)
        # through (7.57, 7.57) to (50, -10): only the middle part near the
        # corner (0, 0) lies inside the viewport
        arc = Arc(-10 + 50j, radius=60 + 60j, rotation=0, large_arc=0,
                  sweep=1, end=50 - 10j)
        frags = clip(Path(arc))
        self.assertEqual(len(frags), 1)
        self.assertEqual(len(frags[0]), 1)
        frag = frags[0][0]
        self.assertIsInstance(frag, Arc)
        # geometry preserved: same circle, same sweep
        self.assertAlmostEqual(frag.radius.real, 60.0)
        self.assertAlmostEqual(frag.radius.imag, 60.0)
        self.assertEqual(frag.sweep, arc.sweep)
        # enters through x=0 at y = 50 - 10*sqrt(11)
        self.assertAlmostEqual(frag.start.real, 0.0, places=6)
        self.assertAlmostEqual(frag.start.imag, 16.833752096446005, places=6)
        self.assertAlmostEqual(frag.end.real, 16.833752096446005, places=6)
        self.assertAlmostEqual(frag.end.imag, 0.0, places=6)
        assert_no_points_outside(self, frags)

    def test_arc_tangent_to_edge(self):
        # half circle of radius 50 centered at (50, 50), passing through
        # (50, 0): entirely inside, tangent to the top edge y=0
        arc = Arc(0 + 50j, radius=50 + 50j, rotation=0, large_arc=0,
                  sweep=1, end=100 + 50j)
        frags = clip(Path(arc))
        self.assertEqual(len(frags), 1)
        self.assertTrue(frags[0].iscontinuous())
        self.assertAlmostEqual(abs(frags[0].start - (0 + 50j)), 0.0, places=6)
        self.assertAlmostEqual(abs(frags[0].end - (100 + 50j)), 0.0, places=6)
        for seg in frags[0]:
            self.assertIsInstance(seg, Arc)
        assert_no_points_outside(self, frags)

    def test_rotated_arc_multiple_fragments(self):
        # rotated ellipse (center (50, 50)) sticking out through the left
        # and right edges: two separate inside fragments
        from math import cos, sin, radians
        phi = radians(20.0)

        def ellipse_point(a_deg):
            a = radians(a_deg)
            x = 50 + 90*cos(phi)*cos(a) - 30*sin(phi)*sin(a)
            y = 50 + 90*sin(phi)*cos(a) + 30*cos(phi)*sin(a)
            return x + 1j*y

        arc = Arc(ellipse_point(0), radius=90 + 30j, rotation=20,
                  large_arc=1, sweep=1, end=ellipse_point(350))
        frags = clip(Path(arc))
        self.assertEqual(len(frags), 2)
        for frag in frags:
            for seg in frag:
                self.assertIsInstance(seg, Arc)
            # fragment endpoints lie on the viewport boundary
            for z in (frag.start, frag.end):
                on_boundary = (abs(z.real - XMIN) < 1e-6 or
                               abs(z.real - XMAX) < 1e-6 or
                               abs(z.imag - YMIN) < 1e-6 or
                               abs(z.imag - YMAX) < 1e-6)
                self.assertTrue(on_boundary)
        assert_no_points_outside(self, frags)

    def test_arc_fully_inside(self):
        arc = Arc(10 + 50j, radius=20 + 20j, rotation=0, large_arc=0,
                  sweep=1, end=50 + 50j)
        path = Path(arc)
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        self.assertIs(frags[0][0], arc)


class TestPathLevelBehavior(unittest.TestCase):
    def test_empty_path(self):
        self.assertEqual(clip(Path()), [])

    def test_empty_result(self):
        path = Path(Line(150 + 150j, 200 + 10j),
                    CubicBezier(200 + 10j, 300 + 300j, -300 + 300j,
                                150 + 150j))
        self.assertEqual(clip(path), [])

    def test_invalid_viewport_raises(self):
        path = Path(Line(0j, 1 + 1j))
        with self.assertRaises(ValueError):
            path.clipped(100, 0, 0, 100)
        with self.assertRaises(ValueError):
            path.clipped(0, 100, 50, 40)

    def test_degenerate_viewport_zero_width(self):
        # viewport collapsed onto the segment x in [0,100] at x == 50
        path = Path(Line(50 + 10j, 50 + 90j))
        frags = path.clipped(50, 50, 0, 100)
        self.assertEqual(len(frags), 1)
        self.assertIs(frags[0][0], path[0])
        # a crossing line leaves only a point -> no fragment
        path2 = Path(Line(0 + 50j, 100 + 50j))
        frags2 = path2.clipped(50, 50, 0, 100)
        for frag in frags2:
            for seg in frag:
                self.assertAlmostEqual(abs(seg.end - seg.start), 0.0,
                                       places=6)

    def test_closed_path_wraparound_merge(self):
        # closed square outline [-25,125]^2 clipped to [0,100]^2:
        # the four corner arcs of the outline are outside; the result is
        # four edge fragments -- but a closed path fully inside stays whole
        inner = Path(Line(10 + 10j, 90 + 10j), Line(90 + 10j, 90 + 90j),
                     Line(90 + 90j, 10 + 90j), Line(10 + 90j, 10 + 10j))
        frags = clip(inner)
        self.assertEqual(len(frags), 1)
        self.assertEqual(len(frags[0]), 4)

        # closed path whose start point is outside: the run wrapping the
        # closure must be merged into one fragment
        path = Path(Line(150 + 50j, 50 + 50j),   # starts outside, enters
                    Line(50 + 50j, 50 + 150j),   # leaves
                    Line(50 + 150j, 150 + 50j),  # outside
                    Line(150 + 50j, 150 + 50j))  # closes outside
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        self.assertEqual(len(frags[0]), 2)
        assert_no_points_outside(self, frags)

    def test_zero_length_segment_inside(self):
        path = Path(Line(50 + 50j, 50 + 50j))
        frags = clip(path)
        self.assertEqual(len(frags), 1)

    def test_exact_continuity_at_boundary_junction(self):
        # the junction between the two segments lies exactly on the
        # viewport boundary; the clipped fragment must be exactly
        # continuous (no floating-point gap from recomputed endpoints)
        path = Path(Arc(20 + 0j, radius=30 + 30j, rotation=0, large_arc=0,
                        sweep=1, end=50 + 30j),
                    CubicBezier(50 + 30j, 70 + 10j, 80 + 90j, 60 + 110j))
        frags = clip(path)
        for frag in frags:
            self.assertTrue(frag.iscontinuous())
            for i in range(len(frag) - 1):
                self.assertEqual(frag[i].end, frag[i + 1].start)

    def test_exact_continuity_after_corner_clip(self):
        # curve passing very close to the corner (0, 100): the near-
        # duplicate crossings must not leave a gap inside the fragment
        path = Path(CubicBezier(-5 + 110j, 30 + 99.99999j, 60 + 100.00001j,
                                110 + 90j))
        frags = clip(path)
        self.assertEqual(len(frags), 1)
        self.assertTrue(frags[0].iscontinuous())
        for i in range(len(frags[0]) - 1):
            self.assertEqual(frags[0][i].end, frags[0][i + 1].start)
        assert_no_points_outside(self, frags)

    def test_zero_length_segment_outside(self):
        path = Path(Line(150 + 50j, 150 + 50j))
        self.assertEqual(clip(path), [])

    def test_continuity_of_fragments(self):
        path = Path(CubicBezier(-10 + 20j, 30 + 10j, 70 + 30j, 110 + 20j),
                    Line(110 + 20j, 50 + 60j),
                        QuadraticBezier(50 + 60j, 20 + 90j, 50 + 120j))
        frags = clip(path)
        for frag in frags:
            self.assertTrue(frag.iscontinuous())
        assert_no_points_outside(self, frags)


class TestSerializationRoundTrip(unittest.TestCase):
    def test_d_roundtrip(self):
        path = Path(
            CubicBezier(-20 + 50j, 20 + 20j, 80 + 80j, 120 + 50j),
            Arc(120 + 50j, radius=30 + 50j, rotation=45, large_arc=1,
                sweep=0, end=50 + 120j),
            QuadraticBezier(50 + 120j, 20 + 60j, -10 + 30j),
        )
        frags = clip(path)
        self.assertGreaterEqual(len(frags), 1)
        for frag in frags:
            d = frag.d()
            back = parse_path(d)
            self.assertEqual(len(back), len(frag))
            for seg_in, seg_out in zip(frag, back):
                self.assertIs(type(seg_in), type(seg_out))
                for t in (0.0, 0.25, 0.5, 0.75, 1.0):
                    self.assertAlmostEqual(abs(seg_in.point(t)
                                               - seg_out.point(t)), 0.0,
                                           places=6)

    def test_original_path_unmodified(self):
        path = Path(Line(-50 + 50j, 150 + 50j),
                    CubicBezier(150 + 50j, 200 + 0j, 200 + 100j, 150 + 60j))
        d_before = path.d()
        clip(path)
        self.assertEqual(path.d(), d_before)
        self.assertEqual(len(path), 2)


if __name__ == '__main__':
    unittest.main()
