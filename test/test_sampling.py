"""Tests for svgpathtools.sampling (error-controlled adaptive sampling)."""

from __future__ import division, absolute_import
import math
import unittest

from svgpathtools import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                          sample_path, sample_segment,
                          MaxPointsExceededError)

NAN = complex(float('nan'), float('nan'))


def point_to_segment_distance(p, a, b):
    """Distance from p to the line segment ab (test helper)."""
    chord = b - a
    L2 = abs(chord)**2
    if L2 == 0:
        return abs(p - a)
    u = ((p.real - a.real) * chord.real +
         (p.imag - a.imag) * chord.imag) / L2
    u = min(1.0, max(0.0, u))
    return abs(p - (a + u * chord))


def curve_to_polyline_error(seg, points, n=4000):
    """Independently measured max distance from the true curve to the
    sampled polyline, by dense sampling of the curve."""
    worst = 0.0
    for i in range(n + 1):
        p = seg.point(i / n)
        d = min(point_to_polyline_segment_distance(p, points))
        worst = max(worst, d)
    return worst


def point_to_polyline_segment_distance(p, points):
    for i in range(len(points) - 1):
        yield point_to_segment_distance(p, points[i], points[i + 1])


def is_unit(z, tol=1e-9):
    return abs(abs(z) - 1) <= tol


def is_nan(z):
    return math.isnan(z.real) and math.isnan(z.imag)


class TestToleranceValidation(unittest.TestCase):
    def test_nonpositive_and_nonfinite_tolerance(self):
        path = Path(Line(0, 1))
        for bad in (0, 0.0, -1e-3, -10, float('nan'), float('inf'),
                    float('-inf')):
            with self.assertRaises(ValueError):
                sample_path(path, bad)

    def test_nonnumeric_tolerance(self):
        path = Path(Line(0, 1))
        for bad in (None, 'abc', 1 + 2j, [0.1]):
            with self.assertRaises(ValueError):
                sample_path(path, bad)

    def test_bad_max_points(self):
        path = Path(Line(0, 1))
        for bad in (0, -5):
            with self.assertRaises(ValueError):
                sample_path(path, 0.1, max_points=bad)


class TestLineSampling(unittest.TestCase):
    def test_line_keeps_only_endpoints(self):
        # Even with a tiny tolerance, a straight line needs no
        # intermediate points.
        path = Path(Line(0, 100 + 50j), Line(100 + 50j, 100 + 150j))
        points, tangents = sample_path(path, 1e-9)
        self.assertEqual(points, [0j, 100 + 50j, 100 + 150j])
        for t, expected in zip(tangents[:1],
                               [(100 + 50j) / abs(100 + 50j)]):
            self.assertAlmostEqual(abs(t - expected), 0)
        self.assertTrue(all(is_unit(t) for t in tangents))

    def test_single_segment_input(self):
        points, tangents = sample_path(Line(0, 3 + 4j), 0.1)
        self.assertEqual(points, [0j, 3 + 4j])
        self.assertEqual(tangents, [0.6 + 0.8j, 0.6 + 0.8j])

    def test_sample_segment_line(self):
        points, tangents = sample_segment(Line(1 + 1j, 1 + 5j), 0.5)
        self.assertEqual(points, [1 + 1j, 1 + 5j])
        self.assertEqual(tangents, [1j, 1j])


class TestBezierSampling(unittest.TestCase):
    def setUp(self):
        # Gentle, almost straight cubic.
        self.gentle = CubicBezier(0, 30 + 1j, 70 - 1j, 100 + 0j)
        # Tight S-curve with high curvature.
        self.tight = CubicBezier(0, 0 + 200j, 100 - 200j, 100 + 0j)

    def test_error_bound_holds(self):
        tol = 0.5
        for seg in (self.gentle, self.tight):
            points, _ = sample_path(seg, tol)
            err = curve_to_polyline_error(seg, points)
            self.assertLessEqual(err, tol * (1 + 1e-9))

    def test_curvature_adapts_point_count(self):
        tol = 0.5
        gentle_points, _ = sample_path(self.gentle, tol)
        tight_points, _ = sample_path(self.tight, tol)
        # The tight curve needs substantially more points than the
        # gentle one, and the gentle one far fewer than fixed-step
        # sampling at the same resolution would use.
        self.assertGreater(len(tight_points), 3 * len(gentle_points))
        self.assertLess(len(gentle_points), 10)

    def test_finer_tolerance_gives_more_points(self):
        seg = self.tight
        counts = [len(sample_path(seg, tol)[0])
                  for tol in (2.0, 0.5, 0.1)]
        self.assertLess(counts[0], counts[1])
        self.assertLess(counts[1], counts[2])

    def test_tangents_are_unit_and_match_derivative(self):
        seg = self.tight
        points, tangents = sample_path(seg, 0.25)
        self.assertEqual(len(points), len(tangents))
        for t in tangents:
            self.assertTrue(is_unit(t), msg="tangent %r not unit" % t)
        # Spot-check the end tangents against the analytic direction.
        d0 = seg.derivative(0)
        d0 /= abs(d0)
        d1 = seg.derivative(1)
        d1 /= abs(d1)
        self.assertAlmostEqual(abs(tangents[0] - d0), 0)
        self.assertAlmostEqual(abs(tangents[-1] - d1), 0)

    def test_quadratic_bezier(self):
        seg = QuadraticBezier(0, 50 + 80j, 100 + 0j)
        tol = 0.2
        points, tangents = sample_path(seg, tol)
        self.assertGreater(len(points), 2)
        self.assertLessEqual(curve_to_polyline_error(seg, points), tol)
        self.assertTrue(all(is_unit(t) for t in tangents))


class TestArcSampling(unittest.TestCase):
    def test_quarter_circle(self):
        # Quarter circle of radius 100, centered at the origin.
        arc = Arc(100 + 0j, 100 + 100j, 0, False, True, 0 + 100j)
        self.assertAlmostEqual(abs(arc.center - 0j), 0)
        tol = 1.0
        points, tangents = sample_path(arc, tol)

        # Error guarantee, measured independently by dense sampling.
        self.assertLessEqual(curve_to_polyline_error(arc, points), tol)

        # Point count in the expected range: theory gives
        # n = ceil((pi/2) / (2*acos(1 - tol/r))) ~ 6 chords; bisection
        # rounds up to 8.  Generous bounds keep the test robust.
        self.assertGreaterEqual(len(points), 6)
        self.assertLessEqual(len(points), 12)

        # Tangents of a circle are perpendicular to the radius.
        for p, t in zip(points, tangents):
            self.assertTrue(is_unit(t))
            radial = p / abs(p)
            self.assertAlmostEqual((t / radial).real, 0, places=9)

    def test_elliptical_arc(self):
        arc = Arc(0, 50 + 20j, 30, True, True, 40 + 40j)
        tol = 0.25
        points, tangents = sample_path(arc, tol)
        self.assertLessEqual(curve_to_polyline_error(arc, points), tol)
        self.assertTrue(all(is_unit(t) for t in tangents))

    def test_arc_sagitta_matches_theory(self):
        # For a circular arc the accepted sub-arcs should each have a
        # sagitta (midpoint chord deviation) at most tol.
        r = 50.0
        arc = Arc(r + 0j, r + 1j * r, 0, False, True, 0 + 1j * r)
        tol = 0.1
        points, _ = sample_path(arc, tol)
        for a, b in zip(points[:-1], points[1:]):
            mid_angle = (math.atan2(a.imag, a.real) +
                         math.atan2(b.imag, b.real)) / 2
            mid = r * complex(math.cos(mid_angle), math.sin(mid_angle))
            sagitta = point_to_segment_distance(mid, a, b)
            self.assertLessEqual(sagitta, tol * (1 + 1e-9))


class TestSeamsAndClosure(unittest.TestCase):
    def test_no_duplicate_points_at_seams(self):
        seam = 50 + 20j
        path = Path(CubicBezier(0, 10 + 40j, 40 + 40j, seam),
                    CubicBezier(seam, 60 - 10j, 90 - 10j, 100 + 0j))
        points, _ = sample_path(path, 0.25)
        self.assertEqual(points.count(seam), 1)
        # No coincident consecutive points anywhere.
        for a, b in zip(points[:-1], points[1:]):
            self.assertNotEqual(a, b)

    def test_discontinuous_jump_is_preserved(self):
        path = Path(Line(0, 10 + 0j), Line(20 + 0j, 30 + 0j))
        points, _ = sample_path(path, 0.1)
        self.assertEqual(points, [0j, 10 + 0j, 20 + 0j, 30 + 0j])

    def test_closed_path_first_equals_last(self):
        # Closed triangle of straight lines.
        tri = Path(Line(0, 100 + 0j), Line(100 + 0j, 50 + 80j),
                   Line(50 + 80j, 0))
        self.assertTrue(tri.isclosed())
        points, _ = sample_path(tri, 0.1)
        self.assertEqual(points[0], points[-1])
        self.assertEqual(len(points), 4)

        # Closed curved path.
        curve = Path(CubicBezier(0, 30 + 50j, 70 + 50j, 100 + 0j),
                     CubicBezier(100 + 0j, 70 - 50j, 30 - 50j, 0))
        self.assertTrue(curve.isclosed())
        points, _ = sample_path(curve, 0.5)
        self.assertEqual(points[0], points[-1])


class TestDegenerateInput(unittest.TestCase):
    def test_zero_length_segment_only(self):
        points, tangents = sample_path(Path(Line(5 + 5j, 5 + 5j)), 0.1)
        self.assertEqual(points, [5 + 5j])
        self.assertEqual(len(tangents), 1)
        self.assertTrue(is_nan(tangents[0]))

    def test_zero_length_segments_are_skipped(self):
        path = Path(Line(0, 0), Line(0, 10 + 0j), Line(10 + 0j, 10 + 0j),
                    CubicBezier(10 + 0j, 10 + 0j, 10 + 0j, 10 + 0j))
        points, tangents = sample_path(path, 0.1)
        self.assertEqual(points, [0j, 10 + 0j])
        self.assertTrue(all(is_unit(t) for t in tangents))

    def test_empty_path(self):
        points, tangents = sample_path(Path(), 0.1)
        self.assertEqual(points, [])
        self.assertEqual(tangents, [])

    def test_cusp_is_sampled_and_tangent_handled(self):
        # Cubic with a cusp at t = 0.5 (derivative vanishes there).
        cusp = CubicBezier(0, 1 + 1j, 1 + 1j, 0)
        tol = 0.01
        points, tangents = sample_path(cusp, tol)
        # Sampling succeeds and respects the error bound.
        self.assertLessEqual(curve_to_polyline_error(cusp, points), tol)
        # Every tangent is either a unit vector or explicitly NaN.
        for t in tangents:
            self.assertTrue(is_unit(t) or is_nan(t),
                            msg="unexpected tangent %r" % t)


class TestMaxPoints(unittest.TestCase):
    def test_max_points_exceeded(self):
        tight = CubicBezier(0, 0 + 200j, 100 - 200j, 100 + 0j)
        with self.assertRaises(MaxPointsExceededError):
            sample_path(tight, 1e-4, max_points=50)

    def test_max_points_sufficient(self):
        tight = CubicBezier(0, 0 + 200j, 100 - 200j, 100 + 0j)
        points, _ = sample_path(tight, 1e-4, max_points=100000)
        self.assertLessEqual(len(points), 100000)


class TestPathOfMixedSegments(unittest.TestCase):
    def test_mixed_path_end_to_end(self):
        arc = Arc(100 + 0j, 50 + 50j, 0, False, True, 100 + 100j)
        path = Path(Line(0, 100 + 0j), arc,
                    CubicBezier(100 + 100j, 50 + 120j, 20 + 120j, 0 + 100j),
                    Line(0 + 100j, 0))
        self.assertTrue(path.isclosed())
        tol = 0.5
        points, tangents = sample_path(path, tol)
        self.assertEqual(points[0], points[-1])
        self.assertEqual(len(points), len(tangents))
        # Per-segment error check against the polyline restricted to
        # that segment's points.
        idx = 0
        for seg in path:
            n = 1
            while points[idx + n] != points[idx] and \
                    not (points[idx + n] == seg.end):
                n += 1
            sub = points[idx:idx + n + 1]
            if not isinstance(seg, Line):
                self.assertLessEqual(curve_to_polyline_error(seg, sub),
                                     tol * (1 + 1e-9))
            idx += n
        self.assertEqual(idx, len(points) - 1)


if __name__ == '__main__':
    unittest.main()
