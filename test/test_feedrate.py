# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
import warnings
from math import sqrt, pi, cos

import numpy as np

# Internal dependencies
from svgpathtools import (
    Line, QuadraticBezier, CubicBezier, Arc, Path, plan_feedrate)


# Tolerances used where the discretized grid is compared against analytic,
# continuous-time reference motions.  The default grid has ~200 nodes, so
# O(1/n) discretization differences around ramp transitions are expected and
# shrink with max_step (verified in test_grid_convergence).
DISK_TOL = 1e-3
LIMIT_TOL = 1e-8


def _locate(path, s_val):
    """Map a global arc length to (segment index, local parameter)."""
    lengths = [seg.length() for seg in path]
    acc = 0.0
    for idx, length in enumerate(lengths):
        if s_val <= acc + length + 1e-12 or idx == len(lengths) - 1:
            local = s_val - acc
            if length <= 1e-12:
                t = 0.0
            elif local >= length - 1e-12:
                t = 1.0
            else:
                t = path[idx].ilength(local)
            return idx, t
        acc += length
    return len(lengths) - 1, 1.0


def _curvature(path, seg_idx, t):
    """Curvature with inf on zero-length / non-differentiable points."""
    if path[seg_idx].length() <= 1e-12:
        return float("inf")
    try:
        return path.curvature(path.t2T(int(seg_idx), float(t)))
    except (ValueError, ZeroDivisionError, FloatingPointError,
            AssertionError):
        return float("inf")


def assert_respects_limits(test_case, plan, v_max, a_t, a_n):
    """Assert the whole plan stays inside every kinematic limit."""
    s, v, t = plan.samples()

    test_case.assertLessEqual(np.max(v), v_max + LIMIT_TOL)
    test_case.assertTrue(np.all(v >= -LIMIT_TOL))
    test_case.assertTrue(np.all(np.diff(t) >= -1e-12))
    test_case.assertEqual(t[0], 0.0)

    # Tangential acceleration per interval: a = (v1^2 - v0^2) / (2 ds).
    for i in range(len(s) - 1):
        ds = s[i + 1] - s[i]
        if ds <= 1e-12:
            continue
        a = (v[i + 1] ** 2 - v[i] ** 2) / (2.0 * ds)
        test_case.assertLessEqual(abs(a), a_t + LIMIT_TOL)

    # Normal acceleration at each grid node.
    for i in range(len(s)):
        kappa = _curvature(plan.path, *_locate(plan.path, s[i]))
        if np.isfinite(kappa):
            test_case.assertLessEqual(kappa * v[i] ** 2, a_n + LIMIT_TOL)
        elif plan.corner_tolerance <= 0.0:
            # Exact-stop cornering: an infinite-curvature node must be still.
            test_case.assertLessEqual(v[i], LIMIT_TOL)


class TestStraightLineReference(unittest.TestCase):
    """Independent trapezoidal/triangular ramp references on a Line."""

    def setUp(self):
        self.v_max = 10.0
        self.a_t = 5.0
        self.a_n = 20.0
        self.line = Path(Line(0, 100 + 0j))

    def test_trapezoid_duration_and_speeds(self):
        plan = plan_feedrate(self.line, self.v_max, self.a_t, self.a_n)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        # Ramp distance s_r = v^2/(2a) = 100/10 = 10 each end; ramp time 2 s.
        # Cruise covers 80 units at 10 -> 8 s; total 12 s.
        self.assertAlmostEqual(plan.duration(), 12.0, places=6)
        self.assertAlmostEqual(float(plan.v.max()), self.v_max, places=8)

    def test_constant_acceleration_laws(self):
        plan = plan_feedrate(self.line, self.v_max, self.a_t, self.a_n)
        # Inside the acceleration ramp: v = sqrt(2 a s), t = sqrt(2s/a),
        # s(t) = 0.5 a t^2.
        self.assertAlmostEqual(plan.velocity_at_s(5.0),
                               sqrt(2 * self.a_t * 5.0), places=8)
        self.assertAlmostEqual(plan.time_at_s(5.0),
                               sqrt(2 * 5.0 / self.a_t), places=8)
        self.assertAlmostEqual(plan.s_at_time(1.0),
                               0.5 * self.a_t * 1.0 ** 2, places=8)

    def test_triangular_profile_short_line(self):
        short = Path(Line(0, 4 + 0j))
        plan = plan_feedrate(short, self.v_max, self.a_t, self.a_n)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        # v_max never reached: peak v = sqrt(a L), duration = 2 peak / a.
        peak = sqrt(self.a_t * 4.0)
        self.assertAlmostEqual(float(plan.v.max()), peak, places=8)
        self.assertAlmostEqual(plan.duration(), 2 * peak / self.a_t, places=6)

    def test_configurable_boundary_speeds(self):
        plan = plan_feedrate(self.line, self.v_max, self.a_t, self.a_n,
                             start_speed=6.0, end_speed=4.0)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        self.assertAlmostEqual(plan.v[0], 6.0, places=10)
        self.assertAlmostEqual(plan.v[-1], 4.0, places=10)
        rest = plan_feedrate(self.line, self.v_max, self.a_t, self.a_n)
        self.assertLess(plan.duration(), rest.duration())

    def test_query_round_trip(self):
        plan = plan_feedrate(self.line, self.v_max, self.a_t, self.a_n)
        for s_val in np.linspace(0.0, plan.path_length(), 25):
            t_val = plan.time_at_s(s_val)
            self.assertAlmostEqual(plan.s_at_time(t_val), s_val, places=6)
        # Clamping outside the domain is deterministic.
        self.assertEqual(plan.s_at_time(-1.0), 0.0)
        self.assertEqual(plan.s_at_time(plan.duration() + 1.0),
                         plan.path_length())
        self.assertEqual(plan.velocity_at_s(plan.path_length()),
                         plan.velocity_at_s(plan.path_length() + 5.0))


class TestCircularArcReference(unittest.TestCase):
    """Constant normal-acceleration ceiling on a radius-1 semicircle."""

    def setUp(self):
        self.arc = Path(Arc(1 + 0j, 1 + 1j, 0, False, True, -1 + 0j))
        self.v_max = 10.0
        self.a_t = 5.0
        self.a_n = 4.0  # curvature 1 -> ceiling v = sqrt(4) = 2

    def test_plateau_speed_and_duration(self):
        plan = plan_feedrate(self.arc, self.v_max, self.a_t, self.a_n)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        self.assertLessEqual(plan.v.max(), 2.0 + LIMIT_TOL)
        # Interior nodes (away from the boundary ramps) sit at the normal
        # ceiling, reduced only by the documented curvature safety factor.
        interior = plan.v[40:-40]
        self.assertTrue(np.allclose(interior, 2.0, rtol=2e-5))
        # Reference time: ramps reach 2 over s = v^2/(2a) = 0.4 each end;
        # the rest of the semicircle is cruised at ~2.
        cruise_len = pi - 0.8
        t_ref = 2 * (2.0 / self.a_t) + cruise_len / 2.0
        self.assertAlmostEqual(plan.duration(), t_ref, delta=DISK_TOL)

    def test_normal_acceleration_binding(self):
        plan = plan_feedrate(self.arc, self.v_max, self.a_t, self.a_n,
                             start_speed=2.0, end_speed=2.0)
        # Every plateau node saturates the normal limit up to the safety
        # margin (a_n is never exceeded; see assert_respects_limits).
        from svgpathtools.feedrate import _CURV_SAFETY
        for i in range(5, len(plan) - 5):
            kappa = _curvature(plan.path, *_locate(plan.path, plan.s[i]))
            self.assertAlmostEqual(kappa * plan.v[i] ** 2,
                                   self.a_n * _CURV_SAFETY ** 2, delta=1e-6)

    def test_very_short_arc_is_deterministic(self):
        short = Arc(0.01 + 0j, 0.01 + 0.01j, 0, False, True, 0 + 0.01j)
        plan = plan_feedrate(Path(short), self.v_max, self.a_t, self.a_n,
                             max_step=1.0)
        self.assertGreaterEqual(len(plan), 2)
        self.assertTrue(np.all(np.isfinite(plan.t)))
        self.assertTrue(np.all(np.isfinite(plan.v)))
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)


class TestPolylineCornerReference(unittest.TestCase):
    """Sharp 90-degree corner: exact stop vs. blend-tolerance cornering."""

    def setUp(self):
        self.poly = Path(Line(0, 10 + 0j), Line(10 + 0j, 10 + 10j))
        self.v_max, self.a_t, self.a_n = 10.0, 5.0, 20.0

    def test_exact_stop_corner(self):
        plan = plan_feedrate(self.poly, self.v_max, self.a_t, self.a_n)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        corner = np.argmin(np.abs(plan.s - 10.0))
        self.assertEqual(plan.v[corner], 0.0)
        # Each 10-unit leg is triangular from/to rest: peak sqrt(a L),
        # leg time 2 peak / a.
        peak = sqrt(self.a_t * 10.0)
        self.assertAlmostEqual(plan.duration(),
                               4.0 * peak / self.a_t, places=5)

    def test_blend_tolerance_corner_speed(self):
        eps = 0.01
        plan = plan_feedrate(self.poly, self.v_max, self.a_t, self.a_n,
                             corner_tolerance=eps)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        corner = np.argmin(np.abs(plan.s - 10.0))
        # R = eps / (1/cos(theta/2) - 1), v = sqrt(a_n R).
        radius = eps / (1.0 / cos(pi / 4.0) - 1.0)
        self.assertAlmostEqual(plan.v[corner],
                               sqrt(self.a_n * radius), places=8)
        # Cornering must be faster than the exact stop.
        stop = plan_feedrate(self.poly, self.v_max, self.a_t, self.a_n)
        self.assertLess(plan.duration(), stop.duration())

    def test_reversal_cusp_is_exact_stop_even_with_tolerance(self):
        # Out-and-back along one line: tangent flips by 180 degrees.
        path = Path(Line(0, 10 + 0j), Line(10 + 0j, 0))
        plan = plan_feedrate(path, self.v_max, self.a_t, self.a_n,
                             corner_tolerance=0.05)
        corner = np.argmin(np.abs(plan.s - 10.0))
        self.assertEqual(plan.v[corner], 0.0)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)


class TestCurvesAndJoints(unittest.TestCase):

    def setUp(self):
        self.v_max, self.a_t, self.a_n = 8.0, 4.0, 10.0

    def test_cubic_bezier_with_cusp_stops(self):
        # This control polygon produces a derivative zero at t = 0.5.
        cusp = CubicBezier(0 + 0j, 1 + 1j, 1 - 1j, 2 + 0j)
        plan = plan_feedrate(Path(cusp), self.v_max, self.a_t, self.a_n)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        self.assertLessEqual(float(plan.v.min()), 1e-10)

    def test_smooth_g1_joint_is_not_stopped(self):
        path = Path(CubicBezier(0, 5 + 0j, 5 + 5j, 10 + 5j),
                    CubicBezier(10 + 5j, 15 + 5j, 15 + 0j, 20 + 0j))
        plan = plan_feedrate(path, self.v_max, self.a_t, self.a_n)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        self.assertGreater(plan.velocity_at_s(path[0].length()), 0.5)

    def test_quadratic_bezier_limits(self):
        path = Path(QuadraticBezier(0, 5 + 8j, 10 + 0j))
        plan = plan_feedrate(path, self.v_max, self.a_t, self.a_n)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        self.assertTrue(np.all(np.isfinite(plan.t)))


class TestClosedLoops(unittest.TestCase):

    def setUp(self):
        self.square = Path(
            Line(0, 10 + 0j), Line(10 + 0j, 10 + 10j),
            Line(10 + 10j, 10j), Line(10j, 0))
        self.v_max, self.a_t, self.a_n = 10.0, 5.0, 20.0

    def test_default_exact_stops(self):
        plan = plan_feedrate(self.square, self.v_max, self.a_t, self.a_n)
        self.assertTrue(plan.closed)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        self.assertEqual(float(plan.v.min()), 0.0)
        self.assertAlmostEqual(plan.v[0], plan.v[-1], places=12)

    def test_natural_seam_corners_with_tolerance(self):
        eps = 0.05
        plan = plan_feedrate(self.square, self.v_max, self.a_t, self.a_n,
                             corner_tolerance=eps)
        assert_respects_limits(self, plan, self.v_max, self.a_t, self.a_n)
        radius = eps / (1.0 / cos(pi / 4.0) - 1.0)
        corner_speed = sqrt(self.a_n * radius)
        # All four corners, including the seam, share the same ceiling.
        for s_corner in (0.0, 10.0, 20.0, 30.0, 40.0):
            node = np.argmin(np.abs(plan.s - s_corner))
            self.assertAlmostEqual(plan.v[node], corner_speed, places=8)
        self.assertAlmostEqual(plan.v[0], plan.v[-1], places=12)

    def test_explicit_seam_rest_is_honored(self):
        free = plan_feedrate(self.square, self.v_max, self.a_t, self.a_n,
                             corner_tolerance=0.05)
        pinned = plan_feedrate(self.square, self.v_max, self.a_t, self.a_n,
                               corner_tolerance=0.05, start_speed=0.0,
                               end_speed=0.0)
        self.assertEqual(pinned.v[0], 0.0)
        self.assertGreater(pinned.duration(), free.duration())

    def test_closed_loop_points_stay_on_path(self):
        plan = plan_feedrate(self.square, self.v_max, self.a_t, self.a_n)
        for t_val in np.linspace(0.0, plan.duration(), 20):
            z = plan.point_at_time(t_val)
            self.assertTrue(-1e-9 <= z.real <= 10 + 1e-9)
            self.assertTrue(-1e-9 <= z.imag <= 10 + 1e-9)


class TestDegenerateAndInfeasibleInputs(unittest.TestCase):

    def test_zero_length_segment_inside_path(self):
        path = Path(Line(0, 0), Line(0, 5 + 0j))
        plan = plan_feedrate(path, 10.0, 5.0, 20.0)
        assert_respects_limits(self, plan, 10.0, 5.0, 20.0)
        # Equivalent to a 5-unit rest-to-rest line: peak 5, duration 2 s.
        self.assertAlmostEqual(plan.duration(), 2.0, places=5)

    def test_entirely_zero_length_path(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            path = Path(Line(3 + 2j, 3 + 2j))
            plan = plan_feedrate(path, 10.0, 5.0, 20.0, start_speed=1.0)
        self.assertEqual(len(caught), 1)
        self.assertEqual(plan.duration(), 0.0)
        self.assertEqual(plan.v[0], 0.0)
        self.assertEqual(plan.point_at_time(0.0), 3 + 2j)
        self.assertEqual(plan.s_at_time(1.0), 0.0)
        self.assertEqual(plan.velocity_at_s(0.0), 0.0)

    def test_infeasible_boundary_speeds_clamp(self):
        arc = Path(Arc(1 + 0j, 1 + 1j, 0, False, True, -1 + 0j))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = plan_feedrate(arc, 10.0, 5.0, 4.0,
                                 start_speed=9.0, end_speed=8.0)
        self.assertEqual(len(caught), 2)
        assert_respects_limits(self, plan, 10.0, 5.0, 4.0)
        # Clamped down to the arc's normal ceiling (safety factor included).
        self.assertAlmostEqual(plan.v[0], 2.0, delta=1e-4)
        self.assertAlmostEqual(plan.v[-1], 2.0, delta=1e-4)

    def test_boundary_above_vmax_on_straight_path(self):
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            plan = plan_feedrate(Path(Line(0, 100 + 0j)), 10.0, 5.0, 20.0,
                                 start_speed=15.0)
        self.assertEqual(len(caught), 1)
        self.assertEqual(plan.v[0], 10.0)

    def test_path_is_not_modified(self):
        path = Path(CubicBezier(0, 5 + 0j, 5 + 5j, 10 + 5j),
                    CubicBezier(10 + 5j, 15 + 5j, 15 + 0j, 20 + 0j))
        snapshot = [seg.bpoints() for seg in path]
        plan = plan_feedrate(path, 8.0, 4.0, 10.0)
        self.assertEqual([seg.bpoints() for seg in path], snapshot)
        self.assertIs(plan.path, path)

    def test_invalid_arguments_raise(self):
        line = Path(Line(0, 1 + 0j))
        defaults = dict(max_speed=10.0, tangential_accel=5.0,
                        normal_accel=20.0)
        for bad in (dict(max_speed=0.0), dict(tangential_accel=-1.0),
                    dict(normal_accel=0.0), dict(start_speed=-1.0),
                    dict(corner_tolerance=-0.1)):
            with self.assertRaises(ValueError):
                plan_feedrate(line, **dict(defaults, **bad))
        with self.assertRaises(ValueError):
            plan_feedrate(Path(), 10.0, 5.0, 20.0)
        with self.assertRaises(TypeError):
            plan_feedrate([Line(0, 1 + 0j)], 10.0, 5.0, 20.0)

    def test_discontinuous_path_raises(self):
        gap = Path(Line(0, 1 + 0j), Line(5 + 0j, 6 + 0j))
        with self.assertRaises(ValueError):
            plan_feedrate(gap, 10.0, 5.0, 20.0)


class TestToleranceAndGridControl(unittest.TestCase):

    def setUp(self):
        self.arc = Path(Arc(1 + 0j, 1 + 1j, 0, False, True, -1 + 0j))
        self.cruise_len = pi - 0.8
        self.t_ref = 2 * (2.0 / 5.0) + self.cruise_len / 2.0

    def test_grid_convergence(self):
        errors = []
        for step in (pi / 200.0, pi / 1000.0, pi / 5000.0):
            plan = plan_feedrate(self.arc, 10.0, 5.0, 4.0, max_step=step)
            errors.append(abs(plan.duration() - self.t_ref))
        # Grid discretization error shrinks as the grid refines; the residual
        # floor is the documented curvature safety factor (~1e-5).
        self.assertLess(errors[1], errors[0])
        self.assertLess(errors[2], errors[1])
        self.assertLess(errors[2], 5e-5)

    def test_time_tolerance_refines_until_stable(self):
        coarse = plan_feedrate(self.arc, 10.0, 5.0, 4.0)
        fine = plan_feedrate(self.arc, 10.0, 5.0, 4.0, time_tolerance=1e-7)
        self.assertGreater(len(fine), len(coarse))
        self.assertAlmostEqual(fine.duration(), self.t_ref, delta=5e-5)

    def test_joints_always_included(self):
        # A huge max_step must still place nodes on every segment joint.
        poly = Path(Line(0, 10 + 0j), Line(10 + 0j, 10 + 10j),
                    Line(10 + 10j, 10j))
        plan = plan_feedrate(poly, 10.0, 5.0, 20.0, max_step=1e6)
        for s_joint in (0.0, 10.0, 20.0, 30.0):
            self.assertTrue(np.any(np.abs(plan.s - s_joint) < 1e-9))


class TestPlanContainer(unittest.TestCase):

    def test_repr_samples_and_accessors(self):
        plan = plan_feedrate(Path(Line(0, 50 + 0j)), 10.0, 5.0, 20.0)
        self.assertIn("FeedRatePlan", repr(plan))
        self.assertEqual(len(plan), len(plan.s))
        s, v, t = plan.samples()
        self.assertIs(s, plan.s)
        self.assertIs(v, plan.v)
        self.assertIs(t, plan.t)
        self.assertAlmostEqual(plan.path_length(), 50.0, places=10)
        # Acceleration accessor matches the per-interval analytic value.
        a = plan.acceleration_at_s(plan.s[3])
        i = 3
        ds = plan.s[i + 1] - plan.s[i]
        expected = (plan.v[i + 1] ** 2 - plan.v[i] ** 2) / (2 * ds)
        self.assertAlmostEqual(a, expected, places=10)

    def test_point_at_s_matches_geometry(self):
        path = Path(CubicBezier(0, 5 + 0j, 5 + 5j, 10 + 5j))
        plan = plan_feedrate(path, 8.0, 4.0, 10.0)
        for s_val in np.linspace(0.0, plan.path_length(), 15):
            T = path.ilength(s_val)
            self.assertAlmostEqual(abs(plan.point_at_s(s_val)
                                       - path.point(T)), 0.0, places=8)


if __name__ == "__main__":
    unittest.main()
