# External dependencies
from __future__ import division, absolute_import, print_function
import unittest

import numpy as np

# Internal dependencies
from svgpathtools import (
    Path, Line, QuadraticBezier, CubicBezier, Arc, parse_path,
    morph_paths, MorphError)


# Test-fixture builders ######################################################

def rect_path(x0, y0, x1, y1):
    """A closed axis-aligned rectangle made of four Line segments."""
    pts = [complex(x0, y0), complex(x1, y0),
           complex(x1, y1), complex(x0, y1)]
    return Path(*[Line(pts[i], pts[(i + 1) % 4]) for i in range(4)])


def circle_path(cx, cy, r):
    """A full circle built from two semicircle Arc segments."""
    center = complex(cx, cy)
    radius = complex(r, r)
    return Path(Arc(center - r, radius, 0, 1, 1, center + r),
                Arc(center + r, radius, 0, 1, 1, center - r))


def dist_to_rect(p, x0, y0, x1, y1):
    """Distance from point p to the boundary of an axis-aligned rectangle."""
    x, y = p.real, p.imag
    if x0 <= x <= x1 and y0 <= y <= y1:
        return min(x - x0, x1 - x, y - y0, y1 - y)
    dx = max(x0 - x, 0.0, x - x1)
    dy = max(y0 - y, 0.0, y - y1)
    return float(np.hypot(dx, dy))


def dist_to_circle(p, center, r):
    return abs(abs(p - center) - r)


def winding_number(path, n=720):
    """Winding number of a closed path around its own centroid.  A twist
    free simple loop returns +/-1 regardless of its start point."""
    pts = [path.point(i / n) for i in range(n + 1)]
    center = sum(pts[:-1]) / n
    total, prev = 0.0, np.angle(pts[0] - center)
    for z in pts[1:]:
        angle = np.angle(z - center)
        delta = angle - prev
        if delta > np.pi:
            delta -= 2 * np.pi
        if delta < -np.pi:
            delta += 2 * np.pi
        total += delta
        prev = angle
    return round(total / (2 * np.pi), 4)


def rotated_start(path, fraction):
    """A copy of a closed cubic path whose parameterization starts at the
    given arc-length fraction, with junctions snapped exactly."""
    first = path.cropped(fraction, 1.0)
    rest = path.cropped(0.0, fraction)
    out = Path(*[seg for seg in first])
    junction = out[-1].end
    for seg in rest:
        seg = CubicBezier(junction, seg.control1, seg.control2, seg.end)
        out.append(seg)
        junction = seg.end
    last = out[-1]
    out[-1] = CubicBezier(last.start, last.control1, last.control2,
                          out[0].start)
    return out


# Tests ######################################################################

class MorphBasicTest(unittest.TestCase):
    def setUp(self):
        self.rect = rect_path(0, 0, 2, 2)
        self.circle = circle_path(1, 1, 1)
        self.n = 3000

    def test_returns_closed_continuous_path_of_cubics(self):
        mid = morph_paths(self.rect, self.circle, 0.5)
        self.assertIsInstance(mid, Path)
        self.assertTrue(mid.iscontinuous())
        self.assertTrue(mid.isclosedac())
        self.assertTrue(all(isinstance(seg, CubicBezier) for seg in mid))
        self.assertGreater(len(mid), 1)

    def test_output_is_drawable_and_round_trips_through_d_string(self):
        mid = morph_paths(self.rect, self.circle, 0.5)
        d_string = mid.d()
        self.assertTrue(d_string)
        reparsed = parse_path(d_string)
        self.assertEqual(len(reparsed), len(mid))
        for t in (0.0, 0.37, 0.81, 1.0):
            self.assertAlmostEqual(reparsed.point(t), mid.point(t),
                                   delta=1e-9)

    def test_endpoints_match_inputs_within_tolerance(self):
        # Lines are degree-elevated exactly, so t=0 reproduces the rectangle
        # to machine precision.
        at_start = morph_paths(self.rect, self.circle, 0.0)
        err_rect = max(dist_to_rect(at_start.point(i / self.n), 0, 0, 2, 2)
                       for i in range(self.n))
        self.assertLess(err_rect, 1e-12)

        # The circle is arc-based, so t=1 matches within the arc tolerance.
        at_end = morph_paths(self.rect, self.circle, 1.0)
        err_circle = max(dist_to_circle(at_end.point(i / self.n),
                                        1 + 1j, 1.0)
                         for i in range(self.n))
        self.assertLess(err_circle, 1e-6)

        # Symmetry of the argument order.
        back_start = morph_paths(self.circle, self.rect, 0.0)
        back_end = morph_paths(self.circle, self.rect, 1.0)
        self.assertLess(max(dist_to_circle(back_start.point(i / self.n),
                                           1 + 1j, 1.0)
                            for i in range(self.n)), 1e-6)
        self.assertLess(max(dist_to_rect(back_end.point(i / self.n),
                                         0, 0, 2, 2)
                            for i in range(self.n)), 1e-12)

    def test_arc_tolerance_parameter_is_respected(self):
        loose = morph_paths(self.rect, self.circle, 1.0, arc_tolerance=1e-3)
        tight = morph_paths(self.rect, self.circle, 1.0, arc_tolerance=1e-9)
        err_loose = max(dist_to_circle(loose.point(i / self.n), 1 + 1j, 1.0)
                        for i in range(self.n))
        err_tight = max(dist_to_circle(tight.point(i / self.n), 1 + 1j, 1.0)
                        for i in range(self.n))
        self.assertLess(err_loose, 1e-3)
        self.assertLess(err_tight, err_loose)
        # A tighter tolerance needs at least as many cubic pieces.
        self.assertGreaterEqual(len(tight), len(loose))


class MorphCorrespondenceTest(unittest.TestCase):
    def setUp(self):
        self.rect = rect_path(0, 0, 2, 2)
        self.circle = circle_path(1, 1, 1)

    def test_different_segment_counts_are_arc_length_aligned(self):
        # Four lines vs. many arc-approximation cubics: the output piece list
        # must still pair up one-to-one and stay continuous.
        mid = morph_paths(self.rect, self.circle, 0.42)
        # Every interior joint and the closing seam must coincide exactly.
        interior_gaps = [abs(mid[i].end - mid[i + 1].start)
                         for i in range(len(mid) - 1)]
        self.assertEqual(max(interior_gaps), 0.0)  # no torn outline
        self.assertEqual(abs(mid[-1].end - mid[0].start), 0.0)  # exact closure
        # Perimeter of the morph stays between the two input perimeters.
        self.assertLessEqual(self.circle.length() - 1e-9, mid.length())
        self.assertLessEqual(mid.length(), self.rect.length() + 1e-9)

    def test_no_twist_for_every_progress(self):
        for t in np.linspace(0.05, 0.95, 10):
            self.assertEqual(
                abs(winding_number(morph_paths(self.rect, self.circle,
                                               float(t)))),
                1.0)

    def test_cyclic_start_change_does_not_break_endpoints_or_twist(self):
        approx = Path(*[seg for seg in self.circle])
        approx.approximate_arcs_with_cubics(error=0.01)
        shifted = rotated_start(approx, 0.37)
        self.assertTrue(shifted.iscontinuous())
        for t in (0.0, 0.25, 0.5, 0.75, 1.0):
            morphed = morph_paths(self.rect, shifted, t)
            self.assertEqual(abs(winding_number(morphed)), 1.0)
        n = 2000
        m0 = morph_paths(self.rect, shifted, 0.0)
        m1 = morph_paths(self.rect, shifted, 1.0)
        self.assertLess(max(dist_to_rect(m0.point(i / n), 0, 0, 2, 2)
                            for i in range(n)), 1e-12)
        self.assertLess(max(dist_to_circle(m1.point(i / n), 1 + 1j, 1.0)
                            for i in range(n)), 1e-3)

    def test_opposite_orientation_is_aligned_not_twisted(self):
        clockwise = self.circle.reversed()
        self.assertNotEqual(np.sign(self.circle.area(chord_length=1e-3)),
                            np.sign(clockwise.area(chord_length=1e-3)))
        for t in (0.0, 0.5, 1.0):
            morphed = morph_paths(self.rect, clockwise, t)
            self.assertTrue(morphed.iscontinuous())
            self.assertTrue(morphed.isclosedac())
            # Output follows the orientation of the first input (CCW).
            self.assertEqual(winding_number(morphed), 1.0)


class MorphHolesTest(unittest.TestCase):
    def setUp(self):
        inner = [complex(1, 1), complex(1, 3), complex(3, 3), complex(3, 1)]
        self.ring_rect = Path(
            *[seg for seg in rect_path(0, 0, 4, 4)],
            *[Line(inner[i], inner[(i + 1) % 4]) for i in range(4)])
        self.ring_circle = Path(
            *[seg for seg in circle_path(2, 2, 2.2)],
            *[seg for seg in circle_path(2, 2, 1.0).reversed()])

    def test_subpath_count_closure_and_containment(self):
        for t in np.linspace(0.0, 1.0, 11):
            morphed = morph_paths(self.ring_rect, self.ring_circle, float(t))
            subs = morphed.continuous_subpaths()
            self.assertEqual(len(subs), 2)
            for sub in subs:
                self.assertTrue(sub.iscontinuous())
                self.assertTrue(sub.isclosedac())
            outer, hole = subs
            self.assertTrue(hole.is_contained_by(outer))

    def test_endpoints_keep_outer_and_hole_correspondence(self):
        m0 = morph_paths(self.ring_rect, self.ring_circle, 0.0)
        n = 1200
        outer_errors, inner_errors = [], []
        for sub in m0.continuous_subpaths():
            outer_errors.append(max(dist_to_rect(sub.point(i / n),
                                                 0, 0, 4, 4)
                                    for i in range(n)))
            inner_errors.append(max(dist_to_rect(sub.point(i / n),
                                                 1, 1, 3, 3)
                                    for i in range(n)))
        # One subpath lies on the outer rectangle, the other on the inner.
        self.assertEqual(sorted(round(min(e), 12)
                               for e in zip(outer_errors, inner_errors)),
                         [0.0, 0.0])

    def test_holes_are_paired_by_position_not_list_order(self):
        # Two holes, swapped in order in the second path; pairing must still
        # match the nearest hole rather than blindly use list indices.
        def square_hole(cx, cy, half):
            return rect_path(cx - half, cy - half, cx + half, cy + half)
        outer1 = rect_path(0, 0, 10, 10)
        hole_a1 = square_hole(2.5, 5, 1.0)
        hole_b1 = square_hole(7.5, 5, 1.0)
        path1 = Path(*[s for s in outer1], *[s for s in hole_a1],
                     *[s for s in hole_b1])
        outer2 = rect_path(0, 0, 10, 10)
        # Same two holes but listed in the opposite order.
        path2 = Path(*[s for s in outer2], *[s for s in hole_b1],
                     *[s for s in hole_a1])
        mid = morph_paths(path1, path2, 0.5)
        self.assertEqual(len(mid.continuous_subpaths()), 3)
        # With identical holes swapped, the mid morph should reproduce the
        # same geometry as the unswapped morph.
        path2_sorted = Path(*[s for s in outer2], *[s for s in hole_a1],
                            *[s for s in hole_b1])
        ref = morph_paths(path1, path2_sorted, 0.5)
        for sub_m, sub_r in zip(mid.continuous_subpaths(),
                                ref.continuous_subpaths()):
            self.assertAlmostEqual(sub_m.length(), sub_r.length(), 9)


class MorphSegmentTypesTest(unittest.TestCase):
    def test_mixed_lines_quadratics_and_arcs(self):
        quad = Path(
            QuadraticBezier(0 + 0j, 1 - 1j, 2 + 0j),
            QuadraticBezier(2 + 0j, 3 + 1j, 4 + 0j),
            QuadraticBezier(4 + 0j, 3 + 2j, 2 + 2j),
            QuadraticBezier(2 + 2j, 1 + 3j, 0 + 2j),
            Line(0 + 2j, 0 + 0j))
        target = circle_path(2, 1, 1.5)
        mid = morph_paths(quad, target, 0.5)
        self.assertTrue(mid.iscontinuous())
        self.assertTrue(mid.isclosedac())
        self.assertTrue(all(isinstance(seg, CubicBezier) for seg in mid))

    def test_bezier_input_is_exact_at_endpoints(self):
        # A closed cubic-bezier loop (no arcs) is reproduced exactly at t=0
        # and t=1.  The morph re-parameterizes by arc length and may rotate
        # the loop start, so equivalence is checked geometrically: every
        # morphed point must lie on the original curve.  The threshold is a
        # small multiple of the dense reference polyline's vertex spacing, so
        # only sampling-grid error (not shape error) is tolerated.
        loop = Path(
            CubicBezier(0 + 0j, 1 - 1j, 2 - 1j, 3 + 0j),
            CubicBezier(3 + 0j, 4 + 1j, 2 + 2j, 1 + 2j),
            CubicBezier(1 + 2j, 0 + 2j, -1 + 1j, 0 + 0j))
        target = circle_path(1.5, 0.75, 1.2)

        def point_set_distance(p, q):
            # Directed Hausdorff from p to q, computed in chunks to avoid
            # materializing a full n_p x n_q distance matrix.
            q_n = max(20000, int(q.length() / 1e-4))
            p_n = 2000
            qs = np.array([[q.point(i / q_n).real, q.point(i / q_n).imag]
                           for i in range(q_n)])
            worst = 0.0
            for start in range(0, p_n, 200):
                ps = np.array(
                    [[p.point(i / p_n).real, p.point(i / p_n).imag]
                     for i in range(start, min(start + 200, p_n))])
                dist = np.sqrt(((ps[:, None, :] - qs[None, :, :]) ** 2)
                               .sum(axis=2))
                worst = max(worst, dist.min(axis=1).max())
            return worst

        spacing = loop.length() / max(20000, int(loop.length() / 1e-4))
        self.assertLess(point_set_distance(
            morph_paths(loop, target, 0.0), loop), 5 * spacing)
        self.assertLess(point_set_distance(
            morph_paths(target, loop, 1.0), loop), 5 * spacing)

class MorphErrorTest(unittest.TestCase):
    def setUp(self):
        self.rect = rect_path(0, 0, 2, 2)
        self.circle = circle_path(1, 1, 1)

    def assertMorphError(self, *args, **kwargs):
        with self.assertRaises(MorphError):
            morph_paths(*args, **kwargs)

    def test_invalid_progress(self):
        for bad in (-0.1, 1.1, float('nan'), float('inf'),
                    float('-inf'), "0.5", None):
            self.assertMorphError(self.rect, self.circle, bad)

    def test_numpy_scalar_progress_is_accepted(self):
        mid = morph_paths(self.rect, self.circle, np.float64(0.5))
        self.assertTrue(mid.isclosedac())

    def test_invalid_keyword_arguments(self):
        self.assertMorphError(self.rect, self.circle, 0.5,
                              arc_tolerance=0.0)
        self.assertMorphError(self.rect, self.circle, 0.5,
                              arc_tolerance=-1.0)
        self.assertMorphError(self.rect, self.circle, 0.5,
                              align_samples=4)

    def test_empty_and_open_paths(self):
        self.assertMorphError(Path(), self.circle, 0.5)
        self.assertMorphError(Path(Line(0, 1)), self.circle, 0.5)
        self.assertMorphError(self.rect, Path(Line(0, 1)), 0.5)

    def test_topology_mismatch(self):
        inner = [complex(1, 1), complex(1, 3), complex(3, 3), complex(3, 1)]
        with_hole = Path(
            *[seg for seg in rect_path(0, 0, 4, 4)],
            *[Line(inner[i], inner[(i + 1) % 4]) for i in range(4)])
        # One subpath vs. two subpaths -> incompatible hole topology.
        self.assertMorphError(self.rect, with_hole, 0.5)
        self.assertMorphError(with_hole, self.circle, 0.5)

    def test_zero_length_contour(self):
        collapsed = Path(Line(0j, 0j), Line(0j, 0j))
        self.assertMorphError(collapsed, self.circle, 0.5)
        self.assertMorphError(self.circle, collapsed, 0.5)

    def test_non_finite_coordinates(self):
        bad_rect = Path(Line(complex(float('nan'), 0), 1 + 0j),
                        Line(1 + 0j, 1 + 1j),
                        Line(1 + 1j, 0 + 1j),
                        Line(0 + 1j, complex(float('nan'), 0)))
        self.assertMorphError(bad_rect, self.circle, 0.5)

    def test_wrong_input_type(self):
        self.assertMorphError([Line(0, 1)], self.circle, 0.5)

    def test_no_nan_anywhere_in_output(self):
        inner = [complex(1, 1), complex(1, 3), complex(3, 3), complex(3, 1)]
        with_hole = Path(
            *[seg for seg in rect_path(0, 0, 4, 4)],
            *[Line(inner[i], inner[(i + 1) % 4]) for i in range(4)])
        ring_circle = Path(
            *[seg for seg in circle_path(2, 2, 2.2)],
            *[seg for seg in circle_path(2, 2, 1.0).reversed()])
        for t in np.linspace(0.0, 1.0, 21):
            for path in (morph_paths(self.rect, self.circle, float(t)),
                         morph_paths(with_hole, ring_circle, float(t))):
                for seg in path:
                    for z in seg.bpoints():
                        self.assertTrue(np.isfinite(z.real))
                        self.assertTrue(np.isfinite(z.imag))


class PathMethodTest(unittest.TestCase):
    def test_morphed_with_matches_module_function(self):
        rect = rect_path(0, 0, 2, 2)
        circle = circle_path(1, 1, 1)
        via_method = rect.morphed_with(circle, 0.3)
        via_function = morph_paths(rect, circle, 0.3)
        self.assertEqual(via_method, via_function)
        # Keyword arguments pass through.
        self.assertEqual(rect.morphed_with(circle, 0.3, arc_tolerance=1e-8),
                         morph_paths(rect, circle, 0.3,
                                     arc_tolerance=1e-8))


if __name__ == '__main__':
    unittest.main()
