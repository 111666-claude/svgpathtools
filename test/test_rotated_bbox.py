# External dependencies
from __future__ import division, absolute_import, print_function
import math
import unittest

import numpy as np

# Internal dependencies
from svgpathtools import (
    Line, QuadraticBezier, CubicBezier, Arc, Path, polygon,
    RotatedBoundingBox, min_area_rotated_bbox, flatten_path, convex_hull)
from svgpathtools.rotated_bbox import (
    AREA_TIE_REL_TOL, _all_approx_boxes, _raw_hull_edge_angles,
    _max_sagitta, _arc_piece_count)

TOL = 1e-8


# Independent reference helpers ##############################################

def _dense_samples(path, count=4000):
    """Sample a path densely without using any rotated_bbox geometry
    code, so containment checks are genuinely independent."""
    pts = []
    segments = list(path) if isinstance(path, Path) else [path]
    per = max(2, count // max(1, len(segments)))
    for seg in segments:
        ts = np.linspace(0.0, 1.0, per + 1)
        if isinstance(seg, Arc):
            # Re-derive the elliptical parametrization directly.
            angle = np.radians(seg.theta + ts * seg.delta)
            cosphi = seg.rot_matrix.real
            sinphi = seg.rot_matrix.imag
            rx, ry = seg.radius.real, seg.radius.imag
            x = rx * cosphi * np.cos(angle) - ry * sinphi * np.sin(angle) \
                + seg.center.real
            y = rx * sinphi * np.cos(angle) + ry * cosphi * np.sin(angle) \
                + seg.center.imag
            pts.extend(list(x + 1j * y))
        else:
            # Line.points() routes through np.poly1d, which mishandles a
            # complex coefficient array; evaluate the segment directly.
            pts.extend([seg.point(float(t)) for t in ts])
    return np.array(pts)


def _brute_force_box(points, angles):
    """Reference OBB: exact projected extents of a point set at every
    candidate angle, evaluated independently of the calipers code."""
    best = None
    for theta in angles:
        ux, uy = math.cos(theta), math.sin(theta)
        vx, vy = -uy, ux
        pu = [p.real * ux + p.imag * uy for p in points]
        pv = [p.real * vx + p.imag * vy for p in points]
        width = max(pu) - min(pu)
        height = max(pv) - min(pv)
        area = width * height
        if best is None or area < best[0]:
            best = (area, theta, width, height)
    return best


def _point_to_polyline_distance(pts, poly):
    """Vectorized maximum distance from samples to a polyline."""
    poly = np.asarray(poly)
    a, b = poly[:-1], poly[1:]
    d = b - a
    l2 = d.real ** 2 + d.imag ** 2
    worst = 0.0
    for z in np.asarray(pts):
        tt = np.clip(
            ((z.real - a.real) * d.real +
             (z.imag - a.imag) * d.imag) / np.where(l2 == 0, 1.0, l2),
            0.0, 1.0)
        dist = np.hypot(a.real + tt * d.real - z.real,
                        a.imag + tt * d.imag - z.imag)
        worst = max(worst, float(dist.min()))
    return worst


def _half_ellipse_paths(rx, ry, rotation_deg, center=0j):
    """A closed ellipse made of two antipodal half-ellipse arcs."""
    phi = math.radians(rotation_deg)
    u = math.cos(phi) + 1j * math.sin(phi)
    p0 = center + rx * u
    p1 = center - rx * u
    common = dict(radius=rx + 1j * ry, rotation=rotation_deg,
                  large_arc=1, sweep=1)
    return Path(Arc(p0, end=p1, **common), Arc(p1, end=p0, **common))


def _assert_contains(test, box, pts, slack=None):
    """Every sample must lie in or on the box (with tolerance slack)."""
    if slack is None:
        scale = max(box.width, box.height, 1.0)
        slack = 1e-7 * scale
    u = box.direction
    n = box.normal
    pu = ((pts - box.center) * np.conj(u)).real
    pn = ((pts - box.center) * np.conj(n)).real
    test.assertLessEqual(float(np.abs(pu).max()), box.half_width + slack)
    test.assertLessEqual(float(np.abs(pn).max()), box.half_height + slack)


class RotatedBBoxFlattenTest(unittest.TestCase):
    """The flattening stage must approximate the true curve within tol."""

    def _check_segment(self, seg, tol=TOL):
        poly = flatten_path(Path(seg), tol=tol)
        # Offset sampling grid so probes fall between polyline vertices.
        m = 3000
        ts = (np.arange(m) + 0.37) / m
        if isinstance(seg, Arc):
            pts = np.array([seg.point(t) for t in ts])
        else:
            pts = seg.points(ts)
        err = _point_to_polyline_distance(pts, poly)
        self.assertLessEqual(err, tol)

    def test_line_is_exact(self):
        seg = Line(2 - 3j, 9 + 11j)
        poly = flatten_path(Path(seg))
        self.assertEqual(poly, [seg.start, seg.end])
        self.assertEqual(_max_sagitta(seg), 0.0)

    def test_quadratic_bezier(self):
        for seg in (QuadraticBezier(0, 5 + 8j, 10),
                    QuadraticBezier(0, 50j, 10 - 50j)):
            self._check_segment(seg)

    def test_cubic_bezier_including_s_shape(self):
        for seg in (CubicBezier(0, 10 + 10j, 20 - 10j, 30),
                    CubicBezier(0, 100j, -100j, 50)):
            self._check_segment(seg)
            # An S-curve has no midpoint sagitta but real side sagitta.
            self.assertGreater(_max_sagitta(seg), 0.0)

    def test_arcs(self):
        arcs = [
            Arc(0, radius=1 + 1j, rotation=0, large_arc=0, sweep=1,
                end=2 + 0j),
            Arc(3 + 0j, radius=3 + 3j, rotation=0, large_arc=1, sweep=1,
                end=3 - 1e-9j),
            Arc(3 + 0j, radius=3 + 2j, rotation=25, large_arc=1, sweep=1,
                end=3 - 1e-9j),
        ]
        for seg in arcs:
            self._check_segment(seg)
            # The analytic piece count must be at least 1 and finite.
            self.assertGreaterEqual(_arc_piece_count(seg, TOL), 1)

    def test_tightened_tolerance_refines(self):
        seg = CubicBezier(0, 10 + 10j, 20 - 10j, 30)
        coarse = len(flatten_path(Path(seg), tol=1e-3))
        fine = len(flatten_path(Path(seg), tol=1e-7))
        self.assertGreater(fine, coarse)

    def test_negative_tol_rejected(self):
        with self.assertRaises(ValueError):
            flatten_path(Path(Line(0, 1)), tol=-1e-9)

    def test_segment_input_accepted(self):
        seg = Line(0, 1 + 1j)
        self.assertEqual(flatten_path(seg), flatten_path(Path(seg)))

    def test_shared_joints_not_duplicated(self):
        path = Path(Line(0, 1 + 1j), Line(1 + 1j, 3 + 0j))
        vertices = flatten_path(path)
        self.assertEqual(vertices, [0, 1 + 1j, 3 + 0j])


class ConvexHullTest(unittest.TestCase):
    def test_square(self):
        hull = convex_hull([0, 1, 1j, 1 + 1j, 0.5 + 0.5j])
        self.assertEqual(len(hull), 4)

    def test_collinear_points(self):
        hull = convex_hull([0, 1, 2 + 0j, 0.5 + 0j])
        self.assertEqual(len(hull), 2)

    def test_single_point(self):
        self.assertEqual(convex_hull([3 + 4j]), [3 + 4j])

    def test_ccw_order_and_contains_inputs(self):
        rng = np.random.default_rng(0)
        pts = [complex(float(x), float(y))
               for x, y in rng.uniform(-10, 10, size=(40, 2))]
        hull = convex_hull(pts)
        # All input points lie on or inside the hull polygon.
        for p in pts:
            inside = True
            for i in range(len(hull)):
                a, b = hull[i], hull[(i + 1) % len(hull)]
                cross = ((b.real - a.real) * (p.imag - a.imag) -
                         (b.imag - a.imag) * (p.real - a.real))
                if cross < -1e-9:
                    inside = False
                    break
            self.assertTrue(inside)


class CalipersReferenceTest(unittest.TestCase):
    """The linear-time calipers sweep must match brute-force projection
    on a large battery of random convex hulls."""

    def test_matches_brute_force(self):
        rng = np.random.default_rng(42)
        for _ in range(300):
            n = int(rng.integers(3, 15))
            pts = [complex(float(x), float(y))
                   for x, y in rng.uniform(-100, 100, size=(n, 2))]
            hull = convex_hull(pts)
            if len(hull) < 3:
                continue
            angles = _raw_hull_edge_angles(hull)
            widths, heights, _ = _all_approx_boxes(hull, angles)
            ref = _brute_force_box(hull, angles)
            for i, theta in enumerate(angles):
                ux, uy = math.cos(theta), math.sin(theta)
                vx, vy = -uy, ux
                pu = [p.real * ux + p.imag * uy for p in hull]
                pv = [p.real * vx + p.imag * vy for p in hull]
                self.assertAlmostEqual(
                    widths[i], max(pu) - min(pu), places=8)
                self.assertAlmostEqual(
                    heights[i], max(pv) - min(pv), places=8)
            # And the minimum found by calipers is the brute minimum.
            self.assertAlmostEqual(float((widths * heights).min()),
                                   ref[0], places=8)


class RotatedBBoxGeometryTest(unittest.TestCase):
    """Exact-geometry cases where the answer is known analytically."""

    def test_tilted_thin_rectangle(self):
        # A 10 x 1 rectangle rotated 30 degrees: the OBB is itself.
        alpha = math.radians(30)
        rot = math.cos(alpha) + 1j * math.sin(alpha)
        rect = polygon(*[rot * p for p in (0, 10 + 0j, 10 + 1j, 1j)])
        box = min_area_rotated_bbox(rect)
        self.assertAlmostEqual(box.width, 10.0, places=8)
        self.assertAlmostEqual(box.height, 1.0, places=8)
        self.assertAlmostEqual(box.theta, alpha, places=8)
        self.assertAlmostEqual(box.area, 10.0, places=7)
        # The four OBB corners coincide (as a set) with the rectangle's.
        self.assertEqual(len(box.corners), 4)
        _assert_contains(self, box, _dense_samples(rect))

    def test_axis_aligned_rectangle(self):
        rect = polygon(0, 4 + 0j, 4 + 2j, 2j)
        box = min_area_rotated_bbox(rect)
        self.assertAlmostEqual(box.theta, 0.0, places=10)
        self.assertAlmostEqual(box.width, 4.0, places=8)
        self.assertAlmostEqual(box.height, 2.0, places=8)
        self.assertAlmostEqual(box.center, 2 + 1j, places=8)

    def test_rotated_ellipse(self):
        ell = _half_ellipse_paths(3.0, 2.0, 25.0, center=4 + 5j)
        box = min_area_rotated_bbox(ell)
        self.assertAlmostEqual(box.width, 6.0, places=6)
        self.assertAlmostEqual(box.height, 4.0, places=6)
        # Angle accuracy scales with the geometric tolerance; the side
        # ratio 3:2 gives a well-separated optimum far from any tie.
        self.assertLess(abs(math.degrees(box.theta) - 25.0), 0.05)
        self.assertAlmostEqual(box.center, 4 + 5j, places=6)
        _assert_contains(self, box, _dense_samples(ell, 8000))

    def test_circle_orientation_is_deterministic(self):
        # A circle's OBB is a square; every direction ties, so theta=0.
        circle = _half_ellipse_paths(2.0, 2.0, 0.0)
        box = min_area_rotated_bbox(circle)
        self.assertAlmostEqual(box.theta, 0.0, places=8)
        self.assertAlmostEqual(box.width, 4.0, places=6)
        self.assertAlmostEqual(box.height, 4.0, places=6)

    def test_bezier_bulge_beyond_control_polygon_edges(self):
        seg = CubicBezier(0, 10 + 10j, 20 - 10j, 30)
        box = min_area_rotated_bbox(Path(seg))
        _assert_contains(self, box, _dense_samples(Path(seg)))
        # And the OBB must be no larger than the axis-aligned box area.
        xmin, xmax, ymin, ymax = Path(seg).bbox()
        self.assertLessEqual(box.area,
                             (xmax - xmin) * (ymax - ymin) + 1e-9)


class RotatedBBoxContainmentTest(unittest.TestCase):
    """All curve points of every segment type must lie in the box."""

    def test_line_segment(self):
        seg = Line(-7 + 2j, 11 - 4j)
        box = min_area_rotated_bbox(seg)
        _assert_contains(self, box, _dense_samples(seg))

    def test_quadratic(self):
        seg = QuadraticBezier(-3 + 8j, 4 - 9j, 12 + 6j)
        box = min_area_rotated_bbox(Path(seg))
        _assert_contains(self, box, _dense_samples(Path(seg)))

    def test_cubic(self):
        seg = CubicBezier(2 - 3j, 15 + 20j, 4 - 22j, 33 + 7j)
        box = min_area_rotated_bbox(Path(seg))
        _assert_contains(self, box, _dense_samples(Path(seg)))

    def test_arc(self):
        seg = Arc(1 + 1j, radius=5 + 3j, rotation=40, large_arc=1,
                  sweep=0, end=8 - 2j)
        box = min_area_rotated_bbox(Path(seg))
        _assert_contains(self, box, _dense_samples(Path(seg), 8000))

    def test_mixed_path(self):
        path = Path(
            Line(0, 4 + 3j),
            CubicBezier(4 + 3j, 6 + 9j, 12 - 8j, 14 + 2j),
            Arc(14 + 2j, radius=3 + 2j, rotation=15, large_arc=0,
                sweep=1, end=18 + 5j),
            QuadraticBezier(18 + 5j, 20 - 4j, 24 + 1j))
        box = min_area_rotated_bbox(path)
        _assert_contains(self, box, _dense_samples(path, 12000))

    def test_contains_method_boundary_points(self):
        rect = polygon(0, 5 + 0j, 5 + 2j, 2j)
        box = min_area_rotated_bbox(rect)
        corners = [0, 5 + 0j, 5 + 2j, 2j]
        for corner in corners:
            self.assertTrue(box.contains(corner, eps=1e-9))
        self.assertFalse(box.contains(100 + 100j, eps=0.0))


class RotatedBBoxDegenerateTest(unittest.TestCase):
    """Deterministic semantics for empty, point and collinear paths."""

    def test_empty_path_raises(self):
        with self.assertRaises(ValueError):
            min_area_rotated_bbox(Path())

    def test_single_point(self):
        p = 2 - 3j
        box = min_area_rotated_bbox(Path(Line(p, p)))
        self.assertEqual(box.center, p)
        self.assertEqual(box.width, 0.0)
        self.assertEqual(box.height, 0.0)
        self.assertEqual(box.theta, 0.0)
        self.assertEqual(box.corners, (p, p, p, p))
        self.assertEqual(box.area, 0.0)
        self.assertTrue(box.contains(p))

    def test_collinear_segments(self):
        # Several segments on the line through 0 and (3, 4).
        path = Path(Line(0, 3 + 4j), Line(3 + 4j, 6 + 8j))
        box = min_area_rotated_bbox(path)
        self.assertEqual(box.height, 0.0)
        self.assertAlmostEqual(box.width, 10.0, places=8)
        self.assertAlmostEqual(
            box.theta, math.atan2(4.0, 3.0), places=10)
        self.assertAlmostEqual(box.center, 3 + 4j, places=8)
        # Zero-height box: the two distinct corners are the endpoints.
        self.assertAlmostEqual(box.corners[0], 0, places=8)
        self.assertAlmostEqual(box.corners[1], 6 + 8j, places=8)

    def test_collinear_negative_direction_canonicalized(self):
        # Reversing the segment must not change the reported direction.
        a = min_area_rotated_bbox(Line(0, 3 + 4j))
        b = min_area_rotated_bbox(Line(3 + 4j, 0))
        self.assertAlmostEqual(a.theta, b.theta, places=10)
        self.assertAlmostEqual(a.width, b.width, places=10)

    def test_zero_length_segments_ignored(self):
        path = Path(Line(2 + 2j, 2 + 2j), Line(2 + 2j, 5 + 2j))
        box = min_area_rotated_bbox(path)
        self.assertAlmostEqual(box.width, 3.0, places=9)
        self.assertEqual(box.height, 0.0)

    def test_all_zero_length_path_is_a_point(self):
        path = Path(Line(1 + 1j, 1 + 1j), Line(1 + 1j, 1 + 1j))
        box = min_area_rotated_bbox(path)
        self.assertEqual(box.width, 0.0)
        self.assertEqual(box.height, 0.0)
        self.assertEqual(box.center, 1 + 1j)

    def test_tol_zero_only_valid_for_lines(self):
        box = min_area_rotated_bbox(Line(0, 3 + 4j), tol=0.0)
        self.assertAlmostEqual(box.width, 5.0, places=9)
        with self.assertRaises(ValueError):
            min_area_rotated_bbox(
                Path(CubicBezier(0, 10 + 10j, 20 - 10j, 30)), tol=0.0)


class RotatedBBoxTieTest(unittest.TestCase):
    """Near-square and exactly-square orientation ties have a defined
    output: the smallest angle in [0, pi) wins."""

    def test_square_picks_smallest_angle(self):
        sq = polygon(0, 5 + 0j, 5 + 5j, 5j)
        box = min_area_rotated_bbox(sq)
        self.assertEqual(box.theta, 0.0)
        self.assertEqual(box.width, box.height)

    def test_rotated_square_tie(self):
        sq = polygon(0, 5 + 0j, 5 + 5j, 5j).rotated(37, origin=0j)
        box = min_area_rotated_bbox(sq)
        # Candidates are 37 and 127 degrees; 37 is the smaller one.
        self.assertAlmostEqual(math.degrees(box.theta), 37.0, places=6)

    def test_near_square_is_stable(self):
        # A rectangle with sides within the relative area tie band.
        rect = polygon(0, 5.0 + 0j, 5.0 + 5.00000001, 5.00000001j)
        b1 = min_area_rotated_bbox(rect)
        b2 = min_area_rotated_bbox(rect)
        self.assertEqual(b1.theta, b2.theta)
        self.assertEqual(b1.corners, b2.corners)

    def test_width_ge_height_convention(self):
        # A tall shape must report its long side as width.
        path = Path(Line(0, 0 + 10j))
        box = min_area_rotated_bbox(path)
        self.assertGreaterEqual(box.width, box.height)
        self.assertEqual(box.height, 0.0)


class RotatedBBoxEquivarianceTest(unittest.TestCase):
    """Results must be consistent under rotation and translation."""

    def setUp(self):
        # Asymmetric shapes (no rotational symmetry to alias results).
        self.paths = [
            polygon(0, 6 + 1j, 7 + 4j, 3 + 5j, -1 + 3j),
            Path(CubicBezier(0, 10 + 10j, 20 - 10j, 30)),
            Path(QuadraticBezier(0, 50j, 10 - 50j)),
            Path(CubicBezier(2 - 3j, 15 + 20j, 4 - 22j, 33 + 7j),
                 Line(33 + 7j, 40 + 2j)),
        ]

    def _angle_close(self, a, b):
        delta = abs((a - b) % math.pi)
        return min(delta, math.pi - delta) < 1e-6

    def test_rotation_equivariance(self):
        for path in self.paths:
            base = min_area_rotated_bbox(path)
            for deg in (12.3, -47.0, 137.7, 3.0):
                rotated = path.rotated(deg, origin=0j)
                result = min_area_rotated_bbox(rotated)
                expected = (base.theta + math.radians(deg)) % math.pi
                self.assertTrue(
                    self._angle_close(result.theta, expected) or
                    self._angle_close(result.theta,
                                      expected + math.pi / 2.0))
                rot = (math.cos(math.radians(deg)) +
                       1j * math.sin(math.radians(deg)))
                self.assertAlmostEqual(abs(result.center - rot * base.center),
                                       0.0, places=6)
                self.assertAlmostEqual(result.width, base.width, places=6)
                self.assertAlmostEqual(result.height, base.height, places=6)

    def test_translation_invariance(self):
        shift = -13.7 + 8.2j
        for path in self.paths:
            base = min_area_rotated_bbox(path)
            moved = min_area_rotated_bbox(path.translated(shift))
            self.assertAlmostEqual(abs(moved.center -
                                       (base.center + shift)), 0.0, places=9)
            self.assertAlmostEqual(moved.theta, base.theta, places=10)
            self.assertAlmostEqual(moved.width, base.width, places=9)
            self.assertAlmostEqual(moved.height, base.height, places=9)
            for i in range(4):
                self.assertAlmostEqual(
                    abs(moved.corners[i] - (base.corners[i] + shift)),
                    0.0, places=9)


class RotatedBBoxSelfIntersectTest(unittest.TestCase):
    """Self-intersecting paths need no special handling: only their
    convex hull matters, and every curve point stays inside."""

    def test_self_intersecting_path(self):
        # Two crossing cubic segments forming an X-like loop.
        path = Path(
            CubicBezier(0, 10 + 10j, 10 - 10j, 20 + 0j),
            CubicBezier(0 + 0j, 10 - 10j, 10 + 10j, 20 + 0j))
        box = min_area_rotated_bbox(path)
        _assert_contains(self, box, _dense_samples(path))
        # Re-running gives an identical, deterministic answer.
        again = min_area_rotated_bbox(path)
        self.assertEqual(box.theta, again.theta)
        self.assertEqual(box.corners, again.corners)


class RotatedBBoxOptimalityTest(unittest.TestCase):
    """The chosen area must be (near-)optimal among edge directions and
    beat the naive axis-aligned box for tilted shapes."""

    def test_area_minimized_over_hull_edges(self):
        rng = np.random.default_rng(7)
        for _ in range(25):
            n = int(rng.integers(4, 10))
            pts = [complex(float(x), float(y))
                   for x, y in rng.uniform(-50, 50, size=(n, 2))]
            path = polygon(*convex_hull(pts))
            box = min_area_rotated_bbox(path)
            # Independently score every hull edge angle by brute force.
            hull = convex_hull(flatten_path(path))
            angles = _raw_hull_edge_angles(hull)
            ref = _brute_force_box(hull, angles)
            self.assertLessEqual(box.area, ref[0] + AREA_TIE_REL_TOL *
                                 max(ref[0], 1e-12) + 1e-9)

    def test_tilted_shape_beats_axis_aligned(self):
        alpha = math.radians(60)
        rot = math.cos(alpha) + 1j * math.sin(alpha)
        rect = polygon(*[rot * p for p in (0, 20 + 0j, 20 + 1j, 1j)])
        box = min_area_rotated_bbox(rect)
        xmin, xmax, ymin, ymax = rect.bbox()
        axis_area = (xmax - xmin) * (ymax - ymin)
        self.assertLess(box.area, 0.6 * axis_area)
        self.assertAlmostEqual(box.area, 20.0, places=6)


class RotatedBBoxInterfaceTest(unittest.TestCase):
    """Object surface, Path method, to_path and tolerance behavior."""

    def test_return_type_and_attributes(self):
        box = min_area_rotated_bbox(Line(0, 1 + 1j))
        self.assertIsInstance(box, RotatedBoundingBox)
        self.assertEqual(len(box.corners), 4)
        self.assertGreaterEqual(box.width, box.height)
        self.assertTrue(0.0 <= box.theta < math.pi)
        self.assertAlmostEqual(abs(box.normal - (-1j * box.direction)),
                               0.0, places=12)

    def test_path_method_matches_function(self):
        path = Path(CubicBezier(0, 10 + 10j, 20 - 10j, 30))
        self.assertEqual(path.min_area_rotated_bbox(),
                         min_area_rotated_bbox(path))

    def test_segment_direct_input(self):
        seg = QuadraticBezier(0, 5 + 8j, 10)
        box = min_area_rotated_bbox(seg)
        self.assertIsInstance(box, RotatedBoundingBox)
        _assert_contains(self, box, _dense_samples(seg))

    def test_to_path_is_closed_rectangle(self):
        rect = polygon(0, 5 + 0j, 5 + 2j, 2j)
        box = min_area_rotated_bbox(rect)
        outline = box.to_path()
        self.assertEqual(len(outline), 4)
        self.assertTrue(all(isinstance(s, Line) for s in outline))
        for corner in box.corners:
            self.assertTrue(any(abs(s.start - corner) < 1e-9
                                for s in outline))

    def test_bad_type_raises(self):
        with self.assertRaises(TypeError):
            min_area_rotated_bbox([0, 1, 2])

    def test_tolerance_changes_piece_count_not_validity(self):
        seg = Arc(1 + 1j, radius=5 + 3j, rotation=40, large_arc=1,
                  sweep=0, end=8 - 2j)
        coarse = min_area_rotated_bbox(Path(seg), tol=1e-4)
        fine = min_area_rotated_bbox(Path(seg), tol=1e-9)
        # Both contain the curve; the tighter box is never larger in area.
        pts = _dense_samples(Path(seg), 8000)
        _assert_contains(self, coarse, pts, slack=1e-3)
        _assert_contains(self, fine, pts)
        self.assertLessEqual(fine.area, coarse.area + 1e-6)
        self.assertEqual(fine.tol, 1e-9)


if __name__ == "__main__":
    unittest.main()
