# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
from math import sqrt, pi, cos, sin, radians

import numpy as np

# Internal dependencies
from svgpathtools import (
    Line, QuadraticBezier, CubicBezier, Arc, Path, parse_path,
    area_moments, AreaMoments,
)


def rect_path(x0, y0, w, h, ccw=True):
    """A rectangle as a closed Path of Lines."""
    pts = [(x0, y0), (x0 + w, y0), (x0 + w, y0 + h), (x0, y0 + h)]
    if not ccw:
        pts = pts[::-1]
    pts = [complex(x, y) for x, y in pts]
    n = len(pts)
    return Path(*[Line(pts[i], pts[(i + 1) % n]) for i in range(n)])


class TestAreaMomentsLines(unittest.TestCase):
    """Analytic checks on polygonal (Line-only) paths."""

    def test_rectangle(self):
        # 3 x 2 rectangle with lower-left corner at (1, 4)
        m = area_moments(rect_path(1, 4, 3, 2))
        self.assertAlmostEqual(m.area, 6.0)
        self.assertAlmostEqual(m.centroid, 2.5 + 5j)
        # central moments of a w x h rectangle
        self.assertAlmostEqual(m.ixx, 3 * 2**3 / 12.0)   # w*h^3/12
        self.assertAlmostEqual(m.iyy, 2 * 3**3 / 12.0)   # h*w^3/12
        self.assertAlmostEqual(m.ixy, 0.0)
        i_min, i_max = m.principal_moments
        self.assertAlmostEqual(i_min, 3 * 2**3 / 12.0)
        self.assertAlmostEqual(i_max, 2 * 3**3 / 12.0)
        # long axis of a 3x2 rectangle is the x-axis
        self.assertAlmostEqual(abs(sin(m.principal_angle)), 0.0)

    def test_rectangle_clockwise(self):
        cw = area_moments(rect_path(1, 4, 3, 2, ccw=False))
        ccw = area_moments(rect_path(1, 4, 3, 2))
        self.assertAlmostEqual(cw.area, -ccw.area)
        self.assertAlmostEqual(cw.centroid, ccw.centroid)
        self.assertAlmostEqual(cw.ixx, -ccw.ixx)
        self.assertAlmostEqual(cw.iyy, -ccw.iyy)
        # principal moments are sign-normalized: identical either way
        self.assertAlmostEqual(cw.principal_moments[0],
                               ccw.principal_moments[0])
        self.assertAlmostEqual(cw.principal_moments[1],
                               ccw.principal_moments[1])

    def test_rotated_rectangle_principal_angle(self):
        # 4 x 1 rectangle rotated by 30 degrees about its centroid
        w, h, ang = 4.0, 1.0, radians(30)
        c, s = cos(ang), sin(ang)
        corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
        pts = [complex(x*c - y*s, x*s + y*c) + (7 + 3j) for x, y in corners]
        path = Path(*[Line(pts[i], pts[(i + 1) % 4]) for i in range(4)])
        m = area_moments(path)
        self.assertAlmostEqual(m.area, w*h)
        self.assertAlmostEqual(m.centroid.real, 7.0)
        self.assertAlmostEqual(m.centroid.imag, 3.0)
        # major principal axis should point along the rotated long axis
        # (axes are defined modulo pi)
        self.assertAlmostEqual(
            (m.principal_angle - ang + pi/2) % pi - pi/2, 0.0, places=7)
        i_min, i_max = m.principal_moments
        self.assertAlmostEqual(i_min, w * h**3 / 12.0)
        self.assertAlmostEqual(i_max, h * w**3 / 12.0)

    def test_triangle(self):
        # right triangle (0,0), (6,0), (0,3): area 9, centroid (2,1)
        path = Path(Line(0, 6), Line(6, 3j), Line(3j, 0))
        m = area_moments(path)
        self.assertAlmostEqual(m.area, 9.0)
        self.assertAlmostEqual(m.centroid, 2 + 1j)
        # central moments of a right triangle with legs b (x) and h (y)
        b, h = 6.0, 3.0
        self.assertAlmostEqual(m.ixx, b * h**3 / 36.0)
        self.assertAlmostEqual(m.iyy, h * b**3 / 36.0)
        self.assertAlmostEqual(m.ixy, -b**2 * h**2 / 72.0)

    def test_huge_coordinates(self):
        # the same 3x2 rectangle shifted by ~1e12: centroid and central
        # moments must survive the cancellation
        shift = 1.2e12 + 3.4e12j
        path = rect_path(1, 4, 3, 2)
        big = Path(*[Line(seg.start + shift, seg.end + shift)
                     for seg in path])
        m = area_moments(big)
        self.assertAlmostEqual(m.area, 6.0)
        self.assertAlmostEqual(m.centroid.real, 1.2e12 + 2.5, places=3)
        self.assertAlmostEqual(m.centroid.imag, 3.4e12 + 5.0, places=3)
        self.assertAlmostEqual(m.ixx, 3 * 2**3 / 12.0, places=8)
        self.assertAlmostEqual(m.iyy, 2 * 3**3 / 12.0, places=8)
        self.assertAlmostEqual(m.ixy, 0.0, places=8)


class TestAreaMomentsBeziers(unittest.TestCase):
    """Quadratic and cubic Bezier segments."""

    def test_quadratic_bezier_lune(self):
        # region between the parabola y = x^2 - 1 (a quadratic Bezier)
        # and the x-axis segment from (1,0) back to (-1,0)
        q = QuadraticBezier(-1 + 0j, 0 - 2j, 1 + 0j)   # y = x^2 - 1
        path = Path(q, Line(1 + 0j, -1 + 0j))
        m = area_moments(path)
        # area between chord and parabola: int_{-1}^{1} (1 - x^2) dx = 4/3
        self.assertAlmostEqual(m.area, 4.0 / 3.0)
        self.assertAlmostEqual(m.centroid.real, 0.0)
        # centroid of a parabolic spandrel: 2/5 of the height below chord
        self.assertAlmostEqual(m.centroid.imag, -2.0 / 5.0)
        # central ixx = int_{-1}^{1} int_{x^2-1}^{0} (y+2/5)^2 dy dx
        self.assertAlmostEqual(m.ixx, 16.0 / 175.0)
        self.assertAlmostEqual(m.ixy, 0.0)

    def test_cubic_bezier_circle_approximation(self):
        # a closed blob made of 4 cubic Beziers approximating the unit
        # circle (k = 4/3*tan(pi/8)); compare with the exact disk values
        k = 4.0 / 3.0 * (sqrt(2) - 1)
        segs = [
            CubicBezier(1 + 0j, 1 + k*1j, k + 1j, 0 + 1j),
            CubicBezier(0 + 1j, -k + 1j, -1 + k*1j, -1 + 0j),
            CubicBezier(-1 + 0j, -1 - k*1j, -k - 1j, 0 - 1j),
            CubicBezier(0 - 1j, k - 1j, 1 - k*1j, 1 + 0j),
        ]
        m = area_moments(Path(*segs))
        self.assertAlmostEqual(m.area, pi, delta=1e-3)
        self.assertAlmostEqual(m.centroid.real, 0.0, places=12)
        self.assertAlmostEqual(m.centroid.imag, 0.0, places=12)
        self.assertAlmostEqual(m.ixx, pi / 4.0, delta=1e-3)
        self.assertAlmostEqual(m.iyy, pi / 4.0, delta=1e-3)
        self.assertAlmostEqual(m.ixy, 0.0, places=12)

    def test_consistency_with_path_area(self):
        # area_moments().area must agree with the existing Path.area()
        path = parse_path(
            "M 0 0 C 1 2 3 2 4 0 C 5 -2 7 -2 8 0 "
            "Q 9 3 6 4 Q 3 5 0 3 Z"
        )
        m = area_moments(path)
        self.assertAlmostEqual(m.area, path.area(), places=9)


class TestAreaMomentsArcs(unittest.TestCase):
    """Arc segments: exact analytic values for circle and semicircle."""

    def test_circle(self):
        r = 2.5
        c = 3 + 4j
        path = Path(
            Arc(c + r, r + r*1j, 0, True, True, c - r),
            Arc(c - r, r + r*1j, 0, True, True, c + r),
        )
        m = area_moments(path)
        self.assertAlmostEqual(m.area, pi * r**2, places=9)
        self.assertAlmostEqual(m.centroid.real, 3.0, places=9)
        self.assertAlmostEqual(m.centroid.imag, 4.0, places=9)
        self.assertAlmostEqual(m.ixx, pi * r**4 / 4.0, places=7)
        self.assertAlmostEqual(m.iyy, pi * r**4 / 4.0, places=7)
        self.assertAlmostEqual(m.ixy, 0.0, places=9)
        i_min, i_max = m.principal_moments
        self.assertAlmostEqual(i_min, pi * r**4 / 4.0, places=7)
        self.assertAlmostEqual(i_max, pi * r**4 / 4.0, places=7)
        self.assertEqual(m.principal_angle, 0.0)  # degenerate -> 0.0

    def test_eccentric_semicircle(self):
        # semicircle of radius r centered at origin: an off-center
        # (eccentric) arc shape whose centroid sits at 4r/(3*pi) from
        # the diameter.  sweep=True from (-r,0) to (r,0) bulges toward
        # negative y (CCW from 180deg goes through 270deg).
        r = 3.0
        path = Path(
            Arc(-r + 0j, r + r*1j, 0, False, True, r + 0j),
            Line(r + 0j, -r + 0j),
        )
        m = area_moments(path)
        self.assertAlmostEqual(m.area, pi * r**2 / 2.0, places=9)
        self.assertAlmostEqual(m.centroid.real, 0.0, places=12)
        self.assertAlmostEqual(m.centroid.imag, -4 * r / (3 * pi), places=9)
        # central moments of a semicircle
        self.assertAlmostEqual(m.ixx, r**4 * (pi / 8.0 - 8.0 / (9 * pi)),
                               places=7)
        self.assertAlmostEqual(m.iyy, pi * r**4 / 8.0, places=7)
        self.assertAlmostEqual(m.ixy, 0.0, places=9)
        # the semicircle's long axis is horizontal
        self.assertAlmostEqual(abs(sin(m.principal_angle)), 0.0, places=9)

    def test_rotated_ellipse(self):
        # full ellipse rx=4, ry=1, rotated 45 degrees, off-origin center
        rx, ry, rot = 4.0, 1.0, 45.0
        c = 10 - 5j
        e0 = c + complex(rx*cos(radians(rot)), rx*sin(radians(rot)))
        e1 = c - complex(rx*cos(radians(rot)), rx*sin(radians(rot)))
        path = Path(
            Arc(e0, rx + ry*1j, rot, True, True, e1),
            Arc(e1, rx + ry*1j, rot, True, True, e0),
        )
        m = area_moments(path)
        self.assertAlmostEqual(m.area, pi * rx * ry, places=8)
        self.assertAlmostEqual(m.centroid.real, 10.0, places=8)
        self.assertAlmostEqual(m.centroid.imag, -5.0, places=8)
        i_min, i_max = m.principal_moments
        self.assertAlmostEqual(i_min, pi * rx * ry**3 / 4.0, places=6)
        self.assertAlmostEqual(i_max, pi * ry * rx**3 / 4.0, places=6)
        # major axis along the 45-degree direction (mod pi)
        self.assertAlmostEqual(
            (m.principal_angle - radians(rot) + pi/2) % pi - pi/2,
            0.0, places=7)

    def test_arc_error_tolerance_respected(self):
        # a loose tolerance must still be honored (and much faster
        # tolerances must keep working)
        r = 1.0
        path = Path(
            Arc(-r + 0j, r + r*1j, 0, False, True, r + 0j),
            Line(r + 0j, -r + 0j),
        )
        loose = area_moments(path, error=1e-3)
        tight = area_moments(path, error=1e-13)
        self.assertAlmostEqual(loose.area, pi / 2.0, delta=1e-2)
        self.assertAlmostEqual(tight.area, pi / 2.0, places=11)


class TestCompositePaths(unittest.TestCase):
    """Multiple closed subpaths: holes by winding direction."""

    def test_square_with_square_hole(self):
        outer = rect_path(0, 0, 10, 10, ccw=True)
        hole = rect_path(4, 4, 2, 2, ccw=False)   # opposite winding
        m = area_moments(Path(*(list(outer) + list(hole))))
        self.assertAlmostEqual(m.area, 100 - 4)
        self.assertAlmostEqual(m.centroid, 5 + 5j)
        self.assertAlmostEqual(m.ixx, (10 * 10**3 - 2 * 2**3) / 12.0)
        self.assertAlmostEqual(m.iyy, (10 * 10**3 - 2 * 2**3) / 12.0)
        self.assertAlmostEqual(m.ixy, 0.0)

    def test_same_winding_subpath_adds(self):
        # same winding as the outer contour: nonzero-winding fill rule
        # keeps the interior filled, so areas add
        outer = rect_path(0, 0, 10, 10, ccw=True)
        inner = rect_path(4, 4, 2, 2, ccw=True)
        m = area_moments(Path(*(list(outer) + list(inner))))
        self.assertAlmostEqual(m.area, 100 + 4)

    def test_eccentric_hole_shifts_centroid(self):
        # a hole punched off-center must pull the centroid away from it
        outer = rect_path(0, 0, 10, 10, ccw=True)
        hole = rect_path(6, 4, 2, 2, ccw=False)   # centered at (7, 5)
        m = area_moments(Path(*(list(outer) + list(hole))))
        self.assertAlmostEqual(m.area, 96.0)
        self.assertAlmostEqual(m.centroid.real, (500 - 4*7) / 96.0)
        self.assertAlmostEqual(m.centroid.imag, 5.0)
        self.assertTrue(m.centroid.real < 5.0)

    def test_circle_with_circular_hole(self):
        outer = Path(
            Arc(4 + 0j, 4 + 4j, 0, True, True, -4 + 0j),
            Arc(-4 + 0j, 4 + 4j, 0, True, True, 4 + 0j),
        )
        hole = Path(
            Arc(1 + 1j, 1 + 1j, 0, True, False, 1 + 3j),
            Arc(1 + 3j, 1 + 1j, 0, True, False, 1 + 1j),
        )
        m = area_moments(Path(*(list(outer) + list(hole))))
        self.assertAlmostEqual(m.area, pi * (16 - 1), places=8)
        # centroid of a disk of radius 4 with a unit-disk hole at (1, 2)
        self.assertAlmostEqual(m.centroid.real, -1.0 / 15.0, places=8)
        self.assertAlmostEqual(m.centroid.imag, -2.0 / 15.0, places=8)


class TestDegenerateAndInvalidInput(unittest.TestCase):

    def test_unclosed_path_raises(self):
        path = Path(Line(0, 1), Line(1, 1 + 1j))
        with self.assertRaises(ValueError):
            area_moments(path)

    def test_unclosed_subpath_raises(self):
        closed = rect_path(0, 0, 1, 1)
        open_seg = Line(5, 6)
        with self.assertRaises(ValueError):
            area_moments(Path(*(list(closed) + [open_seg])))

    def test_empty_path_raises(self):
        with self.assertRaises(ValueError):
            area_moments(Path())

    def test_zero_area_raises(self):
        # degenerate: out and back along the same line
        path = Path(Line(0, 1 + 1j), Line(1 + 1j, 0))
        with self.assertRaises(ValueError):
            area_moments(path)

    def test_self_intersecting_bowtie_is_winding_weighted(self):
        # bowtie with unequal lobes: signed area is the difference of
        # the two lobe areas (shoelace of the vertex sequence)
        pts = [0 + 0j, 4 + 4j, 4 + 0j, 0 + 2j]
        path = Path(*[Line(pts[i], pts[(i + 1) % 4]) for i in range(4)])
        m = area_moments(path)
        # shoelace: lobe1 (0,0),(4,4),(4,0) -> -8 ; lobe2 (0,0),(4,0),
        # (0,2) -> +4 ; total -4
        self.assertAlmostEqual(m.area, -4.0)
        self.assertTrue(np.isfinite(m.centroid.real))
        self.assertTrue(np.isfinite(m.centroid.imag))

    def test_self_intersecting_zero_area_raises(self):
        # symmetric bowtie: lobes cancel exactly -> zero signed area
        pts = [0 + 0j, 2 + 2j, 2 + 0j, 0 + 2j]
        path = Path(*[Line(pts[i], pts[(i + 1) % 4]) for i in range(4)])
        with self.assertRaises(ValueError):
            area_moments(path)

    def test_invalid_error_tolerance_raises(self):
        path = rect_path(0, 0, 1, 1)
        for bad in (0, -1e-12, float('nan'), float('inf'), None, 'x'):
            with self.assertRaises(ValueError):
                area_moments(path, error=bad)

    def test_path_method_and_namedtuple(self):
        m = rect_path(0, 0, 2, 1).area_moments()
        self.assertIsInstance(m, AreaMoments)
        self.assertAlmostEqual(m.area, 2.0)
        self.assertAlmostEqual(m.centroid, 1 + 0.5j)


if __name__ == '__main__':
    unittest.main()
