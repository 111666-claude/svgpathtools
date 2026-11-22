# External dependencies
from __future__ import division, absolute_import, print_function
import unittest
from math import floor, pi

import numpy as np

# Internal dependencies
from svgpathtools import (
    Line, QuadraticBezier, CubicBezier, Arc, Path)
from svgpathtools import grid_coverage, GridCell, GridCoverageLimit


# A compact, independent reference implementation.  It recomputes occupied
# cells purely from dense point evaluation -- a different code path from the
# analytic event-walking implementation under test.  Interior samples use the
# strict half-open floor rule; endpoints use tolerance snapping because a
# curve ending on a grid line legitimately owns the high-side cell.
def _reference_cells(path, origin_x, origin_y, cell_w, cell_h,
                     tol=1e-8, n=50001):
    def raw_index(z):
        return (floor((z.real - origin_x) / cell_w),
                floor((z.imag - origin_y) / cell_h))

    def endpoint_index(z):
        x, y = z.real, z.imag
        i = floor((x - origin_x) / cell_w)
        j = floor((y - origin_y) / cell_h)
        ux = origin_x + (i + 1) * cell_w
        uy = origin_y + (j + 1) * cell_h
        if 0.0 <= ux - x <= tol:
            x = ux
        if 0.0 <= uy - y <= tol:
            y = uy
        return (floor((x - origin_x) / cell_w),
                floor((y - origin_y) / cell_h))

    cells = set()
    for seg in path:
        # Endpoints use tolerance snapping (their true position may be on a
        # grid line with evaluation noise pushing them an ULP low)...
        cells.add(endpoint_index(seg.point(0.0)))
        cells.add(endpoint_index(seg.point(1.0)))
        # ...while strictly interior samples use the raw floor rule, so a
        # near-boundary endpoint cannot also inject a noise-based cell.
        for k in range(1, n - 1):
            cells.add(raw_index(seg.point(k / (n - 1))))
    return cells


class TestGridCoverageBasic(unittest.TestCase):
    def test_hand_computed_diagonal(self):
        # Line (0,0)->(3,3) crosses the corner of every unit cell; under the
        # half-open convention the occupied cells are exactly the diagonal.
        path = Path(Line(0 + 0j, 3 + 3j))
        got = {(c.i, c.j) for c in grid_coverage(path)}
        self.assertEqual(got, {(0, 0), (1, 1), (2, 2), (3, 3)})

    def test_no_bbox_only_cells(self):
        # Two vertical lines ten units apart: columns 1..9 lie inside the
        # path's bbox but contain no center-line point and must be absent.
        path = Path(Line(0 + 0j, 0 + 10j), Line(10 + 0j, 10 + 10j))
        got = {(c.i, c.j) for c in grid_coverage(path)}
        occupied_cols = {i for i, _ in got}
        self.assertEqual(occupied_cols, {0, 10})

    def test_horizontal_line_single_row(self):
        path = Path(Line(0 + 0.5j, 3 + 0.5j))
        got = {(c.i, c.j) for c in grid_coverage(path)}
        self.assertEqual(got, {(0, 0), (1, 0), (2, 0), (3, 0)})

    def test_line_along_grid_line_owns_lower_cell(self):
        # A stroke exactly on y=1 belongs to row j=1 (the half-open cell
        # above the line), never to row j=0.
        path = Path(Line(0 + 1j, 3 + 1j))
        got = {(c.i, c.j) for c in grid_coverage(path)}
        self.assertEqual(got, {(0, 1), (1, 1), (2, 1), (3, 1)})

    def test_negative_coordinates(self):
        path = Path(Line(-1.5 - 1.5j, 0.5 + 0.5j))
        got = {(c.i, c.j) for c in grid_coverage(path)}
        self.assertEqual(got, {(-2, -2), (-1, -1), (0, 0)})

    def test_shifted_origin(self):
        path = Path(Line(0 + 0j, 4 + 4j))
        got = {(c.i, c.j) for c in grid_coverage(path, origin=1 + 1j)}
        self.assertEqual(got, {(-1, -1), (0, 0), (1, 1), (2, 2), (3, 3)})

    def test_rectangular_cells(self):
        path = Path(Line(0 + 0j, 5 + 2j))
        got = {(c.i, c.j) for c in grid_coverage(
            path, cell_size=2.0 + 0.5j)}
        ref = _reference_cells(path, 0, 0, 2.0, 0.5)
        self.assertEqual(got, ref)

    def test_single_segment_accepted(self):
        got = {(c.i, c.j) for c in grid_coverage(Line(0 + 0j, 2 + 0j))}
        self.assertEqual(got, {(0, 0), (1, 0), (2, 0)})

    def test_empty_path(self):
        self.assertEqual(grid_coverage(Path()), [])

    def test_zero_length_segment_is_single_cell(self):
        got = {(c.i, c.j) for c in
               grid_coverage(Path(Line(1.5 + 2.5j, 1.5 + 2.5j)))}
        self.assertEqual(got, {(1, 2)})

    def test_point_on_grid_corner_owns_high_side_cell(self):
        got = {(c.i, c.j) for c in
               grid_coverage(Path(Line(2 + 2j, 2 + 2j)))}
        self.assertEqual(got, {(2, 2)})

    def test_tolerance_snapping_is_configurable(self):
        # A line 1e-12 below the y=2 grid line.
        path = Path(Line(0 + (2 - 1e-12) * 1j, 3 + (2 - 1e-12) * 1j))
        # Default tolerance absorbs the noise into the boundary row j=2.
        self.assertEqual({c.j for c in grid_coverage(path)}, {2})
        # tol=0 performs exact arithmetic, leaving it in the lower row j=1.
        self.assertEqual({c.j for c in grid_coverage(path, tol=0.0)}, {1})
        # A line exactly on the grid line is high-side even with tol=0.
        exact = Path(Line(0 + 2j, 3 + 2j))
        self.assertEqual({c.j for c in grid_coverage(exact, tol=0.0)}, {2})


class TestGridCoverageCurves(unittest.TestCase):
    def test_quarter_circle_arc(self):
        # Arc from (3,0) to (0,3), radius 3, bulging toward the origin.
        arc = Arc(3 + 0j, 3 + 3j, 0, 0, 0, 0 + 3j)
        got = {(c.i, c.j) for c in grid_coverage(arc)}
        expected = {(3, 0), (2, 0), (1, 0), (0, 0),
                    (0, 1), (0, 2), (0, 3)}
        self.assertEqual(got, expected)
        # The cell (1, 1) is inside the arc's bbox but never touched.
        self.assertNotIn((1, 1), got)

    def test_s_shaped_cubic(self):
        curve = CubicBezier(0 + 0j, 3 + 0j, 0 + 3j, 3 + 3j)
        got = {(c.i, c.j) for c in grid_coverage(curve)}
        expected = {(0, 0), (1, 0), (1, 1), (1, 2),
                    (2, 2), (2, 3), (3, 3)}
        self.assertEqual(got, expected)

    def test_quadratic_bezier(self):
        curve = QuadraticBezier(0 + 0j, 3 + 0j, 3 + 3j)
        got = {(c.i, c.j) for c in grid_coverage(curve)}
        ref = _reference_cells(Path(curve), 0, 0, 1, 1)
        self.assertEqual(got, ref)

    def test_tangency_on_grid_line(self):
        # A curve that touches y=1 and turns back must still report the
        # high-side cell at the contact point.
        curve = CubicBezier(0 + 1j, 1.5 + 1j, 1.5 + 1j, 3 + 1j)
        got = {(c.i, c.j) for c in grid_coverage(curve)}
        ref = _reference_cells(Path(curve), 0, 0, 1, 1)
        self.assertEqual(got, ref)
        self.assertTrue(all(j == 1 for _, j in got))

    def test_self_intersecting_path(self):
        path = Path(
            CubicBezier(0 + 0j, 4 + 4j, -1 + 4j, 3 + 0j),
            Line(3 + 0j, 0 + 3j))
        got = {(c.i, c.j) for c in grid_coverage(path)}
        ref = _reference_cells(path, 0, 0, 1, 1)
        self.assertEqual(got, ref)

    def test_arc_both_sweep_directions(self):
        for sweep in (False, True):
            arc = Arc(2 + 0j, 3 + 2j, 35, 1, sweep, -2 + 1j)
            got = {(c.i, c.j) for c in grid_coverage(arc)}
            ref = _reference_cells(Path(arc), 0, 0, 1, 1)
            self.assertEqual(got, ref)


class TestGridCoverageSegmentsAndOrder(unittest.TestCase):
    def setUp(self):
        self.path = Path(
            CubicBezier(0 + 0j, 5 + 5j, -2 + 3j, 4 + 0j),
            Line(4 + 0j, 0 + 4j),
            Arc(0 + 4j, 2 + 2j, 0, 0, 1, -2 + 2j))

    def test_stable_sorted_order(self):
        result = grid_coverage(self.path)
        coords = [(c.i, c.j) for c in result]
        self.assertEqual(coords, sorted(coords, key=lambda c: (c[1], c[0])))

    def test_repeatable(self):
        first = grid_coverage(self.path)
        second = grid_coverage(self.path)
        self.assertEqual(first, second)

    def test_segment_indices(self):
        result = grid_coverage(self.path)
        for cell in result:
            self.assertEqual(cell.segments, tuple(sorted(cell.segments)))
            # every listed segment genuinely passes through the cell
            for idx in cell.segments:
                seg = self.path[idx]
                hit = any(
                    (floor((seg.point(k / 5000.0).real)),
                     floor((seg.point(k / 5000.0).imag))) == (cell.i, cell.j)
                    for k in range(5001))
                self.assertTrue(hit, (cell, idx))

    def test_shared_cell_lists_both_segments(self):
        result = grid_coverage(self.path)
        shared = [c for c in result if len(c.segments) > 1]
        self.assertTrue(shared)
        for cell in shared:
            self.assertEqual(cell.i, cell[0])
            self.assertEqual(cell.j, cell[1])
            self.assertEqual(cell[2], cell.segments)

    def test_grid_cell_equality_with_plain_tuple(self):
        cell = GridCell(2, -3, (0, 2))
        self.assertEqual(cell, (2, -3))
        self.assertEqual(hash(cell), hash((2, -3)))
        self.assertEqual(repr(cell),
                         'GridCell(i=2, j=-3, segments=(0, 2))')
        self.assertEqual(cell.i, 2)
        self.assertEqual(cell.j, -3)
        self.assertEqual(cell.segments, (0, 2))


class TestGridCoverageErrors(unittest.TestCase):
    def setUp(self):
        self.path = Path(Line(0 + 0j, 3 + 3j))

    def test_zero_cell_size(self):
        with self.assertRaises(ValueError):
            grid_coverage(self.path, cell_size=0 + 1j)
        with self.assertRaises(ValueError):
            grid_coverage(self.path, cell_size=1 + 0j)

    def test_negative_cell_size(self):
        with self.assertRaises(ValueError):
            grid_coverage(self.path, cell_size=-1 + 1j)

    def test_non_finite_origin_and_cells(self):
        with self.assertRaises(ValueError):
            grid_coverage(self.path, origin=float('nan') + 0j)
        with self.assertRaises(ValueError):
            grid_coverage(self.path, cell_size=float('inf') + 1j)

    def test_bad_tolerance(self):
        with self.assertRaises(ValueError):
            grid_coverage(self.path, tol=-1e-9)
        with self.assertRaises(ValueError):
            grid_coverage(self.path, tol=float('nan'))

    def test_non_finite_geometry(self):
        with self.assertRaises(ValueError):
            grid_coverage(Path(Line(0 + 0j, float('nan') + 1j)))
        with self.assertRaises(ValueError):
            grid_coverage(Path(Line(0 + 0j, float('inf') + 1j)))

    def test_bad_max_cells(self):
        with self.assertRaises(ValueError):
            grid_coverage(self.path, max_cells=0)
        with self.assertRaises(ValueError):
            grid_coverage(self.path, max_cells=1.5)

    def test_max_cells_cap_long_curve(self):
        with self.assertRaises(GridCoverageLimit):
            grid_coverage(Path(Line(0 + 0j, 1000 + 0j)), max_cells=10)

    def test_max_cells_cap_tiny_cells(self):
        with self.assertRaises(GridCoverageLimit):
            grid_coverage(self.path, cell_size=1e-9 + 1e-9j, max_cells=1000)

    def test_extremely_long_curve_is_bounded(self):
        # Must fail fast on the up-front budget, never attempt a billion
        # crossing events.
        with self.assertRaises(GridCoverageLimit):
            grid_coverage(Path(Line(0 + 0j, 1e9 + 0j)), max_cells=1000)

    def test_wrong_path_type(self):
        with self.assertRaises(TypeError):
            grid_coverage("not a path")


class TestGridCoverageMatchesReference(unittest.TestCase):
    """Cross-check the analytic result against the independent dense
    reference on a fixed battery covering every segment type."""

    def _assert_matches(self, path, origin=0j, cell_w=1.0, cell_h=1.0):
        got = {(c.i, c.j) for c in grid_coverage(
            path, origin=origin, cell_size=cell_w + cell_h * 1j)}
        ref = _reference_cells(
            path, origin.real, origin.imag, cell_w, cell_h)
        self.assertEqual(got, ref)

    def test_mixed_river_path(self):
        river = Path(
            Line(0 + 0j, 10 + 1j),
            CubicBezier(10 + 1j, 12 - 4j, 14 + 6j, 16 + 1j),
            Arc(16 + 1j, 2 + 2j, 0, 0, 1, 20 + 1j),
            Line(20 + 1j, 30 + 0j))
        self._assert_matches(river)

    def test_rotated_ellipse_arcs(self):
        arc = Arc(-4 + 2j, 5 + 3j, 60, 1, 0, 4 - 3j)
        self._assert_matches(Path(arc))

    def test_shifted_origin_and_rect_cells(self):
        path = Path(
            QuadraticBezier(-3 - 2j, 2 + 4j, 5 - 1j),
            Arc(5 - 1j, 2 + 3j, -25, 0, 1, 8 + 2j))
        self._assert_matches(path, origin=-2 + 0.5j, cell_w=1.5, cell_h=0.8)

    def test_full_circle_from_two_arcs(self):
        upper = Arc(-2 + 0j, 2 + 2j, 0, 0, 1, 2 + 0j)
        lower = Arc(2 + 0j, 2 + 2j, 0, 0, 1, -2 + 0j)
        self._assert_matches(Path(upper, lower))


if __name__ == '__main__':
    unittest.main()
