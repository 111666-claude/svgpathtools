"""Tests for the conservative spatial index and the accelerated
``Path.intersect`` query."""

import os
import sys
import unittest
from random import Random

# Make the repo importable when this file is run directly under `python -I`
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svgpathtools import (Arc, CubicBezier, Line, Path,
                          QuadraticBezier, SegmentSpatialIndex)
from svgpathtools.spatial_index import segment_candidate_pairs


def brute_force_pairs(segments1, segments2, tol=1e-12):
    """Reference candidate-pair implementation: every bbox-overlapping
    pair, found by direct O(n*m) comparison with the same padding the
    index uses."""
    from svgpathtools.spatial_index import _padded_bbox, _bboxes_overlap
    boxes1 = [_padded_bbox(s, tol) for s in segments1]
    boxes2 = [_padded_bbox(s, tol) for s in segments2]
    return [(i, j) for i, b1 in enumerate(boxes1)
            for j, b2 in enumerate(boxes2)
            if _bboxes_overlap(b1, b2)]


def brute_force_intersect(path1, path2, justonemode=False, tol=1e-12):
    """Reference Path.intersect: the pre-optimization nested loop."""
    intersection_list = []
    for seg1 in path1:
        for seg2 in path2:
            if justonemode and intersection_list:
                return intersection_list[0]
            for t1, t2 in seg1.intersect(seg2, tol=tol):
                T1 = path1.t2T(seg1, t1)
                T2 = path2.t2T(seg2, t2)
                intersection_list.append(
                    ((T1, seg1, t1), (T2, seg2, t2)))
    if justonemode and intersection_list:
        return intersection_list[0]
    if intersection_list:
        pts = [item[0][1].point(item[0][2]) for item in intersection_list]
        indices2remove = set()
        for ind1 in range(len(pts)):
            for ind2 in range(ind1 + 1, len(pts)):
                if abs(pts[ind1] - pts[ind2]) < tol:
                    indices2remove.add(ind2)
        intersection_list = [inter for ind, inter in
                             enumerate(intersection_list)
                             if ind not in indices2remove]
    return intersection_list


def make_random_segments(rng, n, scale=100.0, include_arcs=False):
    """Build random non-Arc segments (Line/Quad/Cubic).

    Random arcs are excluded by default because the library's Arc-Anything
    exact intersection routine falls back to an expensive recursive solver
    (unrelated to the spatial index); arc preservation is covered by
    dedicated, well-formed test cases.
    """
    segs = []
    for _ in range(n):
        kind = rng.randint(0, 2)
        pts = [complex(rng.uniform(-scale, scale),
                       rng.uniform(-scale, scale)) for _ in range(4)]
        if kind == 0:
            segs.append(Line(pts[0], pts[1]))
        elif kind == 1:
            segs.append(QuadraticBezier(pts[0], pts[1], pts[2]))
        else:
            segs.append(CubicBezier(pts[0], pts[1], pts[2], pts[3]))
    if include_arcs:
        # Well-formed circular arcs (rotation 0, equal radii), which use
        # the library's fast algebraic Arc code paths.
        for _ in range(max(1, n // 10)):
            center = complex(rng.uniform(-scale, scale),
                             rng.uniform(-scale, scale))
            r = rng.uniform(scale * 0.1, scale)
            start = center + complex(r, 0)
            end = center + complex(0, r)
            segs.append(Arc(start, radius=complex(r, r), rotation=0.0,
                            large_arc=bool(rng.randint(0, 1)),
                            sweep=bool(rng.randint(0, 1)), end=end))
    return segs


class TestSegmentSpatialIndex(unittest.TestCase):
    def test_empty_paths(self):
        idx = SegmentSpatialIndex([])
        self.assertTrue(idx.degraded)
        self.assertEqual(idx.candidate_pairs(SegmentSpatialIndex([])), [])
        nonempty = SegmentSpatialIndex([Line(0, 1 + 1j)])
        self.assertEqual(idx.candidate_pairs(nonempty), [])
        self.assertEqual(nonempty.candidate_pairs(idx), [])
        self.assertEqual(Path().intersect(Path(Line(0, 1))), [])

    def test_candidate_pairs_match_brute_force_random(self):
        rng = Random(20261122)
        for trial in range(30):
            segs1 = make_random_segments(rng, rng.randint(1, 60))
            segs2 = make_random_segments(rng, rng.randint(1, 60))
            expected = brute_force_pairs(segs1, segs2)
            got = segment_candidate_pairs(segs1, segs2)
            self.assertEqual(got, expected,
                             msg='candidate mismatch on trial %d' % trial)
            # Ordering contract: ascending lexicographic.
            self.assertEqual(got, sorted(got))

    def test_fully_overlapping_bboxes_degrade_gracefully(self):
        # Every segment has the same bbox: a grid cannot screen anything
        # and must degrade instead of producing quadratic memory.
        segs = [Line(0, 1 + 1j) for _ in range(200)]
        idx = SegmentSpatialIndex(segs)
        self.assertTrue(idx.degraded)
        pairs = idx.candidate_pairs(SegmentSpatialIndex(
            [Line(0.5, 0.5 + 2j)]))
        self.assertEqual(len(pairs), 200)
        self.assertEqual(pairs, sorted(pairs))

    def test_index_is_conservative_never_misses_a_pair(self):
        # Dense, highly overlapping random geometry: even if the grid
        # degrades, no bbox-overlapping pair may be lost.
        rng = Random(42)
        segs1 = make_random_segments(rng, 150, scale=2.0)
        segs2 = make_random_segments(rng, 150, scale=2.0)
        got = set(segment_candidate_pairs(segs1, segs2))
        expected = set(brute_force_pairs(segs1, segs2))
        self.assertTrue(expected.issubset(got))
        self.assertEqual(got, expected)

    def test_tangent_endpoint_and_near_boundary_contacts_kept(self):
        # Two lines meeting exactly at an endpoint (shared bbox boundary).
        a = Line(0, 1 + 0j)
        b = Line(1 + 0j, 2 + 1j)
        pairs = segment_candidate_pairs([a], [b])
        self.assertEqual(pairs, [(0, 0)])

        # Tangent boxes touching on x only.
        c = Line(1 + 1e-13, 2 + 1e-13j)
        pairs = segment_candidate_pairs([a], [c])
        self.assertEqual(pairs, [(0, 0)])

        # Tangency of actual curves must survive to the exact routine.
        path1 = Path(Line(0, 2 + 0j))
        path2 = Path(Line(1 - 1j, 1 + 1j))
        self.assertEqual(len(path1.intersect(path2)), 1)

    def test_arc_intersections_preserved(self):
        l = Line(start=(-20 + 1j), end=(20 + 1j))
        a = Arc(start=(-10 + 0), radius=(10 + 10j), rotation=0.0,
                large_arc=True, sweep=False, end=(10 + 0j))
        path1 = Path(a)
        path2 = Path(l)
        got = path1.intersect(path2)
        expected = brute_force_intersect(path1, path2)
        self.assertEqual(got, expected)
        self.assertEqual(len(got), 2)

    def test_huge_coordinates(self):
        big = 1e15
        segs1 = [Line(complex(big, 0), complex(big + 10, 10)),
                 Line(complex(-big, -big), complex(-big + 1, -big + 1))]
        segs2 = [Line(complex(big + 1, -1), complex(big + 2, 11))]
        got = segment_candidate_pairs(segs1, segs2)
        self.assertEqual(got, brute_force_pairs(segs1, segs2))
        self.assertEqual(got, [(0, 0)])

        path1, path2 = Path(*segs1), Path(*segs2)
        self.assertEqual(path1.intersect(path2),
                         brute_force_intersect(path1, path2))

    def test_nonfinite_coordinates_degrade(self):
        segs = [Line(0, 1 + 1j),
                Line(float('inf'), float('inf') + 1j)]
        idx = SegmentSpatialIndex(segs)
        self.assertTrue(idx.degraded)
        # The infinite box cannot overlap the finite one, so no pair.
        self.assertEqual(idx.candidate_pairs(idx), [])
        # A finite pair in the same degraded index is still found.
        finite = SegmentSpatialIndex(
            [Line(0, 1 + 1j), Line(0.5 + 0.5j, 2 + 2j),
             Line(float('inf'), float('inf') + 1j)])
        self.assertTrue(finite.degraded)
        self.assertEqual(finite.candidate_pairs(finite), [(0, 1)])

    def test_thin_segment_does_not_explode_grid(self):
        # Regression test: a horizontal line has a zero-height bbox whose
        # only extent is the numerical padding.  Building an index and
        # querying it used to span billions of cells and hang; it must now
        # return promptly with the correct conservative pair set.
        arc = Arc(start=(-10 + 0), radius=(10 + 10j), rotation=0.0,
                  large_arc=True, sweep=False, end=(10 + 0j))
        thin = Line(start=(-20 + 1j), end=(20 + 1j))
        segs = [arc, thin]
        idx = SegmentSpatialIndex(segs)
        candidate = idx.candidate_pairs(SegmentSpatialIndex([thin]))
        self.assertEqual(candidate, brute_force_pairs(segs, [thin]))
        # Both boxes conservatively overlap the line (the arc genuinely
        # intersects it twice).
        self.assertEqual(candidate, [(0, 0), (1, 0)])
        # And the exact intersection must still be found.
        self.assertEqual(len(Path(arc).intersect(Path(thin))), 2)

    def test_thin_segment_forces_degradation_in_large_path(self):
        # When enough flat lines make the median bbox height padding-sized,
        # the grid must degrade rather than iterate an enormous cell range.
        flat = [Line(complex(k, 1.0), complex(k + 10, 1.0))
                for k in range(20)]
        flat.append(Line(0, 1e6 + 1e6j))
        idx = SegmentSpatialIndex(flat)
        self.assertTrue(idx.degraded)
        self.assertIn((20, 0),
                      idx.candidate_pairs(
                          SegmentSpatialIndex([Line(0, 1 + 1j)])))

    def test_mismatched_grid_scales_fall_back_safely(self):
        # Two indices with wildly different cell sizes: querying must not
        # materialize an enormous cell range.
        tiny_segs = [Line(complex(k * 1e-9, 0),
                          complex(k * 1e-9 + 1e-9, 1e-9))
                     for k in range(5)]
        huge_segs = [Line(0, 1e6 + 1e6j)]
        tiny = SegmentSpatialIndex(tiny_segs)
        huge = SegmentSpatialIndex(huge_segs)
        pairs = huge.candidate_pairs(tiny)
        self.assertEqual(pairs,
                         brute_force_pairs(huge_segs, tiny_segs))
        self.assertEqual(tiny.candidate_pairs(huge),
                         brute_force_pairs(tiny_segs, huge_segs))

    def test_disjoint_worlds_do_not_falsely_degrade(self):
        # A query box lying entirely outside the other grid's world must
        # produce no pairs without triggering the excessive-span fallback
        # through negative cell indices.
        left = SegmentSpatialIndex(
            [Line(complex(k, 0), complex(k + 1, 1)) for k in range(10)])
        far = SegmentSpatialIndex(
            [Line(complex(k + 1000, 0), complex(k + 1001, 1))
             for k in range(10)])
        self.assertFalse(left.degraded or far.degraded)
        self.assertEqual(left.candidate_pairs(far), [])
        self.assertEqual(far.candidate_pairs(left), [])
        self.assertEqual(
            segment_candidate_pairs(
                [Line(0, 1 + 1j)],
                [Line(complex(1000, 1000), complex(1001, 1001))]),
            [])

    def test_zero_length_and_point_geometry(self):
        # Zero-length lines have zero-size bboxes; indexing must not crash
        # and must not lose contacts.
        segs = [Line(1 + 1j, 1 + 1j), Line(1 + 1j, 2 + 2j)]
        idx = SegmentSpatialIndex(segs)
        pairs = idx.candidate_pairs(idx)
        self.assertIn((0, 1), pairs)

    def test_self_intersection_query(self):
        rng = Random(7)
        segs = make_random_segments(rng, 40)
        idx = SegmentSpatialIndex(segs)
        pairs = idx.candidate_pairs(idx)
        # Every pair unordered, no diagonal, no duplicates, sorted.
        self.assertEqual(pairs, sorted(set(pairs)))
        self.assertTrue(all(i < j for i, j in pairs))
        boxes = [idx.bboxes[k] for k in range(len(segs))]
        from svgpathtools.spatial_index import _bboxes_overlap
        expected = [(i, j) for i in range(len(segs))
                    for j in range(i + 1, len(segs))
                    if _bboxes_overlap(boxes[i], boxes[j])]
        self.assertEqual(pairs, expected)


class TestAcceleratedIntersect(unittest.TestCase):
    def test_matches_brute_force_random_paths(self):
        rng = Random(99)
        for trial in range(8):
            p1 = Path(*make_random_segments(rng, rng.randint(1, 15),
                                            scale=30))
            p2 = Path(*make_random_segments(rng, rng.randint(1, 15),
                                            scale=30))
            self.assertEqual(p1.intersect(p2),
                             brute_force_intersect(p1, p2),
                             msg='intersection mismatch trial %d' % trial)

    def test_matches_brute_force_with_arcs(self):
        # Arc paths use the exact same candidate screening; arcs exercise
        # the trigonometric bbox and the padded-grid degradation path.
        rng = Random(101)
        p1 = Path(*make_random_segments(rng, 12, scale=30,
                                        include_arcs=True))
        p2 = Path(*make_random_segments(rng, 12, scale=30,
                                        include_arcs=True))
        self.assertEqual(p1.intersect(p2),
                         brute_force_intersect(p1, p2))

    def test_justonemode_matches_brute_force(self):
        rng = Random(123)
        for _ in range(5):
            p1 = Path(*make_random_segments(rng, 15, scale=10))
            p2 = Path(*make_random_segments(rng, 15, scale=10))
            self.assertEqual(p1.intersect(p2, justonemode=True),
                             brute_force_intersect(
                                 p1, p2, justonemode=True))

    def test_justonemode_empty_and_nonintersecting(self):
        p1 = Path(Line(0, 1), Line(2, 3))
        p2 = Path(Line(10, 11))
        self.assertEqual(p1.intersect(p2, justonemode=True), [])
        self.assertEqual(Path().intersect(p2, justonemode=True), [])

    def test_segment_input_supported(self):
        p = Path(Line(0, 2 + 2j))
        seg = Line(0 + 2j, 2 + 0j)
        self.assertEqual(p.intersect(seg),
                         brute_force_intersect(p, Path(seg)))

    def test_sparse_corner_overlap_scenario(self):
        # Two large maps that only overlap in a small corner region.
        rng = Random(555)
        left = make_random_segments(rng, 300, scale=500)
        # Shift the second map far away, then add one segment bridging the
        # corner where the two maps meet.
        right = [Line(complex(s.start.real + 5000, s.start.imag + 5000),
                      complex(s.end.real + 5000, s.end.imag + 5000))
                 for s in make_random_segments(rng, 300, scale=500)]
        bridge = Line(complex(-1, -1), complex(5001, 5001))
        right.append(bridge)
        p1, p2 = Path(*left), Path(*right)
        got = p1.intersect(p2)
        expected = brute_force_intersect(p1, p2)
        self.assertEqual(got, expected)
        self.assertGreaterEqual(len(got), 1)
        # The screening should eliminate almost all 300*301 pairs.
        n_candidates = len(
            p1.spatial_index().candidate_pairs(p2.spatial_index()))
        self.assertLess(n_candidates, len(p1) * len(p2) // 10)


class TestIndexInvalidation(unittest.TestCase):
    def _path(self):
        return Path(Line(0, 1 + 1j), Line(2 + 2j, 3 + 3j),
                    Line(5 + 0j, 6 + 1j))

    def test_index_is_cached_and_reused(self):
        p = self._path()
        first = p.spatial_index()
        self.assertIs(first, p.spatial_index())
        # A second intersect call must not rebuild it.
        q = Path(Line(0 + 1j, 1 + 0j))
        p.intersect(q)
        self.assertIs(first, p.spatial_index())

    def test_setitem_invalidates(self):
        p = self._path()
        first = p.spatial_index()
        p[0] = Line(100 + 100j, 101 + 101j)
        self.assertIsNot(first, p.spatial_index())
        self.assertEqual(p.spatial_index().n, 3)

    def test_delitem_and_insert_invalidate(self):
        p = self._path()
        first = p.spatial_index()
        del p[1]
        self.assertIsNot(first, p.spatial_index())
        self.assertEqual(p.spatial_index().n, 2)
        first = p.spatial_index()
        p.insert(0, Line(-5, -4 + 1j))
        self.assertIsNot(first, p.spatial_index())
        self.assertEqual(p.spatial_index().n, 3)

    def test_append_extend_pop_invalidate(self):
        p = self._path()
        first = p.spatial_index()
        p.append(Line(10 + 10j, 11 + 11j))
        self.assertIsNot(first, p.spatial_index())
        first = p.spatial_index()
        p.pop()
        self.assertIsNot(first, p.spatial_index())
        first = p.spatial_index()
        p.extend([Line(0, 0 + 1j)])
        self.assertIsNot(first, p.spatial_index())

    def test_start_end_setters_invalidate(self):
        p = self._path()
        first = p.spatial_index()
        p.start = -1 - 1j
        self.assertIsNot(first, p.spatial_index())
        first = p.spatial_index()
        p.end = 20 + 20j
        self.assertIsNot(first, p.spatial_index())

    def test_results_correct_after_mutation(self):
        rng = Random(31)
        p1 = Path(*make_random_segments(rng, 12, scale=15))
        p2 = Path(*make_random_segments(rng, 12, scale=15))
        self.assertEqual(p1.intersect(p2), brute_force_intersect(p1, p2))
        p1.append(Line(complex(-30, -30), complex(30, 30)))
        del p2[0]
        p2[1] = QuadraticBezier(-20 + 20j, 0, 20 - 20j)
        self.assertEqual(p1.intersect(p2), brute_force_intersect(p1, p2))

    def test_manual_invalidation(self):
        p = self._path()
        first = p.spatial_index()
        p.invalidate_spatial_index()
        self.assertIsNot(first, p.spatial_index())


if __name__ == '__main__':
    unittest.main()
