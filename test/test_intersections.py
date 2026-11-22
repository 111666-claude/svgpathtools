"""Tests for svgpathtools.intersections (geometric classification of
path intersections: crossing / tangent / endpoint / coincident)."""

import unittest

from svgpathtools import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                          classify_intersections,
                          CROSSING, TANGENT, ENDPOINT, COINCIDENT)


def kinds_of(report):
    """ multiset of (kind, rounded point) for point contacts """
    return sorted((i.kind, round(i.point.real, 6), round(i.point.imag, 6))
                  for i in report.points)


class TestCrossingClassification(unittest.TestCase):
    def test_hand_computed_line_crossing(self):
        # diagonals of the square [0,4]x[0,4] cross at 2+2j, t=0.5 on both
        p1 = Path(Line(0 + 0j, 4 + 4j))
        p2 = Path(Line(0 + 4j, 4 + 0j))
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.points), 1)
        self.assertEqual(len(report.coincidences), 0)
        inter = report.points[0]
        self.assertEqual(inter.kind, CROSSING)
        self.assertAlmostEqual(inter.point, 2 + 2j)
        T1, seg1, t1 = inter.side1
        T2, seg2, t2 = inter.side2
        self.assertAlmostEqual(t1, 0.5)
        self.assertAlmostEqual(t2, 0.5)
        self.assertAlmostEqual(T1, 0.5)
        self.assertAlmostEqual(T2, 0.5)
        # consistency with the classic format
        ((lT1, lseg1, lt1), (lT2, lseg2, lt2)) = report.legacy()[0]
        self.assertAlmostEqual(p1.point(lT1), 2 + 2j)
        self.assertAlmostEqual(p2.point(lT2), 2 + 2j)

    def test_crossing_bezier_line(self):
        # parabola y = x^2 (as a quadratic bezier) crossed by y = 1/2
        # at x = +-1/sqrt(2)
        parabola = QuadraticBezier(-1 + 1j, 0 - 1j, 1 + 1j)
        line = Line(-2 + 0.5j, 2 + 0.5j)
        report = classify_intersections(Path(parabola), Path(line))
        self.assertEqual(len(report.points), 2)
        for inter in report.points:
            self.assertEqual(inter.kind, CROSSING)
        xs = sorted(i.point.real for i in report.points)
        self.assertAlmostEqual(xs[0], -2 ** -0.5, places=5)
        self.assertAlmostEqual(xs[1], 2 ** -0.5, places=5)
        for i in report.points:
            self.assertAlmostEqual(i.point.imag, 0.5, places=6)

    def test_crossing_arc_line(self):
        # unit half-circle crossed by the x-axis at +-1
        arc = Arc(1 + 0j, 1 + 1j, 0, 0, 1, -1 + 0j)
        line = Line(-2 + 0j, 2 + 0j)
        report = classify_intersections(Path(arc), Path(line))
        self.assertEqual(len(report.points), 2)
        # both contacts are at arc endpoints -> ENDPOINT
        self.assertEqual({i.kind for i in report.points}, {ENDPOINT})
        # embed the arc in a closed path: now the contacts are crossings
        arc2 = Arc(-1 + 0j, 1 + 1j, 0, 0, 1, 1 + 0j)
        report = classify_intersections(Path(arc, arc2), Path(line))
        self.assertEqual(len(report.points), 2)
        self.assertEqual({i.kind for i in report.points}, {CROSSING})


class TestTangentClassification(unittest.TestCase):
    def test_hand_computed_tangent_line_to_parabola(self):
        # y = x^2 touched by y = 0 at the origin (double contact)
        parabola = QuadraticBezier(-1 + 1j, 0 - 1j, 1 + 1j)
        tangent_line = Line(-2 + 0j, 2 + 0j)
        report = classify_intersections(Path(parabola), Path(tangent_line))
        self.assertEqual(len(report.points), 1)
        inter = report.points[0]
        self.assertEqual(inter.kind, TANGENT)
        self.assertAlmostEqual(inter.point, 0 + 0j, places=5)

    def test_tangent_line_to_arc(self):
        # unit circle (two arcs) touched by y = 1 at 1j
        arc1 = Arc(1 + 0j, 1 + 1j, 0, 0, 1, -1 + 0j)
        arc2 = Arc(-1 + 0j, 1 + 1j, 0, 0, 1, 1 + 0j)
        circle = Path(arc1, arc2)
        line = Path(Line(-2 + 1j, 2 + 1j))
        report = circle.intersect_classified(line)
        self.assertEqual(len(report.points), 1)
        inter = report.points[0]
        self.assertEqual(inter.kind, TANGENT)
        self.assertAlmostEqual(inter.point, 0 + 1j, places=5)

    def test_tangent_arc_to_arc(self):
        # two unit circles centered at 0 and 2+0j, tangent at 1+0j;
        # both arcs pass through the tangent point in their interiors
        a1 = Arc(0 + 1j, 1 + 1j, 0, 0, 0, 0 - 1j)       # through 1+0j
        a2 = Arc(2 + 1j, 1 + 1j, 0, 0, 1, 2 - 1j)       # through 1+0j
        self.assertAlmostEqual(a1.center, 0 + 0j)
        self.assertAlmostEqual(a2.center, 2 + 0j)
        report = classify_intersections(Path(a1), Path(a2))
        tangents = [i for i in report.points if i.kind == TANGENT]
        self.assertTrue(any(abs(i.point - (1 + 0j)) < 1e-5
                            for i in tangents),
                        msg="expected tangency at 1+0j, got %r" % (report,))

    def test_tangent_reversal_at_joint_cusp(self):
        # two quadratics meeting at a cusp at 1+1j: incoming direction +1j,
        # outgoing direction -1j (tangent reverses at the joint)
        q1 = QuadraticBezier(0 + 0j, 1 + 0j, 1 + 1j)
        q2 = QuadraticBezier(1 + 1j, 1 + 0j, 2 + 0j)
        path = Path(q1, q2)
        # vertical line through the cusp point touches (parallel to both
        # one-sided tangents, which point in opposite directions: +1j/-1j)
        report = classify_intersections(path, Path(Line(1 - 1j, 1 + 3j)))
        cusp_hits = [i for i in report.points
                     if abs(i.point - (1 + 1j)) < 1e-5]
        self.assertEqual(len(cusp_hits), 1)
        self.assertEqual(cusp_hits[0].kind, TANGENT)
        # horizontal line through the cusp point crosses
        report = classify_intersections(path, Path(Line(-1 + 1j, 3 + 1j)))
        cusp_hits = [i for i in report.points
                     if abs(i.point - (1 + 1j)) < 1e-5]
        self.assertEqual(len(cusp_hits), 1)
        self.assertEqual(cusp_hits[0].kind, CROSSING)

    def test_cubic_cusp(self):
        # cubic with a cusp at t=0.5 (derivative vanishes): p0+p1 == p2+p3
        cusp_cubic = CubicBezier(0 + 0j, 1 + 0j, 1 + 1j, 0 - 1j)
        cusp_pt = cusp_cubic.point(0.5)  # 0.75 + 0.25j
        self.assertAlmostEqual(cusp_pt, 0.75 + 0.25j)
        # line through the cusp along the cusp axis (direction -1-1j)
        along = Line(cusp_pt - (1 + 1j), cusp_pt + (1 + 1j))
        report = classify_intersections(Path(cusp_cubic), Path(along))
        hits = [i for i in report.points if abs(i.point - cusp_pt) < 1e-4]
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0].kind, TANGENT)


class TestEndpointContacts(unittest.TestCase):
    def test_endpoint_touching_segment_interior(self):
        # endpoint of an open path touches the middle of another segment
        p1 = Path(Line(0 + 0j, 1 + 1j))
        p2 = Path(Line(0 + 1j, 2 + 1j))
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.points), 1)
        inter = report.points[0]
        self.assertEqual(inter.kind, ENDPOINT)
        self.assertAlmostEqual(inter.point, 1 + 1j)

    def test_endpoint_crossing_geometry_still_endpoint(self):
        # even with transversal geometry, a contact at a path endpoint is
        # one-sided and classified ENDPOINT
        p1 = Path(Line(0 + 0j, 2 + 0j))          # ends at 2+0j
        p2 = Path(Line(2 - 1j, 2 + 1j))          # crosses x-axis at 2+0j
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.points), 1)
        self.assertEqual(report.points[0].kind, ENDPOINT)

    def test_closed_path_joint_is_not_endpoint(self):
        # a joint inside a closed path is not an endpoint contact
        square = Path(Line(0 + 0j, 2 + 0j), Line(2 + 0j, 2 + 2j),
                      Line(2 + 2j, 0 + 2j), Line(0 + 2j, 0 + 0j))
        line = Path(Line(1 - 1j, 1 + 3j))
        report = square.intersect_classified(line)
        self.assertEqual(len(report.points), 2)
        self.assertEqual({i.kind for i in report.points}, {CROSSING})


class TestCoincidence(unittest.TestCase):
    def test_partially_overlapping_lines(self):
        p1 = Path(Line(0 + 0j, 4 + 0j))
        p2 = Path(Line(1 + 0j, 3 + 0j))
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.points), 0)
        self.assertEqual(len(report.coincidences), 1)
        co = report.coincidences[0]
        self.assertEqual(co.kind, COINCIDENT)
        self.assertAlmostEqual(co.start_point, 1 + 0j)
        self.assertAlmostEqual(co.end_point, 3 + 0j)
        self.assertAlmostEqual(co.t1_range[0], 0.25)
        self.assertAlmostEqual(co.t1_range[1], 0.75)
        self.assertAlmostEqual(co.t2_range[0], 0.0)
        self.assertAlmostEqual(co.t2_range[1], 1.0)

    def test_overlapping_lines_reversed_orientation(self):
        p1 = Path(Line(0 + 0j, 4 + 0j))
        p2 = Path(Line(3 + 0j, 1 + 0j))  # reversed
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.coincidences), 1)
        co = report.coincidences[0]
        self.assertAlmostEqual(co.start_point, 1 + 0j)
        self.assertAlmostEqual(co.end_point, 3 + 0j)
        # t2 range runs backwards because p2 is oppositely oriented
        self.assertAlmostEqual(co.t2_range[0], 1.0)
        self.assertAlmostEqual(co.t2_range[1], 0.0)

    def test_fully_coincident_paths(self):
        path = Path(Line(0 + 0j, 1 + 0j),
                    QuadraticBezier(1 + 0j, 2 + 1j, 3 + 0j))
        report = classify_intersections(path, path)
        self.assertEqual(len(report.points), 0)
        self.assertEqual(len(report.coincidences), 2)
        for co in report.coincidences:
            self.assertAlmostEqual(co.t1_range[0], 0.0)
            self.assertAlmostEqual(co.t1_range[1], 1.0)
        # the coincidence intervals cover the whole path parameter range
        T_ranges = sorted(co.T1_range for co in report.coincidences)
        self.assertAlmostEqual(T_ranges[0][0], 0.0)
        self.assertAlmostEqual(T_ranges[1][1], 1.0)

    def test_fully_coincident_reversed_path(self):
        path = Path(Line(0 + 0j, 1 + 0j),
                    QuadraticBezier(1 + 0j, 2 + 1j, 3 + 0j))
        report = classify_intersections(path, path.reversed())
        self.assertEqual(len(report.points), 0)
        self.assertEqual(len(report.coincidences), 2)

    def test_coincident_arcs(self):
        # two arcs on the unit circle overlapping from 1j to -1+0j
        a1 = Arc(1 + 0j, 1 + 1j, 0, 0, 1, -1 + 0j)   # upper half: 0..180 deg
        a2 = Arc(0 + 1j, 1 + 1j, 0, 0, 1, 0 - 1j)    # left half: 90..270 deg
        report = classify_intersections(Path(a1), Path(a2))
        self.assertEqual(len(report.coincidences), 1)
        co = report.coincidences[0]
        self.assertAlmostEqual(co.start_point, 0 + 1j, places=5)
        self.assertAlmostEqual(co.end_point, -1 + 0j, places=5)
        self.assertAlmostEqual(co.t1_range[0], 0.5, places=5)
        self.assertAlmostEqual(co.t1_range[1], 1.0, places=5)
        self.assertAlmostEqual(co.t2_range[0], 0.0, places=5)
        self.assertAlmostEqual(co.t2_range[1], 0.5, places=5)

    def test_identical_beziers_coincide(self):
        c = CubicBezier(0 + 0j, 1 + 2j, 3 + 2j, 4 + 0j)
        report = classify_intersections(Path(c), Path(c.reversed()))
        self.assertEqual(len(report.points), 0)
        self.assertEqual(len(report.coincidences), 1)
        co = report.coincidences[0]
        self.assertEqual(co.t1_range, (0.0, 1.0))
        self.assertEqual(co.t2_range, (1.0, 0.0))

    def test_collinear_bezier_overlapping_line(self):
        # a quadratic bezier that is actually a straight segment on the
        # x-axis from 0 to 2, overlapped by a line from 1 to 4
        bez = QuadraticBezier(0 + 0j, 1 + 0j, 2 + 0j)
        line = Line(1 + 0j, 4 + 0j)
        report = classify_intersections(Path(bez), Path(line))
        self.assertEqual(len(report.coincidences), 1)
        co = report.coincidences[0]
        self.assertAlmostEqual(co.start_point, 1 + 0j)
        self.assertAlmostEqual(co.end_point, 2 + 0j)


class TestDegenerateAndDeterminism(unittest.TestCase):
    def test_zero_length_segment_does_not_crash(self):
        # zero-length segment in the middle of a path
        p1 = Path(Line(0 + 0j, 0 + 0j), Line(0 + 0j, 0 + 2j))
        p2 = Path(Line(-1 + 1j, 1 + 1j))
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.points), 1)
        self.assertAlmostEqual(report.points[0].point, 0 + 1j)
        self.assertEqual(report.points[0].kind, CROSSING)

    def test_zero_length_segment_lying_on_other_path(self):
        p1 = Path(Line(0.5 + 0j, 0.5 + 0j))  # a single point
        p2 = Path(Line(0 + 0j, 1 + 0j))
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.points), 1)
        inter = report.points[0]
        self.assertAlmostEqual(inter.point, 0.5 + 0j)
        # a lone point has no tangent: classified as endpoint contact
        self.assertEqual(inter.kind, ENDPOINT)

    def test_zero_length_segment_off_path(self):
        p1 = Path(Line(0.5 + 1j, 0.5 + 1j))
        p2 = Path(Line(0 + 0j, 1 + 0j))
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.points), 0)
        self.assertEqual(len(report.coincidences), 0)

    def test_near_tangent_perturbation_is_deterministic(self):
        # y = x^2 vs the horizontal line y = eps: tangent for eps = 0,
        # two crossings for eps > 0, no contact for eps < 0
        parabola = Path(QuadraticBezier(-1 + 1j, 0 - 1j, 1 + 1j))

        report = classify_intersections(parabola, Path(Line(-2 + 0j, 2 + 0j)))
        self.assertEqual([i.kind for i in report.points], [TANGENT])

        eps = 1e-4
        report = classify_intersections(parabola,
                                        Path(Line(-2 + eps * 1j,
                                                  2 + eps * 1j)))
        self.assertEqual(len(report.points), 2)
        self.assertEqual({i.kind for i in report.points}, {CROSSING})

        report = classify_intersections(parabola,
                                        Path(Line(-2 - eps * 1j,
                                                  2 - eps * 1j)))
        self.assertEqual(len(report.points), 0)

    def test_near_tangent_threshold_parameter(self):
        # a line meeting the x-axis at a tiny angle: classified CROSSING
        # with the default threshold, TANGENT with a large threshold
        p1 = Path(Line(-1 + 0j, 1 + 0j))
        p2 = Path(Line(-1 - 1e-4 * 1j, 1 + 1e-4 * 1j))
        report = classify_intersections(p1, p2)
        self.assertEqual(report.points[0].kind, CROSSING)
        report = classify_intersections(p1, p2, tangency_tol=1e-3)
        self.assertEqual(report.points[0].kind, TANGENT)

    def test_joint_duplicates_are_merged(self):
        # a vertical line through the joint of a two-segment path: the
        # contact is realized once per adjacent segment but must be
        # reported as a single crossing
        p1 = Path(Line(0 + 0j, 1 + 0j), Line(1 + 0j, 2 + 0.5j))
        p2 = Path(Line(1 - 1j, 1 + 1j))
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report.points), 1)
        self.assertAlmostEqual(report.points[0].point, 1 + 0j)
        self.assertEqual(report.points[0].kind, CROSSING)
        # consistent with the classic API (which dedups joints as well)
        classic = p1.intersect(p2)
        self.assertEqual(len(classic), len(report.legacy()))
        self.assertAlmostEqual(p1.point(classic[0][0][0]), 1 + 0j)

    def test_segment_order_independence(self):
        # same geometry, segments in different order -> same contacts
        segs = [Line(0 + 0j, 2 + 0j),
                QuadraticBezier(2 + 0j, 3 + 1j, 4 + 0j),
                CubicBezier(4 + 0j, 5 - 1j, 6 + 1j, 7 + 0j)]
        path_a = Path(*segs)
        path_b = path_a.reversed()
        other = Path(Line(1 - 1j, 1 + 1j), Line(1 + 1j, 5 - 1j))
        rep_a = classify_intersections(path_a, other)
        rep_b = classify_intersections(path_b, other)
        self.assertEqual(kinds_of(rep_a), kinds_of(rep_b))
        # and with the other path reversed as well
        rep_c = classify_intersections(path_a, other.reversed())
        self.assertEqual(kinds_of(rep_a), kinds_of(rep_c))

    def test_repeatability(self):
        p1 = Path(Line(0 + 0j, 3 + 3j), QuadraticBezier(3 + 3j, 4 + 4j,
                                                        6 + 3j))
        p2 = Path(Arc(3 + 1j, 2 + 2j, 0, 0, 1, 3 + 5j),
                  Line(3 + 5j, 3 + 1j))
        rep1 = classify_intersections(p1, p2)
        rep2 = classify_intersections(p1, p2)
        self.assertEqual(kinds_of(rep1), kinds_of(rep2))
        self.assertEqual(len(rep1.coincidences), len(rep2.coincidences))


class TestCompatibilityWithClassicIntersect(unittest.TestCase):
    def test_legacy_format_matches_classic_intersect(self):
        # two cubics crossing twice, no joints involved: the classified
        # points must agree with the classic Path.intersect output
        c1 = CubicBezier(0 + 0j, 1 + 3j, 2 - 3j, 3 + 0j)
        c2 = CubicBezier(0 + 1j, 1 - 1j, 2 + 3j, 3 - 1j)
        p1, p2 = Path(c1), Path(c2)
        classic = p1.intersect(p2)
        report = p1.intersect_classified(p2)
        classic_pts = sorted((p1.point(T1) for (T1, _, _), _ in classic),
                             key=lambda z: (z.real, z.imag))
        new_pts = sorted((i.point for i in report.points),
                         key=lambda z: (z.real, z.imag))
        self.assertEqual(len(classic_pts), len(new_pts))
        for a, b in zip(classic_pts, new_pts):
            self.assertAlmostEqual(a, b, places=6)
        # side tuples use the classic (T, seg, t) format
        for (T1, seg1, t1), (T2, seg2, t2) in report.legacy():
            self.assertAlmostEqual(p1.point(T1), seg1.point(t1))
            self.assertAlmostEqual(p2.point(T2), seg2.point(t2))
            # the two sides agree up to the solver accuracy of the
            # underlying classic segment intersection routines
            self.assertAlmostEqual(seg1.point(t1), seg2.point(t2), places=5)

    def test_report_container_api(self):
        p1 = Path(Line(0 + 0j, 4 + 4j))
        p2 = Path(Line(0 + 4j, 4 + 0j))
        report = p1.intersect_classified(p2)
        self.assertEqual(len(report), 1)
        self.assertEqual(len(report.crossings), 1)
        self.assertEqual(report.tangencies, [])
        self.assertEqual(report.endpoint_contacts, [])
        self.assertEqual(report.kinds(), {CROSSING})
        self.assertEqual(len(list(report)), 1)
        self.assertIn('crossing=1', repr(report))


if __name__ == '__main__':
    unittest.main()
