"""Tests for canonical (stable, quantized) d-string output."""
from __future__ import absolute_import, division, print_function

import unittest
import warnings

import numpy as np

from svgpathtools import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                          parse_path, canonical_path)
from svgpathtools.canonical import (DEFAULT_ERROR, _digits_for_error,
                                    _max_deviation)


class TestCanonicalSpelling(unittest.TestCase):
    """Requirement 1: equivalent command spellings normalize to one rule."""

    def test_relative_absolute_equivalence(self):
        variants = [
            'M 100.0000001,100 l 200,0 L 200,300 L 100,100',
            'm100 100 l200 0 l-100 200 L100 100',
            'M 100 100 L 300 100 L 200 300 Z',
        ]
        outputs = {canonical_path(parse_path(d)) for d in variants}
        self.assertEqual(len(outputs), 1)
        (out,) = tuple(outputs)
        # only absolute, fully-spelled commands are used
        for letter in 'mlhvcstqaz':
            self.assertNotIn(letter, out)
        for letter in 'HVSTZ':
            self.assertNotIn(letter, out)

    def test_hv_expand_to_l(self):
        out = canonical_path(parse_path('M10,20 H50 V60 H10 Z'))
        self.assertEqual(
            out,
            'M 10,20 L 50,20 L 50,60 L 10,60 L 10,20')

    def test_s_expands_to_c_and_t_expands_to_q(self):
        out = canonical_path(parse_path(
            'M100,200 C100,100 250,100 250,200 S400,300 400,200'))
        self.assertIn('C', out)
        self.assertNotIn('S', out)
        # the reflected first control point is spelled out explicitly
        self.assertEqual(parse_path(out), parse_path(
            'M100,200 C100,100 250,100 250,200 S400,300 400,200'))

        out = canonical_path(parse_path('M200,300 Q400,50 600,300 T1000,300'))
        self.assertIn('Q', out)
        self.assertNotIn('T', out)
        self.assertEqual(parse_path(out),
                         parse_path('M200,300 Q400,50 600,300 T1000,300'))

    def test_output_is_reparseable(self):
        samples = [
            'M 100,100 L 300,100 L 200,300 Z',
            'M100,200 C100,100 400,100 400,200',
            'M 0,0 L 50,20 M 100,100 L 300,100 L 200,300 Z',
            'M600,350 l 50,-25 a25,50 -30 0,1 50,-25',
        ]
        for d in samples:
            canonical = canonical_path(parse_path(d))
            # must parse without error and keep the same geometry up to
            # the bound (rotation may be normalized to [0, 360))
            reparsed = parse_path(canonical)
            original = parse_path(d)
            self.assertEqual(len(reparsed), len(original))
            for seg_a, seg_b in zip(original, reparsed):
                self.assertEqual(type(seg_a), type(seg_b))
                self.assertLessEqual(
                    _max_deviation(seg_a, seg_b), 1.5 * DEFAULT_ERROR)

    def test_method_matches_function(self):
        p = parse_path('M10,20 h30 v40 z')
        self.assertEqual(p.canonical_d(), canonical_path(p))


class TestCanonicalStability(unittest.TestCase):
    """Requirement 2: direction, endpoints, topology and byte-stable output."""

    def test_idempotent_bytes(self):
        samples = [
            'M 100.25,100.75 L 300,100 L 200,300 Z',
            'M100,200 C100.125,100.5 250,100 250,200 S400,300 400,200',
            'M200,300 Q400,50 600,300 T1000,300',
            'M600,350 L650,325 A27.9508497,27.9508497 -30 0,1 700,300',
            'M0,0 L50,20 M100,100 L300,100 L200,300 Z',
        ]
        for d in samples:
            once = canonical_path(parse_path(d))
            twice = canonical_path(parse_path(once))
            thrice = canonical_path(parse_path(twice))
            self.assertEqual(once, twice)
            self.assertEqual(twice, thrice)

    def test_direction_and_endpoints_preserved(self):
        # reversed orientation must not normalize to the same key
        fwd = parse_path('M0,0 L10,0 L10,10 L0,10 Z')
        rev = fwd.reversed()
        self.assertNotEqual(canonical_path(fwd), canonical_path(rev))

        # endpoints survive within the error bound
        p = parse_path('M1.2345678,2.3456789 L9.8765432,8.7654321')
        q = parse_path(canonical_path(p))
        self.assertLess(abs(p.start - q.start), DEFAULT_ERROR)
        self.assertLess(abs(p.end - q.end), DEFAULT_ERROR)

    def test_multi_subpath_topology(self):
        d = 'M0,0 L50,20 M100,100 L300,100 L200,300 Z M200,200 L250,250'
        out = canonical_path(parse_path(d))
        self.assertEqual(out.count('M'), 3)
        reparsed = parse_path(out)
        original = parse_path(d)
        # 1 segment in subpath 1, 3 in subpath 2 (Z adds the closing one),
        # 1 in subpath 3
        self.assertEqual(len(reparsed), 5)
        self.assertEqual(len(reparsed), len(original))
        for a, b in zip(reparsed, original):
            self.assertEqual(type(a), type(b))
            self.assertLess(abs(a.start - b.start), DEFAULT_ERROR)
            self.assertLess(abs(a.end - b.end), DEFAULT_ERROR)
        # the three moveto-induced breaks survive as genuine discontinuities
        original_breaks = {
            i for i in range(1, len(original))
            if original[i].start != original[i - 1].end
        }
        self.assertEqual(original_breaks, {1, 4})
        for i in original_breaks:
            self.assertNotEqual(reparsed[i].start, reparsed[i - 1].end)
        # every other joint stays welded
        for i in range(1, len(reparsed)):
            if i not in original_breaks:
                self.assertEqual(reparsed[i].start, reparsed[i - 1].end)

    def test_closed_path_uses_explicit_closing_line(self):
        out = canonical_path(parse_path('M10,10 L50,10 L50,50 Z'))
        self.assertNotIn('Z', out)
        reparsed = parse_path(out)
        self.assertEqual(reparsed[-1], Line(50 + 50j, 10 + 10j))
        self.assertTrue(reparsed.isclosed())

    def test_all_segment_types_round_trip(self):
        p = Path(
            Line(0 + 0j, 10 + 0j),
            QuadraticBezier(10 + 0j, 15 - 5j, 20 + 0j),
            CubicBezier(20 + 0j, 25 + 5j, 30 - 5j, 35 + 0j),
            Arc(35 + 0j, 5 + 7j, 30, 0, 1, 45 + 10j),
        )
        out = canonical_path(p)
        reparsed = parse_path(out)
        self.assertEqual(len(reparsed), 4)
        self.assertIsInstance(reparsed[0], Line)
        self.assertIsInstance(reparsed[1], QuadraticBezier)
        self.assertIsInstance(reparsed[2], CubicBezier)
        self.assertIsInstance(reparsed[3], Arc)
        arc = reparsed[3]
        self.assertEqual(arc.large_arc, False)
        self.assertEqual(arc.sweep, True)
        self.assertAlmostEqual(arc.rotation, 30, places=5)


class TestCanonicalNumbers(unittest.TestCase):
    """Requirement 3: number formats, zeros, non-finite values."""

    def test_no_scientific_notation(self):
        out = canonical_path(
            parse_path('M-3.4e38,3.4E+38 L-3.4E-38,3.4e-38'))
        self.assertNotIn('e', out.lower())
        # still re-parseable and stable
        self.assertEqual(canonical_path(parse_path(out)), out)

    def test_negative_zero_collapsed(self):
        out = canonical_path(parse_path('M-0,-0 L-0.00000001,-0.00000001'))
        for token in out.replace(',', ' ').split():
            if token not in ('M', 'L'):
                self.assertNotIn(token, ('-0', '-0.0'))
        self.assertTrue(out.startswith('M 0,0'))

    def test_non_finite_coordinates_rejected(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            bad_nan_arc = Path(
                Arc(0j, 1 + 1j, float('nan'), 0, 1, 1 + 0j))
        bad_paths = [
            Path(Line(0 + 0j, float('nan') + 1j)),
            Path(Line(0 + 0j, float('inf') + 0j)),
            Path(CubicBezier(0j, float('-inf') + 0j, 1j, 2j)),
            bad_nan_arc,
        ]
        for p in bad_paths:
            with self.assertRaises(ValueError):
                canonical_path(p)

    def test_invalid_error_rejected(self):
        p = parse_path('M0,0 L1,1')
        for bad in (0, -1.0, float('nan'), float('inf')):
            with self.assertRaises(ValueError):
                canonical_path(p, error=bad)

    def test_digit_rule(self):
        # step size 0.5*10^-d must be <= error, hence d = ceil(-log10(2e))
        self.assertEqual(_digits_for_error(1e-6), 6)
        self.assertEqual(_digits_for_error(1e-3), 3)
        self.assertEqual(_digits_for_error(0.0005), 3)  # step 5e-4 == error
        self.assertEqual(_digits_for_error(0.0004999), 4)
        self.assertEqual(_digits_for_error(1.0), 0)


class TestCanonicalErrorBound(unittest.TestCase):
    """Requirement 3/4: quantization stays inside the declared bound."""

    def test_coordinate_error_guarantee(self):
        rng = np.random.default_rng(42)
        for error in (1e-2, 1e-4, 1e-6):
            for _ in range(40):
                vals = rng.uniform(-1000, 1000, size=8)
                p = Path(CubicBezier(complex(vals[0], vals[1]),
                                     complex(vals[2], vals[3]),
                                     complex(vals[4], vals[5]),
                                     complex(vals[6], vals[7])))
                q = parse_path(canonical_path(p, error=error))
                for i in range(25):
                    t = i / 24
                    # each axis is within error, so Euclidean distance <= sqrt2*e
                    self.assertLessEqual(
                        abs(p[0].point(t) - q[0].point(t)),
                        np.sqrt(2) * error * 1.01)

    def test_near_but_unequal_shapes_stay_distinct(self):
        # farther apart than the bound -> different keys
        a = canonical_path(parse_path('M0,0 L10,0'), error=1e-6)
        b = canonical_path(parse_path('M0,0 L10.0005,0'), error=1e-6)
        self.assertNotEqual(a, b)

        # within the bound -> same key, by design of the tolerance
        c = canonical_path(parse_path('M0,0 L10.0000001,0'), error=1e-6)
        self.assertEqual(a, c)

    def test_tighter_error_keeps_distinction(self):
        # the 5e-4 gap that merged at 1e-6 must not merge at 1e-7
        a = canonical_path(parse_path('M0,0 L10,0'), error=1e-7)
        b = canonical_path(parse_path('M0,0 L10.0005,0'), error=1e-7)
        self.assertNotEqual(a, b)


class TestCanonicalArcs(unittest.TestCase):
    """Arc-specific edge strategies."""

    def test_zero_radius_becomes_line(self):
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            p = parse_path('M0,0 a0,5 0 0,1 10,10')
        out = canonical_path(p)
        self.assertNotIn('A', out)
        self.assertEqual(canonical_path(parse_path(out)), out)

    def test_sub_quantum_radius_becomes_bounded_line(self):
        p = Path(Arc(0j, 1e-7 + 2e-7j, 0, 0, 1, 2e-7 + 0j))
        out = canonical_path(p, error=1e-6)
        self.assertNotIn('A', out)
        reparsed = parse_path(out)
        self.assertLessEqual(abs(p.start - reparsed.start), 1e-6)
        self.assertLessEqual(abs(p.end - reparsed.end), 1e-6)
        self.assertEqual(canonical_path(reparsed), out)

    def test_nearly_closed_arc_refines_instead_of_collapsing(self):
        # endpoints 1e-9 apart; at 6 digits they collapse, which would turn
        # the large arc into a straight line 2 units away from the true curve
        seg = Arc(0j, 1 + 1j, 0, 1, 1, complex(1e-9, 0))
        out = canonical_path(Path(seg), error=1e-6)
        self.assertIn('A', out)
        reparsed = parse_path(out)
        self.assertLessEqual(
            _max_deviation(seg, reparsed[0]), 1.5 * 1e-6)
        self.assertEqual(canonical_path(reparsed), out)

    def test_large_arc_and_sweep_flags_distinguish_shapes(self):
        # well-separated endpoints so all four flag combinations are viable
        endpoint = 0.5 + 0.5j
        variants = [
            Arc(0j, 1 + 1j, 0, large, sweep, endpoint)
            for large in (False, True)
            for sweep in (False, True)
        ]
        keys = {canonical_path(Path(a)) for a in variants}
        self.assertEqual(len(keys), 4)

    def test_nearly_coincident_endpoints_keep_viable_flags(self):
        # at near-coincident endpoints the small-arc variants degenerate to
        # a point while the large-arc variants must refine and survive;
        # the two surviving sweeps still produce distinct keys
        endpoint = complex(1e-9, 0)
        large_variants = [
            Arc(0j, 1 + 1j, 0, True, sweep, endpoint)
            for sweep in (False, True)
        ]
        keys = {canonical_path(Path(a)) for a in large_variants}
        self.assertEqual(len(keys), 2)
        for key in keys:
            self.assertIn('A', key)

    def test_rotation_normalized_onto_0_360(self):
        equivalent = [
            Arc(0j, 5 + 5j, 30.0, 0, 1, 8 + 2j),
            Arc(0j, 5 + 5j, 390.0, 0, 1, 8 + 2j),
            Arc(0j, 5 + 5j, -330.0, 0, 1, 8 + 2j),
        ]
        keys = {canonical_path(Path(a)) for a in equivalent}
        self.assertEqual(len(keys), 1)

        distinct = Arc(0j, 1000 + 1000j, 30.0005, 0, 1, 1000 + 0j)
        reference = Arc(0j, 1000 + 1000j, 30.0, 0, 1, 1000 + 0j)
        self.assertNotEqual(canonical_path(Path(distinct)),
                            canonical_path(Path(reference)))

    def test_arc_geometry_within_bound(self):
        seg = Arc(0j, 0.01 + 0.02j, 30, 1, 0, 0.03 + 0.01j)
        out = canonical_path(Path(seg), error=1e-6)
        reparsed = parse_path(out)
        self.assertLessEqual(
            _max_deviation(seg, reparsed[0]), 1.5 * 1e-6)


class TestCanonicalDegenerateSegments(unittest.TestCase):
    """Requirement 3: degenerate segments after quantization."""

    def test_zero_length_line_is_retained_and_stable(self):
        # a genuine zero-length segment is topology information and must not
        # be silently dropped
        p = parse_path('M10,10 L10,10 L20,20')
        out = canonical_path(p)
        self.assertEqual(out.count('L'), 2)
        self.assertEqual(canonical_path(parse_path(out)), out)

    def test_degenerate_bezier_is_retained_and_bounded(self):
        p = Path(CubicBezier(0j, 0.0000001 + 0j, 0j, 0.0000002 + 0j))
        out = canonical_path(p, error=1e-6)
        reparsed = parse_path(out)
        self.assertIsInstance(reparsed[0], CubicBezier)
        for a, b in zip(p[0].bpoints(), reparsed[0].bpoints()):
            self.assertLessEqual(abs(a - b), 1.5e-6)
        self.assertEqual(canonical_path(reparsed), out)

    def test_empty_path(self):
        self.assertEqual(canonical_path(Path()), '')

    def test_unrepresentable_sub_quantum_arc_raises(self):
        # An endpoint below the fixed-point numeric floor would render with
        # coincident endpoints, which is not a valid SVG Arc; the library must
        # refuse rather than emit an unparseable or colliding key.
        seg = Arc(0j, 5 + 5j, 0, 1, 1, complex(4.94e-324, 0))
        with self.assertRaises(ValueError):
            canonical_path(Path(seg), error=1e-6)

    def test_full_fidelity_fallback_is_byte_stable(self):
        # Endpoint visible at 17 digits but below the 6-digit quantum, while
        # the true arc spans far more than the bound: refinement exhausts and
        # the full-fidelity fallback fires.
        seg = Arc(0j, 5 + 5j, 0, 1, 1, complex(1e-16, 0))
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            d = canonical_path(Path(seg), error=1e-6, max_refine_steps=6)
            self.assertTrue(
                any('fidelity' in str(w.message) for w in caught))
        self.assertIn('A', d)
        # re-parse and re-canonicalize repeatedly -> identical bytes
        d2 = canonical_path(parse_path(d), error=1e-6, max_refine_steps=6)
        d3 = canonical_path(parse_path(d2), error=1e-6, max_refine_steps=6)
        self.assertEqual(d, d2)
        self.assertEqual(d2, d3)


class TestCanonicalEquivalenceSamples(unittest.TestCase):
    """Requirement 4: cross-check on a broad set of equivalent spellings."""

    def test_equivalent_d_strings_share_a_key(self):
        groups = [
            [  # triangle: relative/absolute, H/V shorthand, Z closing
                'M 10,10 L 40,10 L 25,35 Z',
                'm10 10 h30 l-15 25 L10 10',
                'M10.0000001,10 H40 L25,35 L10,10',
            ],
            [  # cubic with S shorthand
                'M0,0 C10,20 30,20 40,0 S70,-20 80,0',
                'M0,0 C10,20 30,20 40,0 C50,-20 70,-20 80,0',
                'm0 0 c10 20 30 20 40 0 s30 -20 40 0',
            ],
            [  # quadratic with T shorthand
                'M0,0 Q20,-30 40,0 T80,0',
                'M0,0 Q20,-30 40,0 Q60,30 80,0',
            ],
        ]
        for group in groups:
            keys = {canonical_path(parse_path(d)) for d in group}
            self.assertEqual(len(keys), 1, group)

    def test_approximately_equal_but_unequal_samples(self):
        # same command structure, perturbations larger than the tolerance
        base = parse_path('M0,0 C10,20 30,20 40,0 L50,10 A5,5 0 0,1 60,20')
        perturbed = parse_path(
            'M0,0 C10,20 30,20 40.001,0 L50,10 A5,5 0 0,1 60,20.001')
        self.assertNotEqual(canonical_path(base, error=1e-6),
                            canonical_path(perturbed, error=1e-6))
        # but a sub-tolerance perturbation merges
        close = parse_path(
            'M0,0 C10,20 30,20 40.0000001,0 L50,10 A5,5 0 0,1 60,20')
        self.assertEqual(canonical_path(base, error=1e-6),
                         canonical_path(close, error=1e-6))

    def test_original_parser_suite_geometry_unchanged(self):
        # every path from the parsing test set keeps identical segment
        # geometry (up to the bound) after a canonicalize/parse round trip
        paths = [
            'M 100 100 L 300 100 L 200 300 z',
            'M 0 0 L 50 20 M 100 100 L 300 100 L 200 300 z',
            'M100,200 C100,100 250,100 250,200 S400,300 400,200',
            'M200,300 Q400,50 600,300 T1000,300',
            'M300,200 h-150 a150,150 0 1,0 150,-150 z',
            'M275,175 v-150 a150,150 0 0,0 -150,150 z',
            'M-3.4e38 3.4e38 L-3.4e-38 3.4e-38',
        ]
        for d in paths:
            original = parse_path(d)
            canonical = canonical_path(original)
            reparsed = parse_path(canonical)
            self.assertEqual(len(reparsed), len(original))
            for seg_a, seg_b in zip(original, reparsed):
                self.assertEqual(type(seg_a), type(seg_b))
                self.assertLessEqual(
                    _max_deviation(seg_a, seg_b), 1.5 * DEFAULT_ERROR)


if __name__ == '__main__':
    unittest.main()
