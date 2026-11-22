"""Tests for the optional SVG import budget (`svgpathtools.limits`).

Covers the five limits across all three entry points (Document, SaxDocument
and svg2paths), boundary semantics, diagnostic positions, entity hardening,
file-handle cleanup on failure, numeric edge cases, and the parity of the
budgeted parsers with the historical ones.
"""
from __future__ import division, absolute_import, print_function
import gc
import os
import tempfile
import unittest
from io import BytesIO, StringIO

from svgpathtools import (Document, SaxDocument, svg2paths, svgstr2paths,
                          ImportBudget, ImportLimitExceeded,
                          InvalidSvgEntityError, parse_path)
from svgpathtools.limits import scan_path, account_shape

NS = 'xmlns="http://www.w3.org/2000/svg"'


def svg(body):
    return '<svg {0}>{1}</svg>'.format(NS, body)


# One path with 3 commands / 2 segments, depth 3, three elements.
THREE_ELEM = svg('<g><path d="M0 0L1 1L2 2"/></g>')
THREE_ELEM_BYTES = THREE_ELEM.encode('utf-8')


class _Base(object):
    """Shared cases, parameterized per entry point by subclasses."""

    def parse(self, source, budget):  # pragma: no cover - overridden
        raise NotImplementedError

    # -- each limit fires ---------------------------------------------------

    def test_depth_limit(self):
        with self.assertRaises(ImportLimitExceeded) as cm:
            self.parse(StringIO(THREE_ELEM),
                       ImportBudget(max_xml_depth=2))
        self.assertEqual(cm.exception.limit, 'max_xml_depth')
        self.assertEqual(cm.exception.observed, 3)
        self.assertIn('line', cm.exception.location)
        self.assertIn('g', cm.exception.element)

    def test_element_limit(self):
        with self.assertRaises(ImportLimitExceeded) as cm:
            self.parse(StringIO(THREE_ELEM),
                       ImportBudget(max_elements=2))
        self.assertEqual(cm.exception.limit, 'max_elements')
        self.assertEqual(cm.exception.observed, 3)

    def test_path_command_limit(self):
        with self.assertRaises(ImportLimitExceeded) as cm:
            self.parse(StringIO(THREE_ELEM),
                       ImportBudget(max_path_commands=2))
        self.assertEqual(cm.exception.limit, 'max_path_commands')
        self.assertEqual(cm.exception.observed, 3)
        self.assertIn('path', cm.exception.element)

    def test_total_segment_limit(self):
        with self.assertRaises(ImportLimitExceeded) as cm:
            self.parse(StringIO(THREE_ELEM),
                       ImportBudget(max_total_segments=1))
        self.assertEqual(cm.exception.limit, 'max_total_segments')
        self.assertGreaterEqual(cm.exception.observed, 2)

    def test_byte_limit_streaming(self):
        with self.assertRaises(ImportLimitExceeded) as cm:
            self.parse(BytesIO(THREE_ELEM_BYTES),
                       ImportBudget(max_bytes=10))
        self.assertEqual(cm.exception.limit, 'max_bytes')
        self.assertGreaterEqual(cm.exception.observed, 11)

    # -- boundary semantics: equal accepts, one less rejects ----------------

    def test_boundaries_accept_then_reject(self):
        actual = {'max_xml_depth': 3, 'max_elements': 3,
                  'max_path_commands': 3, 'max_total_segments': 2}
        for name, value in actual.items():
            # Equal to the real size -> accepted.
            self.parse(StringIO(THREE_ELEM), ImportBudget(**{name: value}))
            # One under -> rejected.
            with self.assertRaises(ImportLimitExceeded):
                self.parse(StringIO(THREE_ELEM),
                           ImportBudget(**{name: value - 1}))

    def test_byte_boundary(self):
        size = len(THREE_ELEM_BYTES)
        self.parse(BytesIO(THREE_ELEM_BYTES),
                   ImportBudget(max_bytes=size))
        with self.assertRaises(ImportLimitExceeded):
            self.parse(BytesIO(THREE_ELEM_BYTES),
                       ImportBudget(max_bytes=size - 1))

    # -- no limit means accept ---------------------------------------------

    def test_open_budget_accepts(self):
        self.parse(StringIO(THREE_ELEM), ImportBudget())

    # -- entities -----------------------------------------------------------

    def test_custom_entity_rejected(self):
        evil = ('<!DOCTYPE svg [<!ENTITY x "abc">]>'
                '<svg {0}><a>&x;</a></svg>').format(NS)
        with self.assertRaises(InvalidSvgEntityError):
            self.parse(BytesIO(evil.encode()), ImportBudget())

    def test_external_entity_rejected(self):
        evil = ('<!DOCTYPE svg [<!ENTITY x SYSTEM "file:///etc/hostname">]>'
                '<svg {0}><a>&x;</a></svg>').format(NS)
        with self.assertRaises(InvalidSvgEntityError):
            self.parse(BytesIO(evil.encode()), ImportBudget())

    def test_predefined_entities_allowed(self):
        ok = svg('<path d="M0 0L1 1" id="a&amp;b"/>')
        self.parse(StringIO(ok), ImportBudget())

    def test_billion_laughs_stopped_early(self):
        bl = b'<!DOCTYPE svg [<!ENTITY a0 "aaaa">'
        for i in range(1, 30):
            bl += ('<!ENTITY a%d "&a%d;&a%d;&a%d;&a%d;">'
                   % (i, i - 1, i - 1, i - 1, i - 1)).encode()
        bl += b']><svg ' + NS.encode() + b'><a>&a29;</a></svg>'
        # Tiny on disk, would expand astronomically without protection.
        self.assertLess(len(bl), 2000)
        with self.assertRaises(InvalidSvgEntityError):
            self.parse(BytesIO(bl), ImportBudget())

    # -- numeric edge cases stay stable ------------------------------------

    def test_overflow_and_long_numbers(self):
        weird = [
            'M0 0L1e9999 1e9999',
            'M0 0L' + '9' * 400 + ' ' + '9' * 400,
            'M0 0L1.' + '0' * 5000 + ' 2',
        ]
        for d in weird:
            doc = svg('<path d="{0}"/>'.format(d))
            # Deterministic: two runs raise or succeed identically.
            r1 = self._try(StringIO(doc))
            r2 = self._try(StringIO(doc))
            self.assertEqual(r1, r2)

    def _try(self, source):
        try:
            self.parse(source, ImportBudget(max_path_commands=1000))
            return 'ok'
        except ImportLimitExceeded as e:
            return e.limit
        except Exception as e:  # pragma: no cover - would indicate instability
            return type(e).__name__

    # -- malformed input is a parser error, not a budget error --------------

    def test_malformed_xml_raises_parse_error(self):
        import xml.parsers.expat
        with self.assertRaises(xml.parsers.expat.ExpatError):
            self.parse(StringIO('<svg><g></svg>'), ImportBudget())

    # -- file handle discipline --------------------------------------------

    def test_path_input_file_closed_on_failure(self):
        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, 'deep.svg')
        with open(p, 'w') as f:
            f.write(svg('<g>' * 50 + '</g>' * 50))
        before = len(os.listdir('/proc/self/fd'))
        for _ in range(20):
            with self.assertRaises(ImportLimitExceeded):
                self.parse(p, ImportBudget(max_xml_depth=5))
        gc.collect()
        after = len(os.listdir('/proc/self/fd'))
        self.assertLessEqual(after, before + 1)

    def test_caller_file_object_not_force_closed(self):
        class Tracked(object):
            def __init__(self, raw):
                self.raw = raw
                self.closed = False

            def read(self, n=-1):
                return self.raw.read(n)

            def close(self):
                self.closed = True
                self.raw.close()

        tmp = tempfile.mkdtemp()
        p = os.path.join(tmp, 'deep.svg')
        with open(p, 'w') as f:
            f.write(svg('<g>' * 50 + '</g>' * 50))
        t = Tracked(open(p, 'rb'))
        with self.assertRaises(ImportLimitExceeded):
            self.parse(t, ImportBudget(max_xml_depth=5))
        self.assertFalse(t.closed)
        t.raw.close()


class TestDocumentBudget(_Base, unittest.TestCase):
    def parse(self, source, budget):
        return Document(source, budget=budget).paths()


class TestSaxBudget(_Base, unittest.TestCase):
    def parse(self, source, budget):
        return SaxDocument(source, budget=budget).flatten_all_paths()


class TestSvg2PathsBudget(_Base, unittest.TestCase):
    def parse(self, source, budget):
        return svg2paths(source, budget=budget)[0]

    def test_convert_flags_respected(self):
        doc = svg('<circle cx="0" cy="0" r="5"/><line x1="0" y1="0" '
                  'x2="1" y2="1"/>')
        both = svg2paths(StringIO(doc), budget=ImportBudget())[0]
        self.assertEqual(len(both), 2)
        no_circles = svg2paths(
            StringIO(doc), budget=ImportBudget(),
            convert_circles_to_paths=False)[0]
        self.assertEqual(len(no_circles), 1)


class TestScanPath(unittest.TestCase):
    def test_known_counts(self):
        cases = {
            'M0 0L1 1L2 2': (3, 2),
            'M0 0H10V20': (3, 2),
            'M0,0 1,1 2,2 3,3': (4, 3),       # implicit moveto -> lineto
            'M0 0 A5 5 0 0110 10': (2, 1),     # packed arc flags
            '': (0, 0),
        }
        for d, expected in cases.items():
            self.assertEqual(scan_path(d), expected)

    def test_never_undercounts(self):
        # The scan is a conservative upper bound versus the realized Path.
        for d in ['M0 0L1 1L2 2 Z', 'M0 0 Z',
                  'M0 0L10 10 M20 20L30 30 Z']:
            _, segs = scan_path(d)
            self.assertGreaterEqual(segs, len(parse_path(d)))

    def test_early_abort_does_not_scan_whole_string(self):
        d = 'M0 0 ' + ' '.join('L%d %d' % (i, i) for i in range(100000))
        b = ImportBudget(max_path_commands=500)
        with self.assertRaises(ImportLimitExceeded) as cm:
            account_shape('path', {'d': d}, b, 'line 1', '{ns}path')
        self.assertLessEqual(cm.exception.observed, 502)


class TestDirectParsePathBudget(unittest.TestCase):
    def test_direct_parse_path_segment_budget(self):
        d = 'M0 0L1 1L2 2L3 3'
        with self.assertRaises(ImportLimitExceeded):
            parse_path(d, budget=ImportBudget(max_total_segments=2))

    def test_direct_parse_path_within_budget(self):
        d = 'M0 0L1 1L2 2'
        path = parse_path(d, budget=ImportBudget(max_total_segments=10))
        self.assertEqual(len(path), 2)


class TestDefaultBehaviorUnchanged(unittest.TestCase):
    def test_no_budget_keyword_still_works(self):
        # Every entry point accepts its historical call signature.
        Document(StringIO(THREE_ELEM)).paths()
        SaxDocument(StringIO(THREE_ELEM)).flatten_all_paths()
        svg2paths(StringIO(THREE_ELEM))
        svgstr2paths(THREE_ELEM)

    def test_budget_validation(self):
        for bad in (-1, 1.5, '100', True):
            with self.assertRaises(ValueError):
                ImportBudget(max_elements=bad)


if __name__ == '__main__':
    unittest.main()
