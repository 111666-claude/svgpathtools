"""Tests for effective presentation style computation (Document.paths
with include_styles=True).

To run these tests, you can use (from root svgpathtools directory):
$ python -m unittest test.test_effective_styles
"""
from __future__ import division, absolute_import, print_function
import unittest
from svgpathtools import Document, DEFAULT_STYLE, compute_effective_style
from svgpathtools.svg_style import (parse_style_attribute,
                                    element_declarations, is_valid_color)
from os.path import join, dirname


def styles_by_id(doc, **kwargs):
    """Return {element id: effective_style} for all paths, preserving
    duplicates as lists of (index, style)."""
    result = {}
    for path in doc.paths(include_styles=True, **kwargs):
        elem_id = path.element.get('id')
        result.setdefault(elem_id, []).append(path.effective_style)
    return result


class TestEffectiveStyles(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.doc = Document(join(dirname(__file__), 'effective_styles.svg'))
        cls.styles = styles_by_id(cls.doc)

    def style(self, elem_id):
        matches = self.styles[elem_id]
        self.assertEqual(1, len(matches),
                         'expected exactly one path with id %r' % elem_id)
        return matches[0]

    def test_defaults_when_nothing_declared(self):
        # p-missing sits in a group whose fill/stroke-width/opacity
        # declarations are all invalid, so it sees the root's values
        # (fill=gray, stroke=black, stroke-width=1) and default
        # visibility/display/opacity.
        style = self.style('p-missing')
        self.assertEqual(style['fill'], 'gray')
        self.assertEqual(style['stroke'], 'black')
        self.assertEqual(style['stroke-width'], '1')
        self.assertEqual(style['visibility'], 'visible')
        self.assertEqual(style['display'], 'inline')
        self.assertAlmostEqual(style['opacity'], 1.0)

    def test_inheritance_through_nested_groups(self):
        # fill/stroke/stroke-width on the grandparent group, opacity on
        # the parent group: the path inherits all of them.
        style = self.style('p-inherit')
        self.assertEqual(style['fill'], 'red')
        self.assertEqual(style['stroke'], 'blue')
        self.assertEqual(style['stroke-width'], '2')
        self.assertAlmostEqual(style['opacity'], 0.5)

    def test_local_attribute_overrides_inheritance(self):
        style = self.style('p-override')
        self.assertEqual(style['fill'], 'green')
        # non-overridden properties still inherit
        self.assertEqual(style['stroke'], 'blue')
        self.assertEqual(style['stroke-width'], '2')

    def test_inline_style_beats_attributes_and_inheritance(self):
        style = self.style('p-style')
        # inline style wins over the element's own fill attribute
        self.assertEqual(style['fill'], 'orange')
        # inline style wins over the inherited stroke-width
        self.assertEqual(style['stroke-width'], '4')
        self.assertEqual(style['visibility'], 'hidden')
        # inherited stroke is untouched
        self.assertEqual(style['stroke'], 'blue')

    def test_visibility_hidden_and_reenabled(self):
        self.assertEqual(self.style('p-hidden')['visibility'], 'hidden')
        # per CSS, a descendant may override visibility back to visible
        self.assertEqual(self.style('p-visible-again')['visibility'],
                         'visible')

    def test_display_none_is_sticky(self):
        self.assertEqual(self.style('p-display-none')['display'], 'none')
        # a descendant cannot re-enable display
        self.assertEqual(self.style('p-display-none-nested')['display'],
                         'none')

    def test_opacity_accumulates_multiplicatively(self):
        # 0.5 (outer) * 0.4 (inner) * 0.5 (path)
        self.assertAlmostEqual(self.style('p-opacity')['opacity'], 0.1)

    def test_invalid_values_are_ignored(self):
        style = self.style('p-invalid')
        # invalid group fill is dropped -> root's fill inherits
        self.assertEqual(style['fill'], 'gray')
        # invalid path stroke is dropped -> root's stroke inherits
        self.assertEqual(style['stroke'], 'black')
        # negative stroke-width is dropped -> root's stroke-width
        self.assertEqual(style['stroke-width'], '1')
        # bogus path opacity is dropped; group opacity 2 clamps to 1
        self.assertAlmostEqual(style['opacity'], 1.0)

    def test_duplicate_ids_each_get_their_own_style(self):
        matches = self.styles['dup']
        self.assertEqual(2, len(matches))
        fills = sorted(style['fill'] for style in matches)
        self.assertEqual(fills, ['navy', 'teal'])

    def test_result_order_and_count_unchanged(self):
        # include_styles must not change which paths are returned or
        # their order
        plain = self.doc.paths()
        styled = self.doc.paths(include_styles=True)
        self.assertEqual(len(plain), len(styled))
        for p_plain, p_styled in zip(plain, styled):
            self.assertIs(p_plain.element, p_styled.element)
            self.assertEqual(p_plain.d(), p_styled.d())
        # and plain results carry no effective_style attribute
        for p in plain:
            self.assertFalse(hasattr(p, 'effective_style'))

    def test_tree_is_not_modified(self):
        before = repr(self.doc)
        self.doc.paths(include_styles=True)
        self.assertEqual(before, repr(self.doc))

    def test_paths_from_group_include_styles(self):
        # the cascade includes ancestors between the root and the group
        paths = self.doc.paths_from_group(['fill-parent', 'opacity-child'],
                                          include_styles=True)
        by_id = {p.element.get('id'): p.effective_style for p in paths}
        self.assertEqual(by_id['p-inherit']['fill'], 'red')
        self.assertAlmostEqual(by_id['p-inherit']['opacity'], 0.5)


class TestStyleHelpers(unittest.TestCase):

    def test_parse_style_attribute(self):
        self.assertEqual(
            parse_style_attribute('fill: red; stroke-width:2 ;'),
            {'fill': 'red', 'stroke-width': '2'})
        self.assertEqual(parse_style_attribute(''), {})
        self.assertEqual(parse_style_attribute(None), {})
        # malformed declarations are skipped; later wins
        self.assertEqual(parse_style_attribute('nonsense; fill:a; fill:b'),
                         {'fill': 'b'})

    def test_is_valid_color(self):
        for good in ['red', '#fff', '#FF00AA', '#ff00aa80', 'none',
                     'currentColor', 'transparent', 'rgb(1,2,3)',
                     'rgba(1,2,3,0.5)', 'url(#grad)', 'hsl(0, 0%, 0%)']:
            self.assertTrue(is_valid_color(good), good)
        for bad in ['', 'not-a-color', '#12', '#12345', 'rgb(', '123']:
            self.assertFalse(is_valid_color(bad), bad)

    def test_element_declarations_style_beats_attribute(self):
        doc = Document.from_svg_string(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<path d="M0,0 L1,1" fill="red" '
            'style="fill: blue; opacity: 0.5"/></svg>')
        elem = doc.tree.getroot()[0]
        decls = element_declarations(elem)
        self.assertEqual(decls['fill'], 'blue')
        self.assertEqual(decls['opacity'], '0.5')

    def test_inherit_keyword_keeps_inherited_value(self):
        doc = Document.from_svg_string(
            '<svg xmlns="http://www.w3.org/2000/svg" fill="red">'
            '<path d="M0,0 L1,1" fill="inherit"/></svg>')
        root, elem = doc.tree.getroot(), doc.tree.getroot()[0]
        style = compute_effective_style([root, elem])
        self.assertEqual(style['fill'], 'red')

    def test_default_style_keys(self):
        doc = Document.from_svg_string(
            '<svg xmlns="http://www.w3.org/2000/svg">'
            '<path d="M0,0 L1,1"/></svg>')
        style = doc.paths(include_styles=True)[0].effective_style
        self.assertEqual(set(style.keys()), set(DEFAULT_STYLE.keys()))
        self.assertEqual(style['fill'], 'black')
        self.assertEqual(style['stroke'], 'none')
        self.assertEqual(style['stroke-width'], '1')
        self.assertEqual(style['visibility'], 'visible')
        self.assertEqual(style['display'], 'inline')
        self.assertAlmostEqual(style['opacity'], 1.0)


if __name__ == '__main__':
    unittest.main()
