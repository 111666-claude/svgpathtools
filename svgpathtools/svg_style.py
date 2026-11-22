"""Computation of effective presentation styles for SVG elements.

This module implements a small, self-contained subset of the SVG/CSS
cascade so that callers can ask "what fill/stroke/... does this path
effectively have?" when presentation attributes live on ancestor groups
(e.g. a designer puts `fill` on a parent <g> and `opacity` on a child
<g>).

Scope and rules (deliberately limited):

* Only the presentation attributes listed in `PRESENTATION_ATTRIBUTES`
  are tracked: fill, stroke, stroke-width, opacity, visibility, display.
* Two sources are merged per element: XML presentation attributes and
  the inline `style` attribute.  Per the CSS cascade, a declaration in
  the inline `style` attribute wins over the same property given as a
  presentation attribute on the same element.  External stylesheets
  (<style> elements, linked CSS, class selectors) are NOT parsed.
* Inheritance/override: fill, stroke, stroke-width and visibility
  inherit down the tree; a declaration on a descendant overrides the
  inherited value.  The keyword `inherit` is treated as "no
  declaration" (the inherited value is kept).  Note that, per CSS,
  `visibility:hidden` on an ancestor CAN be re-enabled by a descendant
  declaring `visibility:visible`.
* opacity does not inherit; each element's opacity applies to the
  subtree as a whole.  The effective opacity reported for a path is the
  cumulative product of the (clamped) opacity values of the path and
  all of its ancestors.
* display does not inherit either, but `display:none` on any ancestor
  (or the element itself) hides the whole subtree and CANNOT be
  re-enabled by descendants.  The effective display is therefore 'none'
  if any element in the ancestor chain declares display:none, and
  'inline' otherwise.
* Invalid values (malformed colors, non-numeric stroke-width/opacity,
  unknown visibility keywords) are ignored, per the SVG error-handling
  rules: the declaration is dropped and the inherited/default value is
  used.  Missing values fall back to the SVG defaults in DEFAULT_STYLE.
* Nothing here mutates the SVG tree; results are computed read-only.
"""

from __future__ import division, absolute_import, print_function
import re

# The presentation attributes we track, in canonical order.
PRESENTATION_ATTRIBUTES = ('fill', 'stroke', 'stroke-width',
                           'opacity', 'visibility', 'display')

# SVG 1.1 initial values for the tracked properties.
DEFAULT_STYLE = {'fill': 'black',
                 'stroke': 'none',
                 'stroke-width': '1',
                 'visibility': 'visible',
                 'display': 'inline',
                 'opacity': 1.0}

# CSS named colors (SVG 1.1 / CSS Color 3 keyword set).
CSS_NAMED_COLORS = frozenset("""
aliceblue antiquewhite aqua aquamarine azure beige bisque black
blanchedalmond blue blueviolet brown burlywood cadetblue chartreuse
chocolate coral cornflowerblue cornsilk crimson cyan darkblue darkcyan
darkgoldenrod darkgray darkgrey darkgreen darkkhaki darkmagenta
darkolivegreen darkorange darkorchid darkred darksalmon darkseagreen
darkslateblue darkslategray darkslategrey darkturquoise darkviolet
deeppink deepskyblue dimgray dimgrey dodgerblue firebrick floralwhite
forestgreen fuchsia gainsboro ghostwhite gold goldenrod gray grey
green greenyellow honeydew hotpink indianred indigo ivory khaki
lavender lavenderblush lawngreen lemonchiffon lightblue lightcoral
lightcyan lightgoldenrodyellow lightgray lightgrey lightgreen
lightpink lightsalmon lightseagreen lightskyblue lightslategray
lightslategrey lightsteelblue lightyellow lime limegreen linen magenta
maroon mediumaquamarine mediumblue mediumorchid mediumpurple
mediumseagreen mediumslateblue mediumspringgreen mediumturquoise
mediumvioletred midnightblue mintcream mistyrose moccasin navajowhite
navy oldlace olive olivedrab orange orangered orchid palegoldenrod
palegreen paleturquoise palevioletred papayawhip peachpuff peru pink
plum powderblue purple rebeccapurple red rosybrown royalblue
saddlebrown salmon sandybrown seagreen seashell sienna silver skyblue
slateblue slategray slategrey snow springgreen steelblue tan teal
thistle tomato turquoise violet wheat white whitesmoke yellow
yellowgreen
""".split())

_HEX_COLOR_RE = re.compile(r'^#(?:[0-9a-fA-F]{3,4}|[0-9a-fA-F]{6}|[0-9a-fA-F]{8})$')
_FUNC_COLOR_RE = re.compile(r'^(?:rgb|rgba|hsl|hsla)\([^()]+\)$', re.IGNORECASE)
_NUMBER_WITH_UNIT_RE = re.compile(
    r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)'
    r'(em|ex|px|pt|pc|mm|cm|in|%)?$')
_OPACITY_RE = re.compile(
    r'^([+-]?(?:\d+\.?\d*|\.\d+)(?:[eE][+-]?\d+)?)(%)?$')

_VALID_VISIBILITIES = frozenset(['visible', 'hidden', 'collapse'])


def parse_style_attribute(style_string):
    """Parse an inline `style` attribute into a {property: value} dict.

    Property names are lowercased; values are stripped of surrounding
    whitespace but otherwise kept as written.  Malformed declarations
    (no colon, empty property) are skipped.  Later declarations of the
    same property override earlier ones, as in CSS.
    """
    declarations = {}
    if not style_string:
        return declarations
    for chunk in style_string.split(';'):
        if ':' not in chunk:
            continue
        prop, _, value = chunk.partition(':')
        prop = prop.strip().lower()
        value = value.strip()
        if prop and value:
            declarations[prop] = value
    return declarations


def is_valid_color(value):
    """Check whether a string is a usable SVG paint/color value."""
    if not value:
        return False
    v = value.strip()
    low = v.lower()
    if low in ('none', 'currentcolor', 'transparent'):
        return True
    if low in CSS_NAMED_COLORS:
        return True
    if _HEX_COLOR_RE.match(v):
        return True
    if _FUNC_COLOR_RE.match(v):
        return True
    # paint server references, e.g. url(#gradient) optionally followed
    # by a fallback; we accept the reference without resolving it
    if low.startswith('url(') and v.endswith(')'):
        return True
    return False


def _valid_stroke_width(value):
    """Return True if value is a non-negative length/number/percentage."""
    if not value:
        return False
    match = _NUMBER_WITH_UNIT_RE.match(value.strip())
    return bool(match) and float(match.group(1)) >= 0


def _parse_opacity(value):
    """Parse an opacity value, clamped to [0, 1]; None if invalid."""
    if not value:
        return None
    match = _OPACITY_RE.match(value.strip())
    if not match:
        return None
    number = float(match.group(1))
    if match.group(2):  # percentage
        number /= 100.0
    return min(1.0, max(0.0, number))


def element_declarations(element):
    """Collect the valid style declarations an element makes for itself.

    Merges the element's presentation attributes with its inline `style`
    attribute (the latter wins on conflict), drops declarations with
    invalid values, and resolves the `inherit` keyword to "no
    declaration".  Only properties in PRESENTATION_ATTRIBUTES are
    returned.
    """
    raw = {}
    for attr in PRESENTATION_ATTRIBUTES:
        value = element.get(attr)
        if value is not None:
            raw[attr] = value.strip()
    for prop, value in parse_style_attribute(element.get('style')).items():
        if prop in PRESENTATION_ATTRIBUTES:
            raw[prop] = value

    declarations = {}
    for prop, value in raw.items():
        if value.lower() == 'inherit':
            continue  # explicit inherit == keep the inherited value
        if prop in ('fill', 'stroke'):
            if is_valid_color(value):
                declarations[prop] = value
        elif prop == 'stroke-width':
            if _valid_stroke_width(value):
                declarations[prop] = value
        elif prop == 'opacity':
            if _parse_opacity(value) is not None:
                declarations[prop] = value
        elif prop == 'visibility':
            if value.lower() in _VALID_VISIBILITIES:
                declarations[prop] = value.lower()
        elif prop == 'display':
            if value:
                declarations[prop] = value.lower()
    return declarations


def merge_styles(parent_effective, own_declarations):
    """Combine a parent's effective style with an element's own
    declarations to produce the element's effective style.

    `parent_effective` and the result are dicts with the keys of
    DEFAULT_STYLE.  fill/stroke/stroke-width/visibility override,
    opacity accumulates multiplicatively, and display:none is sticky
    down the tree.
    """
    effective = dict(parent_effective)
    for prop in ('fill', 'stroke', 'stroke-width', 'visibility'):
        if prop in own_declarations:
            effective[prop] = own_declarations[prop]

    if 'opacity' in own_declarations:
        effective['opacity'] = (effective['opacity'] *
                                _parse_opacity(own_declarations['opacity']))

    if effective['display'] != 'none' and \
            own_declarations.get('display') == 'none':
        effective['display'] = 'none'

    return effective


def compute_effective_style(element_chain):
    """Compute the effective style for the last element of a chain.

    `element_chain` is a list of elements from the outermost ancestor
    down to (and including) the element itself.  Returns a dict with
    the keys of DEFAULT_STYLE.
    """
    effective = dict(DEFAULT_STYLE)
    for element in element_chain:
        effective = merge_styles(effective, element_declarations(element))
    return effective
