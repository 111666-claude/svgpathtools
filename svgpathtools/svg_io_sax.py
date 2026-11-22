"""(Experimental) replacement for import/export functionality SAX

"""

# External dependencies
from __future__ import division, absolute_import, print_function
import os
from xml.etree.ElementTree import iterparse, Element, ElementTree, SubElement
import numpy as np

# Internal dependencies
from .parser import parse_path
from .parser import parse_transform
from .svg_to_paths import (path2pathd, ellipse2pathd, line2pathd,
                           polyline2pathd, polygon2pathd, rect2pathd)
from .misctools import open_in_browser
from .path import transform

# To maintain forward/backward compatibility
try:
    string = basestring
except NameError:
    string = str

NAME_SVG = "svg"
ATTR_VERSION = "version"
VALUE_SVG_VERSION = "1.1"
ATTR_XMLNS = "xmlns"
VALUE_XMLNS = "http://www.w3.org/2000/svg"
ATTR_XMLNS_LINK = "xmlns:xlink"
VALUE_XLINK = "http://www.w3.org/1999/xlink"
ATTR_XMLNS_EV = "xmlns:ev"
VALUE_XMLNS_EV = "http://www.w3.org/2001/xml-events"
ATTR_WIDTH = "width"
ATTR_HEIGHT = "height"
ATTR_VIEWBOX = "viewBox"
NAME_PATH = "path"
ATTR_DATA = "d"
ATTR_FILL = "fill"
ATTR_STROKE = "stroke"
ATTR_STROKE_WIDTH = "stroke-width"
ATTR_TRANSFORM = "transform"
VALUE_NONE = "none"


class _Attrs(dict):
    """Attribute dictionary that also mimics an Element's ``.attrib`` view.

    The SAX pipeline historically passed plain dictionaries to the shape
    converters, but ``line2pathd`` reads ``l.attrib``; exposing the dict
    itself under ``attrib`` keeps every converter happy with one object.
    """

    @property
    def attrib(self):
        return self


class SaxDocument:
    def __init__(self, filename, budget=None):
        """A container for a SAX SVG light tree objects document.

        This class provides functions for extracting SVG data into Path objects.

        Args:
            filename (str or file-like): The filename of the SVG file or an
                open binary/text file object containing SVG data.
            budget (ImportBudget or None): optional resource budget. When
                given, the document is streamed through a hardened expat
                parser and `ImportLimitExceeded` is raised at the first limit
                breach; ``None`` selects the historical ``iterparse`` path.
        """
        self.root_values = {}
        self.tree = []
        # remember location of original svg file
        if filename is not None and isinstance(filename, str) \
                and os.path.dirname(filename) == '':
            self.original_filename = os.path.join(os.getcwd(), filename)
        else:
            self.original_filename = filename

        self.budget = budget
        if filename is not None:
            if budget is None:
                self.sax_parse(filename)
            else:
                self.sax_parse_budgeted(filename, budget)

    def sax_parse(self, filename):
        self.root_values = {}
        self.tree = []
        stack = []
        values = {}
        matrix = None
        for event, elem in iterparse(filename, events=('start', 'end')):
            if event == 'start':
                stack.append((values, matrix))
                if matrix is not None:
                    matrix = matrix.copy()  # copy of matrix
                current_values = values
                values = {}
                values.update(current_values)  # copy of dictionary
                attrs = elem.attrib
                values.update(attrs)
                name = elem.tag[28:]
                if "style" in attrs:
                    for equate in attrs["style"].split(";"):
                        equal_item = equate.split(":")
                        values[equal_item[0]] = equal_item[1]
                if "transform" in attrs:
                    transform_matrix = parse_transform(attrs["transform"])
                    if matrix is None:
                        matrix = np.identity(3)
                    matrix = transform_matrix.dot(matrix)
                if "svg" == name:
                    current_values = values
                    values = {}
                    values.update(current_values)
                    self.root_values = current_values
                    continue
                elif "g" == name:
                    continue
                elif 'path' == name:
                    values['d'] = path2pathd(values)
                elif 'circle' == name:
                    values["d"] = ellipse2pathd(values)
                elif 'ellipse' == name:
                    values["d"] = ellipse2pathd(values)
                elif 'line' == name:
                    values["d"] = line2pathd(values)
                elif 'polyline' == name:
                    values["d"] = polyline2pathd(values)
                elif 'polygon' == name:
                    values["d"] = polygon2pathd(values)
                elif 'rect' == name:
                    values["d"] = rect2pathd(values)
                else:
                    continue
                values["matrix"] = matrix
                values["name"] = name
                self.tree.append(values)
            else:
                v = stack.pop()
                values = v[0]
                matrix = v[1]

    def sax_parse_budgeted(self, filename, budget):
        """SAX parse with all ImportBudget limits enforced incrementally."""
        from .limits import (guarded_iterparse, _count_path,
                             account_shape)
        self.root_values = {}
        self.tree = []
        stack = []
        values = {}
        matrix = None

        def on_start(tag, name, attrs, location, depth):
            nonlocal values, matrix
            stack.append((values, matrix))
            if matrix is not None:
                matrix = matrix.copy()
            current_values = values
            values = _Attrs()
            values.update(current_values)
            values.update(attrs)
            if "style" in attrs:
                for equate in attrs["style"].split(";"):
                    equal_item = equate.split(":")
                    values[equal_item[0]] = equal_item[1]
            if "transform" in attrs:
                transform_matrix = parse_transform(attrs["transform"])
                if matrix is None:
                    matrix = np.identity(3)
                matrix = transform_matrix.dot(matrix)
            if "svg" == name:
                cur = values
                values = _Attrs()
                values.update(cur)
                self.root_values = cur
                return
            elif "g" == name:
                return
            elif name in ('path', 'polyline', 'polygon'):
                # Enforce limits from the raw attributes before the converter
                # materializes a long list of coordinates.
                account_shape(name, attrs, budget, location, tag)
                if 'path' == name:
                    values['d'] = path2pathd(values)
                elif 'polyline' == name:
                    values["d"] = polyline2pathd(values)
                else:
                    values["d"] = polygon2pathd(values)
            elif 'circle' == name:
                values["d"] = ellipse2pathd(values)
            elif 'ellipse' == name:
                values["d"] = ellipse2pathd(values)
            elif 'line' == name:
                values["d"] = line2pathd(values)
            elif 'rect' == name:
                values["d"] = rect2pathd(values)
            else:
                return

            if name not in ('path', 'polyline', 'polygon'):
                # Small fixed-footprint shapes are counted from their d.
                _count_path(values['d'], budget=budget, location=location,
                            element=tag)

            values["matrix"] = matrix
            values["name"] = name
            self.tree.append(values)

        def on_end(tag):
            nonlocal values, matrix
            v = stack.pop()
            values = v[0]
            matrix = v[1]

        reader, opened_here = guarded_iterparse(
            filename, budget, on_start, on_end)
        if opened_here:
            reader.close()

    def flatten_all_paths(self):
        flat = []
        for values in self.tree:
            pathd = values['d']
            matrix = values['matrix']
            parsed_path = parse_path(
                pathd, budget=self.budget,
                location=self.budget.last_location if self.budget else None,
                count_against_budget=False)
            if matrix is not None:
                transform(parsed_path, matrix)
            flat.append(parsed_path)
        return flat

    def get_pathd_and_matrix(self):
        flat = []
        for values in self.tree:
            pathd = values['d']
            matrix = values['matrix']
            flat.append((pathd, matrix))
        return flat

    def generate_dom(self):
        root = Element(NAME_SVG)
        root.set(ATTR_VERSION, VALUE_SVG_VERSION)
        root.set(ATTR_XMLNS, VALUE_XMLNS)
        root.set(ATTR_XMLNS_LINK, VALUE_XLINK)
        root.set(ATTR_XMLNS_EV, VALUE_XMLNS_EV)
        width = self.root_values.get(ATTR_WIDTH, None)
        height = self.root_values.get(ATTR_HEIGHT, None)
        if width is not None:
            root.set(ATTR_WIDTH, width)
        if height is not None:
            root.set(ATTR_HEIGHT, height)
        viewbox = self.root_values.get(ATTR_VIEWBOX, None)
        if viewbox is not None:
            root.set(ATTR_VIEWBOX, viewbox)
        identity = np.identity(3)
        for values in self.tree:
            pathd = values.get('d', '')
            matrix = values.get('matrix', None)
            # path_value = parse_path(pathd)

            path = SubElement(root, NAME_PATH)
            if matrix is not None and not np.all(np.equal(matrix, identity)):
                matrix_string = "matrix("
                matrix_string += " "
                matrix_string += string(matrix[0][0])
                matrix_string += " "
                matrix_string += string(matrix[1][0])
                matrix_string += " "
                matrix_string += string(matrix[0][1])
                matrix_string += " "
                matrix_string += string(matrix[1][1])
                matrix_string += " "
                matrix_string += string(matrix[0][2])
                matrix_string += " "
                matrix_string += string(matrix[1][2])
                matrix_string += ")"
                path.set(ATTR_TRANSFORM, matrix_string)
            if ATTR_DATA in values:
                path.set(ATTR_DATA, values[ATTR_DATA])
            if ATTR_FILL in values:
                path.set(ATTR_FILL, values[ATTR_FILL])
            if ATTR_STROKE in values:
                path.set(ATTR_STROKE, values[ATTR_STROKE])
        return ElementTree(root)

    def save(self, filename):
        with open(filename, 'wb') as output_svg:
            dom_tree = self.generate_dom()
            dom_tree.write(output_svg)

    def display(self, filename=None):
        """Displays/opens the doc using the OS's default application."""
        if filename is None:
            filename = 'display_temp.svg'
        self.save(filename)
        open_in_browser(filename)
