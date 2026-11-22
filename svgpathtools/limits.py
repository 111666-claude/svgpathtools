"""Optional import budgets that make untrusted SVG ingestion bounded.

A small but malicious SVG can exhaust a process's memory by combining a
multi-megabyte path ``d`` string, deeply nested groups and billions-of-laughs
style entity expansion.  The :class:`ImportBudget` class collects hard limits
on the resources an SVG import may consume:

* ``max_bytes``        -- raw input bytes (counted while the stream is read,
                          so an oversized upload is rejected *before* the XML
                          parser ever builds a tree),
* ``max_xml_depth``    -- nesting depth of XML elements (the root is depth 1),
* ``max_elements``     -- total number of XML elements encountered,
* ``max_path_commands``-- SVG path commands allowed in a *single* path,
* ``max_total_segments``-- total number of path segments produced by the whole
                          document after shape-to-path conversion.

The budgets are opt-in: when no :class:`ImportBudget` is supplied every entry
point (``Document``, ``SaxDocument`` and ``svg2paths``) keeps its historical,
unbounded behavior exactly.

The same counters and the same :class:`ImportLimitExceeded` exception are used
by every entry point, so a given drawing is accepted or rejected for the same
reason regardless of which parser consumes it.  All limits are enforced
incrementally while the input streams through the parser -- paths are never
fully constructed first and checked afterwards.
"""

# External dependencies
from __future__ import division, absolute_import, print_function
import os
import xml.parsers.expat as _expat
from io import BytesIO

# Internal dependencies
from .path import COMMAND_RE, FLOAT_RE, _tokenize_arc_args

try:
    from os import PathLike as _PathLike
except ImportError:  # pragma: no cover
    _PathLike = str

# The SVG namespace in Clark notation as produced by namespace-aware expat.
_SVG_NS = 'http://www.w3.org/2000/svg'
_SVG_PREFIX = '{%s}' % _SVG_NS

# Names of the elements that can be converted into path data.  The ordering
# matches svg_io_sax.py / svg_to_paths.py.
_SHAPE_TAGS = frozenset((
    'path', 'circle', 'ellipse', 'line', 'polyline', 'polygon', 'rect',
))

# Regexes reused by the streaming path scanner.  COMMAND_RE has one capture
# group, so finditer() yields the command letter as group(1) and the argument
# text between two commands as group(0) minus that letter.
_COMMAND_SPLIT_RE = COMMAND_RE


class ImportLimitExceeded(Exception):
    """Raised when an SVG import would exceed an :class:`ImportBudget`.

    Attributes:
        limit (str): name of the budget that was exceeded, e.g.
            ``'max_total_segments'``.
        limit_value (int): the configured ceiling.
        observed (int): the counter value that crossed the ceiling.
        location (str): best-known position in the source, of the form
            ``'line L, column C'`` (may be ``'unknown location'``).
        element (str): Clark-notation tag of the element being processed, if
            any (otherwise ``None``).
        element_index (int): 1-based ordinal of the element among all elements
            seen so far.
    """

    def __init__(self, limit, limit_value, observed, location,
                 element=None, element_index=None):
        self.limit = limit
        self.limit_value = limit_value
        self.observed = observed
        self.location = location or 'unknown location'
        self.element = element
        self.element_index = element_index
        where = element if element else 'document'
        msg = ('SVG import budget {name} exceeded: {observed} > {value} '
               '(at {where}, {loc})').format(
                    name=limit, observed=observed, value=limit_value,
                    where=where, loc=self.location)
        super(ImportLimitExceeded, self).__init__(msg)


class InvalidSvgEntityError(ValueError):
    """Raised when an SVG uses a (general/parameter/external) entity.

    ElementTree's default parser accepts internal entity expansion, which is
    exactly the primitive behind billion-laughs denial-of-service payloads.
    Budgeted imports therefore reject every entity declaration or reference
    up front.  The five XML predefined entities (``&amp;`` ``&lt;`` ``&gt;``
    ``&quot;`` ``&apos;``) are not declared by the document and remain
    permitted.
    """

    def __init__(self, detail, location, element_index=None):
        self.detail = detail
        self.location = location or 'unknown location'
        self.element_index = element_index
        super(InvalidSvgEntityError, self).__init__(
            'SVG entities are not allowed: {detail} ({loc})'.format(
                detail=detail, loc=self.location))


class ImportBudget(object):
    """Resource limits for a single SVG import.

    Every limit is optional; a limit of ``None`` (the default) disables that
    check, so ``ImportBudget()`` only activates the structural entity
    protections.  A budget instance is consumed by exactly one import -- the
    counters are mutable and advanced as the document streams through.

    Args:
        max_bytes (int or None): maximum number of raw input bytes.
        max_xml_depth (int or None): maximum XML element nesting depth
            (the document root counts as depth 1).
        max_elements (int or None): maximum total number of XML elements.
        max_path_commands (int or None): maximum number of path commands
            (including implicit repeats and ``Z``) in any single path.
        max_total_segments (int or None): maximum total number of
            :class:`~svgpathtools.path.Path` segments produced by the import.
    """

    __slots__ = ('max_bytes', 'max_xml_depth', 'max_elements',
                 'max_path_commands', 'max_total_segments',
                 'elements', 'depth', 'path_commands', 'total_segments',
                 'last_location', 'last_element', 'reader', 'source_lines')

    def __init__(self, max_bytes=None, max_xml_depth=None,
                 max_elements=None, max_path_commands=None,
                 max_total_segments=None):
        self.max_bytes = _check_limit('max_bytes', max_bytes)
        self.max_xml_depth = _check_limit('max_xml_depth', max_xml_depth)
        self.max_elements = _check_limit('max_elements', max_elements)
        self.max_path_commands = _check_limit(
            'max_path_commands', max_path_commands)
        self.max_total_segments = _check_limit(
            'max_total_segments', max_total_segments)

        # Mutable counters, advanced during parsing.
        self.elements = 0
        self.depth = 0
        self.path_commands = 0
        self.total_segments = 0

        # Diagnostic bookkeeping.
        self.last_location = 'unknown location'
        self.last_element = None
        # Set by CountingReader so byte-limit failures can report a position.
        self.reader = None
        # Element-id -> source line map, populated by the tree-building parser.
        self.source_lines = {}

    # -- counter primitives -------------------------------------------------

    def _fail(self, name, limit_value, observed):
        raise ImportLimitExceeded(
            name, limit_value, observed, self.last_location,
            element=self.last_element, element_index=self.elements)

    def add_bytes(self, num_bytes):
        """Advance the raw-input-byte counter; raise once over budget."""
        if self.max_bytes is None or num_bytes == 0:
            return
        # The reader keeps the authoritative cumulative count so this method
        # only needs the running total when it is handed one.
        if num_bytes > self.max_bytes:
            location = (self.reader.location() if self.reader is not None
                        else self.last_location)
            raise ImportLimitExceeded(
                'max_bytes', self.max_bytes, num_bytes, location,
                element=self.last_element, element_index=self.elements)

    def enter_element(self, tag, location):
        """Record one opened element, enforcing depth and element limits."""
        self.elements += 1
        self.depth += 1
        self.last_location = location or self.last_location
        self.last_element = tag
        if (self.max_xml_depth is not None
                and self.depth > self.max_xml_depth):
            self._fail('max_xml_depth', self.max_xml_depth, self.depth)
        if (self.max_elements is not None
                and self.elements > self.max_elements):
            self._fail('max_elements', self.max_elements, self.elements)

    def leave_element(self):
        self.depth -= 1

    def note_path(self, num_commands, num_segments, location, element):
        """Account for one converted shape/path *before* it is parsed.

        This is what keeps memory bounded: both the per-path command limit and
        the document-wide segment limit are checked from a single linear,
        streaming scan of the ``d`` string, before any segment objects are
        allocated.
        """
        self.last_location = location or self.last_location
        self.last_element = element
        if (self.max_path_commands is not None
                and num_commands > self.max_path_commands):
            raise ImportLimitExceeded(
                'max_path_commands', self.max_path_commands, num_commands,
                self.last_location, element=element,
                element_index=self.elements)
        new_total = self.total_segments + num_segments
        if (self.max_total_segments is not None
                and new_total > self.max_total_segments):
            raise ImportLimitExceeded(
                'max_total_segments', self.max_total_segments, new_total,
                self.last_location, element=element,
                element_index=self.elements)
        self.total_segments = new_total
        self.path_commands = max(self.path_commands, num_commands)


def _check_limit(name, value):
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(
            '{0} must be a non-negative int or None, got {1!r}'.format(
                name, value))
    return value


# --------------------------------------------------------------------------
# Input normalization and byte counting
# --------------------------------------------------------------------------

class CountingReader(object):
    """Binary file-like wrapper that counts bytes and tracks line numbers.

    Every chunk expat pulls through :meth:`read` is measured immediately, so a
    stream that starts with gigabytes of data trips ``max_bytes`` on its first
    chunk rather than after the whole document has been buffered.
    """

    def __init__(self, stream, budget):
        self._stream = stream
        self._budget = budget
        self._bytes = 0
        self._line = 1
        self._column = 0
        budget.reader = self

    def read(self, size=-1):
        chunk = self._stream.read(size)
        if chunk:
            self._bytes += len(chunk)
            newline_count = chunk.count(b'\n')
            if newline_count:
                self._line += newline_count
                self._column = len(chunk) - chunk.rfind(b'\n') - 1
            else:
                self._column += len(chunk)
            self._budget.add_bytes(self._bytes)
        return chunk

    def location(self):
        return 'line {0}, column {1}'.format(self._line, self._column)

    def close(self):
        close = getattr(self._stream, 'close', None)
        if close is not None:
            close()


class _Utf8EncodingReader(object):
    """Adapt a text stream to the binary ``read(size)`` interface using UTF-8.

    expat (and ElementTree's XMLParser) require byte streams; this encoder
    keeps the count in *bytes*, matching what a file on disk would consume.
    """

    def __init__(self, text_stream):
        self._stream = text_stream

    def read(self, size=-1):
        if size is None or size < 0:
            text = self._stream.read()
        else:
            text = self._stream.read(size)
        return text.encode('utf-8')

    def close(self):
        close = getattr(self._stream, 'close', None)
        if close is not None:
            close()


def is_path_like(obj):
    return isinstance(obj, str) or isinstance(obj, _PathLike)


def open_budget_input(source, budget):
    """Normalize every supported input into a counted binary stream.

    Args:
        source: a filesystem path, ``bytes``/``str`` SVG document, or an
            open binary/text file-like object.
        budget (ImportBudget): budget whose byte counter is wired to the
            returned reader.

    Returns:
        tuple: ``(reader, opened_here)`` where ``reader`` is a binary
        file-like object and ``opened_here`` is True when this function opened
        the underlying file (and the caller therefore owns closing it).
    """
    if isinstance(source, bytes):
        budget.add_bytes(len(source))
        return CountingReader(BytesIO(source), budget), False
    if isinstance(source, str) and not is_path_like(source):  # pragma: no cover
        # str is always path-like; kept for clarity of the contract.
        pass
    if is_path_like(source):
        # Reject an obviously oversized file straight from its size without
        # even opening it; the per-chunk counter remains the authoritative
        # check for non-seekable/streamed inputs.
        try:
            file_size = os.path.getsize(source)
        except OSError:
            file_size = None
        if file_size is not None:
            budget.add_bytes(file_size)
        stream = open(source, 'rb')
        return CountingReader(stream, budget), True
    if hasattr(source, 'read'):
        # Distinguish text from binary streams with a zero-length probe.
        probe = source.read(0)
        if isinstance(probe, str):
            stream = _Utf8EncodingReader(source)
        else:
            stream = source
        return CountingReader(stream, budget), False
    raise TypeError(
        'Unsupported SVG input type: {0}'.format(type(source)))


# --------------------------------------------------------------------------
# Streaming path scanning: command and segment accounting without building
# the Path.
# --------------------------------------------------------------------------

# Number of numeric/flag tokens consumed by one invocation of each command.
_ARITY = {'M': 2, 'L': 2, 'H': 1, 'V': 1, 'C': 6, 'S': 4,
          'Q': 4, 'T': 2, 'A': 7, 'Z': 0}
# Number of segments appended by one invocation.
_SEGMENTS = {'M': 0, 'L': 1, 'H': 1, 'V': 1, 'C': 1, 'S': 1,
             'Q': 1, 'T': 1, 'A': 1, 'Z': 1}


def _iter_commands(pathdef):
    """Yield ``(command_letter, arg_chunk)`` pairs in source order.

    A command's data follows its letter, occupying the span up to the next
    command letter, so each command is buffered and emitted only when the
    following letter is found (or the string ends).  An implicit repeat
    (coordinate groups with no new letter) appears as further tokens inside
    the same ``arg_chunk``.

    Implemented with ``finditer`` (not ``split``) so only one argument chunk
    is materialized at a time; an oversized path can therefore abort early
    without first allocating a list with one entry per command.
    """
    last_cmd = None
    last_end = 0
    for match in _COMMAND_SPLIT_RE.finditer(pathdef):
        if last_cmd is not None:
            # Data of the previous command lies between its letter and this
            # one.
            yield last_cmd, pathdef[last_end:match.start()]
        last_cmd = match.group(1)
        last_end = match.end()
    if last_cmd is not None:
        yield last_cmd, pathdef[last_end:]


def _iter_data_tokens(command, arg_chunk):
    """Lazily tokenize one command's argument chunk.

    Arc chunks need the flag-aware tokenizer (packed large-arc/sweep flags);
    every other command only carries plain numbers, so it can use the much
    faster compiled ``FLOAT_RE``.
    """
    if command in ('A', 'a'):
        for token in _tokenize_arc_args(arg_chunk):
            yield token
    else:
        # finditer (not findall) so a giant chunk yields one number at a time
        # and an early budget abort never materializes every token.
        for m in FLOAT_RE.finditer(arg_chunk):
            yield m.group()


def scan_path(pathdef):
    """Stream over a ``d`` string and count its commands and segments.

    The scan is linear and allocates no segment objects.  Its semantics match
    :meth:`svgpathtools.path.Path._parse_path`: implicit command repeats
    (repeated coordinate groups after one command letter) each count as one
    command invocation, ``M`` creates no segment, and ``Z`` creates at most
    one closing :class:`Line` (counted whenever a ``Z`` is seen, so the
    reported segment count is a conservative upper bound -- it never
    undercounts, and only a redundant ``Z`` that closes an already-closed
    subpath can make it one larger than the realized Path).

    Returns:
        tuple: ``(command_count, segment_count)``.
    """
    return _count_path(pathdef)


def _count_path(pathdef, budget=None, location=None, element=None):
    """Counting core shared by :func:`scan_path` and budget enforcement.

    When ``budget`` is given, its per-path-command and total-segment limits
    are checked *as each coordinate group is consumed* and an oversized path
    aborts the scan early -- before the remainder of a potentially huge
    ``d`` string is even tokenized.  With ``budget=None`` it just returns the
    totals and never mutates anything.
    """
    command_count = 0
    segment_count = 0
    current = None
    # True while consuming the first group of an M/m command (a moveto, which
    # appends no segment); flips False once that group is complete so later
    # implicit repeats are treated as L groups.
    first_moveto_group = False
    pending = 0

    def _check():
        if budget is None:
            return
        budget.last_location = location or budget.last_location
        budget.last_element = element
        if (budget.max_path_commands is not None
                and command_count > budget.max_path_commands):
            raise ImportLimitExceeded(
                'max_path_commands', budget.max_path_commands,
                command_count, budget.last_location, element=element,
                element_index=budget.elements)
        new_total = budget.total_segments + segment_count
        if (budget.max_total_segments is not None
                and new_total > budget.max_total_segments):
            raise ImportLimitExceeded(
                'max_total_segments', budget.max_total_segments, new_total,
                budget.last_location, element=element,
                element_index=budget.elements)

    for command, arg_chunk in _iter_commands(pathdef):
        current = command.upper()
        first_moveto_group = (current == 'M')
        pending = 0
        if current == 'Z':
            command_count += 1
            segment_count += 1
            _check()
            current = None
            continue

        for token in _iter_data_tokens(command, arg_chunk):
            if pending > 0:
                pending -= 1
                if pending == 0 and first_moveto_group:
                    first_moveto_group = False
                    current = 'L'
                continue
            if current is None:
                # Malformed d-string; stop and let Path raise the real error.
                return command_count, segment_count

            effective = current
            command_count += 1
            if effective != 'M':
                segment_count += _SEGMENTS[effective]
            pending = _ARITY[effective] - 1
            if pending == 0 and first_moveto_group:
                first_moveto_group = False
                current = 'L'
            _check()

    if budget is not None:
        # Commit the path's segments to the document-wide counter only once
        # the whole path has cleared every per-path limit.
        budget.total_segments += segment_count
    return command_count, segment_count


# A lazy counter for the coordinate pairs inside a polyline/polygon points
# attribute.  Using finditer (not findall) means a million-point attribute is
# rejected on the count without first allocating a million tuples.
def _count_coord_pairs(points_attr):
    from .svg_to_paths import COORD_PAIR_TMPLT
    return sum(1 for _ in COORD_PAIR_TMPLT.finditer(points_attr or ''))


def account_shape(local_name, attrs, budget, location, element):
    """Enforce path/segment limits for one shape element.

    Runs *before* the (potentially expensive) conversion to a ``d`` string,
    so shapes whose size alone blows the budget are rejected cheaply.

    - ``path``: scan the ``d`` attribute incrementally.
    - ``polyline``/``polygon``: count coordinate pairs lazily (the converter
      otherwise materializes every pair into tuples before any limit check).
    - circle/ellipse/line/rect: tiny fixed shapes, counted from their
      converted ``d`` string by the caller (see the entry points).
    """
    if local_name == 'path':
        _count_path(attrs.get('d', ''), budget=budget,
                    location=location, element=element)
        return

    if local_name in ('polyline', 'polygon'):
        n_points = _count_coord_pairs(attrs.get('points', ''))
        if n_points == 0:
            return
        # Both converters emit a moveto then one lineto per interval; a
        # polygon additionally closes the ring.  Command/segment totals are
        # conservative upper bounds (a redundant closure adds nothing to the
        # realized Path).
        if local_name == 'polygon':
            n_commands = n_points + 1
        else:
            n_commands = n_points
        n_segments = n_commands
        budget.note_path(n_commands, n_segments, location, element)
        return

    # circle/ellipse/line/rect: nothing to pre-check; the entry point scans
    # their converted d-string via _count_path after conversion.


# --------------------------------------------------------------------------
# Namespace-aware expat driver with budget enforcement
# --------------------------------------------------------------------------

def _clark(namespace_name):
    """Convert expat's ``namespace|local`` name into Clark ``{ns}local``."""
    if namespace_name is None:
        return None
    if '|' in namespace_name:
        ns, local = namespace_name.split('|', 1)
        if ns:
            return '{' + ns + '}' + local
        return local
    return namespace_name


def _location(parser):
    return 'line {0}, column {1}'.format(
        parser.CurrentLineNumber, parser.CurrentColumnNumber)


class _BudgetExpatParser(object):
    """Wraps pyexpat with entity rejection and ImportBudget bookkeeping.

    Subclasses implement :meth:`on_start` / :meth:`on_end`; this base class
    owns parser creation, stream wiring, entity hardening and the shared
    depth/element counters.
    """

    def __init__(self, budget):
        self.budget = budget
        self.parser = _expat.ParserCreate(namespace_separator='|')
        p = self.parser
        # Never fetch external DTDs/entities (XXE) and never expand parameter
        # entities.
        try:
            p.SetParamEntityParsing(_expat.XML_PARAM_ENTITY_PARSING_NEVER)
        except (AttributeError, ValueError):  # pragma: no cover
            pass
        p.ExternalEntityRefHandler = self._external_entity
        p.EntityDeclHandler = self._entity_decl
        p.StartElementHandler = self._start
        p.EndElementHandler = self._end
        p.StartNamespaceDeclHandler = lambda *a: None
        p.EndNamespaceDeclHandler = lambda *a: None

    # -- entity hardening ---------------------------------------------------

    def _external_entity(self, context, base, system_id, public_id):
        # Returning 0 makes expat abort with a descriptive ExpatError; raise
        # our own exception first for a stable, typed diagnostic.
        raise InvalidSvgEntityError(
            'external entity reference {0!r}'.format(system_id or public_id),
            _location(self.parser), self.budget.elements)

    def _entity_decl(self, entity_name, is_parameter_entity, value,
                     base, system_id, public_id, notation_name):
        kind = 'parameter entity' if is_parameter_entity else 'entity'
        raise InvalidSvgEntityError(
            'declaration of {0} {1!r}'.format(kind, entity_name),
            _location(self.parser), self.budget.elements)

    # -- expat callbacks ----------------------------------------------------

    def _start(self, name, attrs):
        tag = _clark(name)
        clark_attrs = {}
        for key, value in attrs.items():
            clark_attrs[_clark(key)] = value
        location = _location(self.parser)
        # enter_element raises once a structural limit is crossed; no
        # leave_element() is needed on failure because the matching _end
        # callback never fires for an element whose start aborted.
        self.budget.enter_element(tag, location)
        self.on_start(tag, clark_attrs, location)

    def _end(self, name):
        try:
            self.on_end(_clark(name))
        finally:
            self.budget.leave_element()

    # -- hooks for subclasses ----------------------------------------------

    def on_start(self, tag, attrs, location):
        pass

    def on_end(self, tag):
        pass

    def parse(self, reader):
        self.parser.ParseFile(reader)


class _TreeBuildingParser(_BudgetExpatParser):
    """Budgeted replacement for ``xml.etree.ElementTree.parse``.

    Produces a real ElementTree whose elements carry Clark-notation tags,
    matching what ElementTree itself returns.  Each element is annotated with
    its 1-based source line (``sourceline``) for diagnostics.
    """

    def __init__(self, budget):
        super(_TreeBuildingParser, self).__init__(budget)
        import xml.etree.ElementTree as etree
        self._etree = etree
        self._stack = []
        # Modern Element objects are slot-based and cannot carry arbitrary
        # attributes, so source lines live in a side table keyed by element id.
        self.source_lines = {}
        self.root = None

    def on_start(self, tag, attrs, location):
        elem = self._etree.Element(tag, attrs)
        self.source_lines[id(elem)] = self.parser.CurrentLineNumber
        if self._stack:
            self._stack[-1].append(elem)
        else:
            self.root = elem
        self._stack.append(elem)

    def on_end(self, tag):
        self._stack.pop()

    def parse(self, reader):
        super(_TreeBuildingParser, self).parse(reader)
        return self._etree.ElementTree(self.root), self.source_lines


def _local_name(tag):
    """Strip a Clark namespace, returning the bare SVG element name."""
    if tag and tag.startswith('{') and '}' in tag:
        return tag.split('}', 1)[1]
    return tag


def guarded_parse(source, budget, on_shape=None):
    """Parse ``source`` under ``budget`` with an expat stream parser.

    Args:
        source: path, bytes/str, or open file-like SVG input.
        budget (ImportBudget): the budget to enforce.
        on_shape (callable or None): optional callback invoked for every
            convertible shape element as soon as its start tag is seen, with
            signature ``on_shape(tag_local, attrs, location, budget)``. This
            is where per-path command/segment limits are enforced, before any
            Path object exists.

    Returns:
        tuple: ``(tree, source_lines, reader, opened_here)`` where ``tree``
        is an ElementTree (never None for well-formed input) and
        ``source_lines`` maps element id to its 1-based source line.  The
        caller is responsible for closing ``reader`` when ``opened_here`` is
        True.
    """
    reader, opened_here = open_budget_input(source, budget)
    try:
        builder = _TreeBuildingParser(budget)
        if on_shape is not None:
            base_start = builder.on_start

            def hooked_start(tag, attrs, location):
                base_start(tag, attrs, location)
                local = _local_name(tag)
                if local in _SHAPE_TAGS:
                    on_shape(local, attrs, location, budget)

            builder.on_start = hooked_start
        tree, source_lines = builder.parse(reader)
    except BaseException:
        if opened_here:
            reader.close()
        raise
    return tree, source_lines, reader, opened_here


def guarded_iterparse(source, budget, on_start, on_end=None):
    """Stream-parse ``source`` under ``budget`` without retaining a tree.

    Used by the SAX entry point, which only needs start/end events and keeps
    its own lightweight ``tree`` list.

    Args:
        on_start (callable): ``on_start(tag, local_name, attrs, location,
            depth)`` invoked for every element start. ``tag`` is the Clark
            notation tag, ``depth`` is the post-entry nesting depth with the
            root at 1.
        on_end (callable or None): optional ``on_end(tag_local)``.

    Returns:
        tuple: ``(reader, opened_here)``.
    """
    reader, opened_here = open_budget_input(source, budget)
    try:
        driver = _BudgetExpatParser(budget)

        def start(tag, attrs, location):
            # Pass both the Clark tag (diagnostics) and the local name (the
            # SAX branch logic keys off bare SVG element names).
            on_start(tag, _local_name(tag), attrs, location, budget.depth)

        def end(tag):
            if on_end is not None:
                on_end(_local_name(tag))

        driver.on_start = start
        driver.on_end = end
        driver.parse(reader)
    except BaseException:
        if opened_here:
            reader.close()
        raise
    return reader, opened_here
