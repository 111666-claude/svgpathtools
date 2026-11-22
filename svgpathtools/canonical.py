"""Canonical (stable) d-string output for Path objects.

Two paths that describe the same geometry may be exported by different SVG
editors using different, but equivalent, command spellings -- relative vs.
absolute commands, ``H``/``V`` shorthands instead of ``L``, ``S``/``T``
shorthands instead of ``C``/``Q``, differing numbers of decimal places,
scientific notation, negative zero, etc.  A naive text comparison (for
example when path data is used as a cache key) then treats them as two
different assets.

This module provides :func:`canonical_path` and the convenience wrapper
:meth:`Path.canonical_d <svgpathtools.path.Path.canonical_d>`, which turn a
:class:`~svgpathtools.path.Path` into a deterministic d-string.

Canonical spelling rules
------------------------
* Only absolute commands are emitted and every command letter is written
  explicitly: ``M``, ``L``, ``C``, ``Q``, ``A``.  Relative ``m/l/c/...``
  commands, the ``H``/``V`` line shorthands and the ``S``/``T`` smooth
  shorthands are expanded into their full equivalents, so the output never
  depends on the preceding command.
* ``Z`` is never emitted: a close path is written as the explicit closing
  ``L`` back to the subpath start (exactly what the parser itself produces
  when reading ``Z``).  Direction, endpoints and topology are unchanged.
* A new ``M`` is emitted whenever a segment does not start where the previous
  one ended, which preserves multi-subpath structure.
* Numbers are written in fixed-point notation only (never scientific
  notation), with negative zero normalized to ``0``, and quantized to a
  configurable geometric error bound.

Quantization and the error bound
--------------------------------
All coordinate-like values are rounded to multiples of ``q`` where
``q = 0.5 * 10**-d`` and ``d = ceil(-log10(2 * error))``.  Consequently every
emitted coordinate differs from the corresponding input coordinate by at
most ``q <= error``; the deviation of any point on a segment is bounded by
the maximum deviation of its (control) points, so the Hausdorff distance
between the input and the canonical geometry is at most ``error`` per
coordinate axis.

If quantizing an :class:`~svgpathtools.path.Arc` would produce a degenerate
segment (a zero radius or coincident endpoints) or move it by more than the
allowed error, the precision is increased for that segment alone until the
bound is met (see the ``max_refine_steps`` parameter).  As a last resort the
segment is emitted at full floating-point fidelity and a warning is issued,
which guarantees that shapes which are genuinely farther apart than
``error`` can never collapse onto the same key.  If even full fidelity
cannot represent the segment as valid fixed-point SVG (an Arc whose
endpoints would coincide or whose radii would vanish at the numeric
limit), :class:`ValueError` is raised instead of emitting an unparseable
or ambiguous key.

Non-finite coordinates (``NaN``/``Inf``) cannot be represented in an SVG
d-string and raise :class:`ValueError`.
"""

from __future__ import absolute_import, division, print_function

import math
from decimal import Decimal, ROUND_HALF_UP, localcontext
from warnings import warn

from .path import Arc, CubicBezier, Line, QuadraticBezier

# Number of interior samples (excluding the two endpoints) used when checking
# that a quantized Arc stays within the allowed geometric error.
_ARC_SAMPLES = 64

#: Default geometric error bound applied to every coordinate (in user units).
DEFAULT_ERROR = 1e-6

#: Hard cap on decimal places (doubles as the full-fidelity fallback width;
#: a Python float has at most ~17 significant decimal digits).
_MAX_DIGITS = 17


def _digits_for_error(error):
    """Return the number of decimal places ``d`` whose quantization step
    ``0.5 * 10**-d`` is no larger than ``error``."""
    if not math.isfinite(error) or error <= 0:
        raise ValueError("error must be a finite, positive number, got %r"
                         % (error,))
    d = int(math.ceil(-math.log10(2.0 * error)))
    return max(0, min(d, _MAX_DIGITS))


def _rotation_digits(error, radius_scale):
    """Decimal places needed so that an angular error in degrees moves an
    arc point by at most ``error`` user units.

    A rotation error of ``dphi`` radians shifts a point by at most
    ``r * |dphi|`` where ``r`` is the larger radius, hence the extra
    ``log10(r)`` term in the digit count.
    """
    if radius_scale <= 1.0:
        return _digits_for_error(error)
    extra = int(math.ceil(math.log10(radius_scale)))
    return max(0, min(_digits_for_error(error) + extra, _MAX_DIGITS))


def _fmt(value, digits):
    """Quantize ``value`` to ``0.5 * 10**-digits`` and format it as a
    fixed-point string with no scientific notation and no negative zero."""
    quantum = Decimal(1).scaleb(-digits)
    # str(float) gives the shortest round-trippable decimal representation,
    # which keeps Decimal quantization free of binary representation noise.
    dec = Decimal(str(float(value)))
    # The default Decimal context carries only 28 significant digits; very
    # large magnitudes (e.g. 3.4e38) quantized to fine steps need more room.
    # The result carries ``adjusted() + 1`` integer digits plus ``digits``
    # fractional ones (and at least one digit for a zero result).
    needed = max(1, dec.adjusted() + 1) + digits + 2
    with localcontext() as ctx:
        ctx.prec = max(28, needed)
        dec = dec.quantize(quantum, rounding=ROUND_HALF_UP)
    if dec == 0:
        dec = Decimal(0)  # collapse -0 onto 0
    text = format(dec, 'f')
    if '.' in text:
        text = text.rstrip('0').rstrip('.')
    return text or '0'


def _q(value, digits):
    """Quantize ``value`` and parse it back into a Python float."""
    return float(_fmt(value, digits))


def _pt(p, digits):
    """Quantize a complex point."""
    return complex(_q(p.real, digits), _q(p.imag, digits))


def _check_finite(name, value):
    if not math.isfinite(value.real) or not math.isfinite(value.imag):
        raise ValueError(
            "Cannot canonicalize a path containing non-finite coordinates "
            "(got %s=%r); NaN and Inf are not valid SVG path numbers."
            % (name, value))


def _max_deviation(seg_a, seg_b, arc_samples=_ARC_SAMPLES):
    """Maximum distance between matching points of two segments.

    For line and Bezier segments the convex-hull property of the Bernstein
    basis bounds the deviation by the largest displacement of a (control)
    point, so no sampling is needed there.  Arcs are compared on a uniform
    parameter grid.
    """
    if isinstance(seg_a, Line):
        return max(abs(seg_a.start - seg_b.start),
                   abs(seg_a.end - seg_b.end))
    if isinstance(seg_a, QuadraticBezier):
        return max(abs(seg_a.start - seg_b.start),
                   abs(seg_a.control - seg_b.control),
                   abs(seg_a.end - seg_b.end))
    if isinstance(seg_a, CubicBezier):
        return max(abs(seg_a.start - seg_b.start),
                   abs(seg_a.control1 - seg_b.control1),
                   abs(seg_a.control2 - seg_b.control2),
                   abs(seg_a.end - seg_b.end))
    if isinstance(seg_a, Arc):
        ts = (i / arc_samples for i in range(arc_samples + 1))
        return max(abs(seg_a.point(t) - seg_b.point(t)) for t in ts)
    raise TypeError("Unsupported segment type: %r" % type(seg_a))


def _normalize_rotation(rotation):
    """Map an arc's x-axis-rotation to the half-open interval [0, 360)."""
    r = float(rotation) % 360.0
    # ``%`` can return -0.0 in edge cases; normalize the sign as well.
    return r + 0.0


def _format_point_full(p):
    return '%s,%s' % (_fmt(p.real, _MAX_DIGITS), _fmt(p.imag, _MAX_DIGITS))


def _line_candidate(segment, digits):
    return Line(_pt(segment.start, digits), _pt(segment.end, digits))


def _quadratic_candidate(segment, digits):
    return QuadraticBezier(_pt(segment.start, digits),
                           _pt(segment.control, digits),
                           _pt(segment.end, digits))


def _cubic_candidate(segment, digits):
    return CubicBezier(_pt(segment.start, digits),
                       _pt(segment.control1, digits),
                       _pt(segment.control2, digits),
                       _pt(segment.end, digits))


def _arc_candidates(segment, digits, error):
    """Build the canonical candidate(s) for an Arc at the given precision.

    Returns a list of segment candidates in preference order: a quantized
    Arc when one exists and a degenerate Line replacement (the SVG-mandated
    rendering for zero-radius / zero-length arcs) otherwise.
    """
    start = _pt(segment.start, digits)
    end = _pt(segment.end, digits)
    rx = abs(_q(segment.radius.real, digits))
    ry = abs(_q(segment.radius.imag, digits))
    rdigits = _rotation_digits(error, max(rx, ry))
    rotation = _q(_normalize_rotation(segment.rotation), rdigits) % 360.0

    candidates = []
    if rx > 0 and ry > 0 and start != end:
        try:
            candidates.append(
                Arc(start, complex(rx, ry), rotation,
                    segment.large_arc, segment.sweep, end))
        except (ValueError, AssertionError):
            pass
    # SVG renders zero-radius arcs and zero-length arcs as straight lines.
    candidates.append(Line(start, end))
    return candidates


def _with_start(segment, new_start):
    """Return a copy of a non-Arc segment whose start point is ``new_start``.

    Only the start coordinate is moved; for continuous input paths the old
    and new start coincide up to the quantization step, so the change is
    itself within the error bound.
    """
    if isinstance(segment, Line):
        return Line(new_start, segment.end)
    if isinstance(segment, QuadraticBezier):
        return QuadraticBezier(new_start, segment.control, segment.end)
    if isinstance(segment, CubicBezier):
        return CubicBezier(new_start, segment.control1, segment.control2,
                           segment.end)
    raise TypeError("Cannot weld segment of type %r" % type(segment))


def _render_segment(seg, digits, error):
    if isinstance(seg, Line):
        return 'L %s,%s' % (_fmt(seg.end.real, digits),
                            _fmt(seg.end.imag, digits))
    if isinstance(seg, QuadraticBezier):
        return 'Q %s,%s %s,%s' % (
            _fmt(seg.control.real, digits), _fmt(seg.control.imag, digits),
            _fmt(seg.end.real, digits), _fmt(seg.end.imag, digits))
    if isinstance(seg, CubicBezier):
        return 'C %s,%s %s,%s %s,%s' % (
            _fmt(seg.control1.real, digits),
            _fmt(seg.control1.imag, digits),
            _fmt(seg.control2.real, digits),
            _fmt(seg.control2.imag, digits),
            _fmt(seg.end.real, digits),
            _fmt(seg.end.imag, digits))
    if isinstance(seg, Arc):
        rdigits = _rotation_digits(
            error, max(seg.radius.real, seg.radius.imag))
        return 'A %s,%s %s %d,%d %s,%s' % (
            _fmt(seg.radius.real, digits),
            _fmt(seg.radius.imag, digits),
            _fmt(seg.rotation, rdigits),
            int(seg.large_arc), int(seg.sweep),
            _fmt(seg.end.real, digits), _fmt(seg.end.imag, digits))
    raise TypeError("Unsupported segment type: %r" % type(seg))


def _render_full_segment(seg):
    """Fixed-point, full-precision rendering used by the fallback path."""
    if isinstance(seg, Line):
        return 'L %s' % _format_point_full(seg.end)
    if isinstance(seg, QuadraticBezier):
        return 'Q %s %s' % (_format_point_full(seg.control),
                            _format_point_full(seg.end))
    if isinstance(seg, CubicBezier):
        return 'C %s %s %s' % (_format_point_full(seg.control1),
                               _format_point_full(seg.control2),
                               _format_point_full(seg.end))
    if isinstance(seg, Arc):
        return 'A %s,%s %s %d,%d %s' % (
            _fmt(seg.radius.real, _MAX_DIGITS),
            _fmt(seg.radius.imag, _MAX_DIGITS),
            _fmt(_normalize_rotation(seg.rotation), _MAX_DIGITS),
            int(seg.large_arc), int(seg.sweep),
            _format_point_full(seg.end))
    raise TypeError("Unsupported segment type: %r" % type(seg))


def _is_reparseable(seg):
    """Check that the full-fidelity rendering of ``seg`` can be parsed back.

    Fixed-point notation at :data:`_MAX_DIGITS` places a floor on the
    smallest representable non-zero coordinate.  An Arc whose endpoints (or
    radii) fall below that floor would be rendered with coincident endpoints
    or zero radii, which the SVG grammar / the parser reject.  Such geometry
    cannot be given a canonical key without lossy structural changes, so it
    must be reported rather than silently merged with another shape.
    """
    if not isinstance(seg, Arc):
        return True
    if seg.start == seg.end:
        return False
    if seg.radius.real <= 0 or seg.radius.imag <= 0:
        return False
    # The rendered (fixed-point, 17-digit) coordinates must still differ.
    start_txt = (_fmt(seg.start.real, _MAX_DIGITS),
                 _fmt(seg.start.imag, _MAX_DIGITS))
    end_txt = (_fmt(seg.end.real, _MAX_DIGITS),
               _fmt(seg.end.imag, _MAX_DIGITS))
    if start_txt == end_txt:
        return False
    if _fmt(seg.radius.real, _MAX_DIGITS) in ('0', '-0') or \
            _fmt(seg.radius.imag, _MAX_DIGITS) in ('0', '-0'):
        return False
    return True


def _canonical_segment(segment, error, max_refine_steps):
    """Return ``(digits, emitted_segment, fell_back)`` for one input segment,
    increasing precision whenever quantization would violate the error
    bound.  ``fell_back`` marks segments that could not meet the bound even
    at maximum precision; such segments are rendered and welded at full
    fidelity so they remain byte-stable on a second canonicalization."""
    base_digits = _digits_for_error(error)
    max_digits = min(base_digits + max_refine_steps, _MAX_DIGITS)

    for digits in range(base_digits, max_digits + 1):
        if isinstance(segment, Line):
            candidate = _line_candidate(segment, digits)
        elif isinstance(segment, QuadraticBezier):
            candidate = _quadratic_candidate(segment, digits)
        elif isinstance(segment, CubicBezier):
            candidate = _cubic_candidate(segment, digits)
        elif isinstance(segment, Arc):
            candidate = None
            for cand in _arc_candidates(segment, digits, error):
                if _max_deviation(segment, cand) <= error:
                    candidate = cand
                    break
            if candidate is None:
                continue
        else:
            raise TypeError(
                "Cannot canonicalize segment of type %r" % type(segment))

        if not isinstance(segment, Arc) and \
                _max_deviation(segment, candidate) > error:
            # Unreachable for Line/Bezier segments given the digit
            # construction, but kept as a defensive guard.
            continue
        return digits, candidate, False

    # Full-fidelity fallback: never silently merge distinct shapes.
    if not _is_reparseable(segment):
        raise ValueError(
            "Segment %r cannot be canonicalized at the fixed-point numeric "
            "limit of %d decimal places: its rendered endpoints would "
            "coincide (or its radii vanish), which is not a valid SVG Arc. "
            "Loosen the error bound or simplify the geometry before "
            "computing a canonical key." % (segment, _MAX_DIGITS))
    warn("Canonical quantization could not satisfy error=%g within %d extra "
         "digits; emitting segment %r at full floating-point fidelity."
         % (error, max_refine_steps, segment))
    return _MAX_DIGITS, segment, True


def _validate_path(path):
    """Reject coordinates that cannot be represented in an SVG d-string."""
    for segment in path:
        _check_finite('segment.start', segment.start)
        _check_finite('segment.end', segment.end)
        if isinstance(segment, QuadraticBezier):
            _check_finite('segment.control', segment.control)
        elif isinstance(segment, CubicBezier):
            _check_finite('segment.control1', segment.control1)
            _check_finite('segment.control2', segment.control2)
        elif isinstance(segment, Arc):
            if not math.isfinite(segment.radius.real) or \
                    not math.isfinite(segment.radius.imag) or \
                    not math.isfinite(segment.rotation):
                raise ValueError(
                    "Cannot canonicalize Arc with non-finite radius or "
                    "rotation: %r" % (segment,))
        elif not isinstance(segment, (Line, QuadraticBezier, CubicBezier)):
            raise TypeError(
                "Cannot canonicalize segment of type %r" % type(segment))


def canonical_path(path, error=DEFAULT_ERROR, max_refine_steps=6):
    """Return a deterministic, re-parseable d-string for ``path``.

    Parameters
    ----------
    path : Path
        The path to canonicalize.  It may contain
        :class:`Line`, :class:`QuadraticBezier`, :class:`CubicBezier` and
        :class:`Arc` segments in any of the spellings accepted by
        :func:`~svgpathtools.parser.parse_path` (relative/absolute
        commands, ``H``/``V``/``S``/``T`` shorthands, scientific notation,
        ...).
    error : float, optional
        Geometric error bound in user units.  Every emitted coordinate is
        quantized so that it differs from the input by at most ``error``;
        the output geometry is therefore within ``error`` of the input (per
        coordinate axis).  Defaults to :data:`DEFAULT_ERROR` (``1e-6``).
    max_refine_steps : int, optional
        Number of extra decimal places an individual segment may use when
        plain quantization would produce a degenerate Arc or violate the
        error bound.  If the bound still cannot be met, the segment is
        emitted at full fidelity and a warning is raised.  Defaults to 6.

    Returns
    -------
    str
        A d-string using only absolute ``M``/``L``/``Q``/``C``/``A``
        commands, fixed-point numbers and explicit subpath moves.  Feeding
        the result back through :func:`~svgpathtools.parser.parse_path` and
        canonicalizing again yields the identical string (byte for byte).

    Raises
    ------
    ValueError
        If ``error`` is not a finite positive number; if the path contains
        non-finite (``NaN``/``Inf``) coordinates; or if a segment cannot be
        written as valid fixed-point SVG at the numeric limit (an Arc whose
        endpoints would coincide or whose radii would vanish).
    """
    if len(path) == 0:
        return ''
    _validate_path(path)

    base_digits = _digits_for_error(error)

    # First pass: quantize each segment against the error bound.  Each entry
    # stores (rendering precision, emitted segment, fell_back).
    built = [_canonical_segment(segment, error, max_refine_steps)
             for segment in path]

    # Weld quantized joints of continuous segments: independently rounding
    # two coincident endpoints must not introduce a gap or a spurious moveto.
    # A full-fidelity segment forces its neighbor's shared joint to full
    # fidelity as well, so the welded point renders identically on a second
    # canonicalization run.
    for i in range(1, len(built)):
        digits, seg, fell_back = built[i]
        _, prev_seg, prev_fell_back = built[i - 1]
        if seg.start != prev_seg.end and path[i].start == path[i - 1].end:
            joint_digits = _MAX_DIGITS if (fell_back or prev_fell_back) \
                else digits
            built[i] = (joint_digits,
                        _with_start(seg, prev_seg.end), fell_back)

    # Second pass: render, placing a moveto at the first segment and at every
    # genuine subpath break of the *quantized* geometry.
    parts = []
    current_pos = None
    for digits, seg, fell_back in built:
        if current_pos is None or seg.start != current_pos:
            m_digits = _MAX_DIGITS if fell_back else base_digits
            parts.append('M %s,%s' % (_fmt(seg.start.real, m_digits),
                                      _fmt(seg.start.imag, m_digits)))
        if fell_back:
            parts.append(_render_full_segment(seg))
        else:
            parts.append(_render_segment(seg, digits, error))
        current_pos = seg.end

    return ' '.join(parts)
