"""Error-controlled adaptive sampling of paths into polylines.

This module converts Path objects (which may contain cubic/quadratic
Bezier curves and elliptical arcs) into polylines suitable for consumers
that only accept straight segments, e.g. vector cutting machines.

Unlike fixed-step sampling, the number of sample points adapts to the
local geometry: regions of high curvature are sampled densely while
straight regions are represented by their endpoints only.  The maximum
deviation of the true curve from the returned polyline (the chord error)
is bounded by a user-supplied tolerance.

Error bound checks (per segment type)
-------------------------------------
Line
    Exact: a line coincides with its chord, so only the two endpoints
    are emitted.

QuadraticBezier / CubicBezier
    Rigorous bound via the convex hull property.  The distance from a
    point to a line segment is a convex function of the point (a segment
    is a convex set), and a Bezier curve is a convex combination of its
    control points.  Hence, by Jensen's inequality, the distance of any
    curve point to the chord is at most the maximum distance of the
    *interior* control points to the chord.  A (sub-)curve is accepted
    as flat when this bound does not exceed the tolerance; otherwise it
    is bisected and the halves are checked recursively.

Arc
    Verified numerically at the quarter, mid and three-quarter parameter
    values of each (sub-)arc.  For a circular arc the chord deviation is
    symmetric and maximal exactly at the angular midpoint, where it
    equals the sagitta ``r*(1 - cos(dtheta/2))``, so the midpoint check
    is exact for circles.  The quarter-point checks guard the asymmetric
    case of elliptical arcs.  A sub-arc is accepted when all three
    measured deviations are within the tolerance, otherwise it is
    bisected in parameter space.

Both checks are conservative acceptance tests applied recursively, so
the returned polyline deviates from the true curve by at most
``max_chord_error`` (up to floating point round-off), which the tests
verify independently by dense sampling.
"""

# External Dependencies
from __future__ import division, absolute_import
import math

# Internal Dependencies
from .path import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                   is_path_segment)

__all__ = ['MaxPointsExceededError', 'sample_segment', 'sample_path']

DEFAULT_MAX_POINTS = 10000
_MAX_SUBDIVISION_DEPTH = 30
_NAN = complex(float('nan'), float('nan'))


class MaxPointsExceededError(RuntimeError):
    """Raised when a path cannot be sampled within ``max_points`` points
    at the requested tolerance.  Reduce the tolerance or raise the cap."""
    pass


def _validate_tolerance(max_chord_error):
    """Returns max_chord_error as a float, or raises ValueError."""
    try:
        tol = float(max_chord_error)
    except (TypeError, ValueError):
        raise ValueError(
            "max_chord_error must be a positive, finite number; got "
            "{!r}".format(max_chord_error))
    if isinstance(max_chord_error, complex) or not math.isfinite(tol) \
            or tol <= 0:
        raise ValueError(
            "max_chord_error must be a positive, finite number; got "
            "{!r}".format(max_chord_error))
    return tol


def _chord_distance(p, a, b):
    """Distance from point p to the line segment from a to b."""
    chord = b - a
    length_sq = chord.real**2 + chord.imag**2
    if length_sq == 0:
        return abs(p - a)
    u = ((p.real - a.real) * chord.real +
         (p.imag - a.imag) * chord.imag) / length_sq
    if u <= 0.0:
        return abs(p - a)
    if u >= 1.0:
        return abs(p - b)
    proj = a + u * chord
    return abs(p - proj)


def _bezier_chord_error_bound(seg, t0, t1):
    """Upper bound on the distance of seg restricted to [t0, t1] from its
    chord, from the distances of the interior control points to the
    chord (see module docstring for the justification)."""
    bpts = seg.cropped(t0, t1).bpoints()
    a, b = bpts[0], bpts[-1]
    bound = 0.0
    for p in bpts[1:-1]:
        d = _chord_distance(p, a, b)
        if d > bound:
            bound = d
    return bound


def _arc_chord_error(seg, t0, t1):
    """Measured chord deviation of an arc at the quarter, mid and
    three-quarter parameters of [t0, t1] (exact at the midpoint for
    circular arcs, where the deviation is the sagitta)."""
    a, b = seg.point(t0), seg.point(t1)
    err = 0.0
    for f in (0.25, 0.5, 0.75):
        d = _chord_distance(seg.point(t0 + f * (t1 - t0)), a, b)
        if d > err:
            err = d
    return err


def _is_flat(seg, t0, t1, tol):
    if isinstance(seg, Arc):
        return _arc_chord_error(seg, t0, t1) <= tol
    return _bezier_chord_error_bound(seg, t0, t1) <= tol


def _adaptive_ts(seg, tol):
    """Parameter values 0 = t_0 < t_1 < ... < t_n = 1 such that every
    sub-interval passes the flatness check for the segment type."""
    ts = [0.0]

    def recurse(t0, t1, depth):
        if _is_flat(seg, t0, t1, tol):
            ts.append(t1)
            return
        if depth >= _MAX_SUBDIVISION_DEPTH:
            raise RuntimeError(
                "subdivision depth limit ({}) reached while sampling "
                "{!r}; the curve may contain a singularity".format(
                    _MAX_SUBDIVISION_DEPTH, seg))
        tm = 0.5 * (t0 + t1)
        recurse(t0, tm, depth + 1)
        recurse(tm, t1, depth + 1)

    recurse(0.0, 1.0, 0)
    return ts


def _is_zero_length(seg):
    """A segment that occupies no extent and therefore contributes no
    sample points.  Note: an Arc with coincident endpoints is omitted
    entirely, matching the SVG specification."""
    if isinstance(seg, Line):
        return seg.start == seg.end
    if isinstance(seg, (QuadraticBezier, CubicBezier)):
        return all(p == seg.start for p in seg.bpoints())
    return seg.start == seg.end


def _safe_unit_tangent(seg, t):
    """Unit tangent of seg at t, or complex NaN where the tangent is
    undefined (e.g. at a cusp, where the derivative vanishes and has no
    limiting direction)."""
    try:
        ut = seg.unit_tangent(t)
    except (ZeroDivisionError, FloatingPointError, ValueError,
            AssertionError):
        return _NAN
    if not (math.isfinite(ut.real) and math.isfinite(ut.imag)):
        return _NAN
    return ut


def sample_segment(segment, max_chord_error):
    """Adaptively samples a single path segment.

    Parameters
    ----------
    segment : Line, QuadraticBezier, CubicBezier or Arc
        The segment to sample.
    max_chord_error : float
        Maximum allowed distance between the segment and the returned
        polyline.  Must be positive and finite.

    Returns
    -------
    (points, tangents) : tuple of two lists of complex
        Sample points in order of travel, including both endpoints, and
        the unit tangent at each point.  A tangent is complex NaN where
        it is undefined (cusp).  A zero-length segment yields its single
        point with a NaN tangent.
    """
    tol = _validate_tolerance(max_chord_error)
    if _is_zero_length(segment):
        return [segment.start], [_NAN]
    if isinstance(segment, Line):
        ts = [0.0, 1.0]
    else:
        ts = _adaptive_ts(segment, tol)
    points = [segment.point(t) for t in ts]
    tangents = [_safe_unit_tangent(segment, t) for t in ts]
    return points, tangents


def sample_path(path, max_chord_error, max_points=DEFAULT_MAX_POINTS):
    """Adaptively samples a path into a polyline with unit tangents.

    Parameters
    ----------
    path : Path, or a single path segment, or an iterable of segments
        The geometry to sample.
    max_chord_error : float
        Maximum allowed distance between the true curve and the returned
        polyline.  Must be positive and finite; a ValueError is raised
        otherwise.
    max_points : int
        Cap on the number of returned points.  A MaxPointsExceededError
        is raised if the path cannot be sampled within this budget at
        the requested tolerance.

    Returns
    -------
    (points, tangents) : tuple of two lists of complex
        Sample points in path order and the unit tangent at each point.

        * Points are not duplicated at segment seams: where two
          consecutive segments meet, the shared point appears once
          (carrying the incoming segment's end tangent).  A jump between
          discontinuous segments is preserved as two distinct points.
        * For a closed path the first and last points coincide
          (points[0] == points[-1]).
        * Straight (Line) segments contribute only their endpoints.
        * A tangent is complex NaN where it is undefined (cusp).  A path
          consisting only of zero-length segments yields its single
          point with a NaN tangent.
    """
    tol = _validate_tolerance(max_chord_error)
    if not max_points >= 1:
        raise ValueError(
            "max_points must be at least 1; got {!r}".format(max_points))

    if is_path_segment(path):
        segments = [path]
    else:
        segments = list(path)
    if not segments:
        return [], []

    points = []
    tangents = []

    def emit(seg, t):
        if len(points) >= max_points:
            raise MaxPointsExceededError(
                "sampling exceeds max_points={} at tolerance {}; reduce "
                "the tolerance or raise max_points".format(max_points, tol))
        points.append(seg.point(t))
        tangents.append(_safe_unit_tangent(seg, t))

    for seg in segments:
        if _is_zero_length(seg):
            continue
        if isinstance(seg, Line):
            ts = (0.0, 1.0)
        else:
            ts = _adaptive_ts(seg, tol)
        for k, t in enumerate(ts):
            # Skip the first sample of a segment when it coincides with
            # the last emitted point (continuous seam): no duplicates.
            if k == 0 and points and points[-1] == seg.start:
                continue
            emit(seg, t)

    if not points:
        # Path made entirely of zero-length segments: a single point
        # with an undefined tangent.
        points.append(segments[0].start)
        tangents.append(_NAN)

    return points, tangents
