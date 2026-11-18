"""Area moments of closed paths: signed area, centroid, second (area)
moments and principal axes.

These are the quantities a typesetting or layout tool needs to rotate a
glyph/icon about its true area centroid (rather than its bounding-box
center) and to know how the enclosed area is distributed.

The computation uses Green's theorem to turn each area integral into a
line integral along the boundary:

    A   = integral x dy                      (signed area)
    Cx  = (1/2A) integral x^2 dy             (centroid x)
    Cy  = -(1/2A) integral y^2 dx            (centroid y)
    Ixx = -(1/3) integral y^3 dx             (second moment, int y^2 dA)
    Iyy = (1/3)  integral x^3 dy             (second moment, int x^2 dA)
    Ixy = (1/2)  integral x^2 y dy           (product moment, int xy dA)

For Line, QuadraticBezier and CubicBezier segments these integrals are
evaluated exactly (the integrands are polynomials in the curve
parameter).  For Arc segments they are evaluated by adaptive
Gauss-Legendre quadrature that refines until the requested `error`
threshold is met.

Notes
-----
* The sign convention matches ``Path.area``: counter-clockwise
  parameterizations contribute positive area.  Composite paths
  (multiple closed subpaths) are summed with their signs, so a
  clockwise subpath inside a counter-clockwise one is automatically
  subtracted as a hole.
* Self-intersecting contours are not rejected; the result is the
  winding-number-weighted moment of the contour (each region counts
  with its winding number).  If that weighted area is zero the centroid
  is undefined and a ValueError is raised.
* To stay accurate for paths with very large coordinates, all
  integrals are evaluated relative to the path's start point and the
  results are shifted back analytically (parallel axis theorem).
"""

from __future__ import division, absolute_import, print_function

from collections import namedtuple

import numpy as np

from .path import Path, Line, QuadraticBezier, CubicBezier, Arc
from .polytools import real, imag

__all__ = ['AreaMoments', 'area_moments']


class AreaMoments(namedtuple('AreaMoments',
                             ['area', 'centroid', 'ixx', 'iyy', 'ixy',
                              'principal_angle', 'principal_moments'])):
    r"""Container returned by :func:`area_moments`.

    Attributes
    ----------
    area : float
        Signed enclosed area.  Positive when the (net) parameterization
        is counter-clockwise, negative when clockwise.
    centroid : complex
        The area centroid ``x + 1j*y``.  Rotating the path about this
        point (instead of the bounding-box center) keeps it visually
        stationary.
    ixx, iyy, ixy : float
        Central second area moments, i.e. taken about axes through the
        centroid: ``ixx = int (y-cy)^2 dA``, ``iyy = int (x-cx)^2 dA``,
        ``ixy = int (x-cx)(y-cy) dA``.  They carry the sign of `area`.
    principal_angle : float
        Direction (radians, CCW from the positive x-axis) of the major
        principal axis -- the axis about which the central moment is
        smallest, i.e. the direction of greatest extent.  Defined modulo
        pi.  If the two principal moments are (numerically) equal, any
        direction is a principal axis and 0.0 is returned.
    principal_moments : tuple[float, float]
        ``(i_min, i_max)``, the central second moments about the major
        and minor principal axes.  Normalized by the sign of `area`, so
        they are non-negative for any simple closed contour.
    """
    __slots__ = ()


def _validate_error(error):
    """Checks that `error` is a real, finite, strictly positive number."""
    if isinstance(error, bool):
        raise ValueError("error tolerance must be a real number, "
                         "got %r" % (error,))
    try:
        error = float(error)
    except (TypeError, ValueError):
        raise ValueError("error tolerance must be a real number, "
                         "got %r" % (error,))
    if not np.isfinite(error) or error <= 0:
        raise ValueError("error tolerance must be positive and finite, "
                         "got %r" % (error,))
    return error


def _integrate_poly(poly):
    """Exact integral of a numpy.poly1d over t in [0, 1]."""
    integral = poly.integ()
    return float(integral(1) - integral(0))


def _bezier_segment_integrals(seg, origin):
    """The six Green's-theorem line integrals for a Line/Quadratic/
    Cubic segment, evaluated exactly, relative to `origin`."""
    p = seg.poly() - origin
    x, y = real(p), imag(p)
    dx, dy = x.deriv(), y.deriv()
    a = _integrate_poly(x * dy)
    mx = 0.5 * _integrate_poly(x * x * dy)
    my = -0.5 * _integrate_poly(y * y * dx)
    ixx = -_integrate_poly(y * y * y * dx) / 3.0
    iyy = _integrate_poly(x * x * x * dy) / 3.0
    ixy = 0.5 * _integrate_poly(x * x * y * dy)
    return np.array([a, mx, my, ixx, iyy, ixy])


def _arc_integrals_once(seg, origin, n):
    """One Gauss-Legendre estimate (n nodes) of the six integrals."""
    nodes, weights = np.polynomial.legendre.leggauss(n)
    ts = 0.5 * (nodes + 1.0)          # map [-1, 1] -> [0, 1]
    ws = 0.5 * weights
    pts = np.array([seg.point(t) for t in ts]) - origin
    der = np.array([seg.derivative(t) for t in ts])
    x, y = pts.real, pts.imag
    dx, dy = der.real, der.imag
    return np.array([
        np.dot(ws, x * dy),
        0.5 * np.dot(ws, x * x * dy),
        -0.5 * np.dot(ws, y * y * dx),
        -np.dot(ws, y * y * y * dx) / 3.0,
        np.dot(ws, x * x * x * dy) / 3.0,
        0.5 * np.dot(ws, x * x * y * dy),
    ])


def _arc_segment_integrals(seg, origin, error, min_depth, max_depth):
    """Adaptive Gauss-Legendre evaluation of the six integrals for an
    Arc segment, refined until the relative change is below `error`."""
    # characteristic size of the arc, used to normalize the (mixed-unit)
    # integrals: a ~ s^2, mx/my ~ s^3, ixx/iyy/ixy ~ s^4
    s = max(abs(seg.start - seg.center), abs(seg.end - seg.center),
            abs(seg.radius), 1.0)
    units = np.array([s**2, s**3, s**3, s**4, s**4, s**4])

    n = 2 ** max(3, min_depth)
    previous = _arc_integrals_once(seg, origin, n)
    for depth in range(min_depth, max_depth + 1):
        n *= 2
        current = _arc_integrals_once(seg, origin, n)
        if np.all(np.abs(current - previous) <= error * units):
            return current
        previous = current
    raise ArithmeticError(
        "Arc segment %r did not converge to error=%s after %d "
        "refinements." % (seg, error, max_depth))


def _segment_integrals(seg, origin, error, min_depth, max_depth):
    if isinstance(seg, (Line, QuadraticBezier, CubicBezier)):
        return _bezier_segment_integrals(seg, origin)
    elif isinstance(seg, Arc):
        return _arc_segment_integrals(seg, origin, error,
                                      min_depth, max_depth)
    raise NotImplementedError(
        "area moments are not implemented for segment type %s"
        % type(seg).__name__)


def area_moments(path, error=1e-12, min_depth=5, max_depth=50):
    """Computes signed area, centroid, central second moments and
    principal axes of the region enclosed by `path`.

    Args:
        path (Path): a closed path.  May contain several closed
            subpaths; each contributes with the sign of its winding
            direction, so oppositely-wound interior subpaths act as
            holes.
        error (float): relative accuracy required of the quadrature
            used for Arc segments (Line/QuadraticBezier/CubicBezier
            segments are always integrated exactly).  Must be a
            positive, finite real number.
        min_depth (int): minimum quadrature refinement depth for arcs.
        max_depth (int): maximum quadrature refinement depth for arcs
            before an ArithmeticError is raised.

    Returns:
        AreaMoments: see the AreaMoments docstring for the fields.

    Raises:
        ValueError: if `path` is empty or has an unclosed subpath, if
            the enclosed (winding-weighted) area is zero -- in which
            case the centroid is undefined -- or if `error` is not a
            positive finite number.

    Notes
    -----
    * Self-intersecting contours are evaluated with winding-number
      weighting; no error is raised for them unless the weighted area
      vanishes.
    * The second moments `ixx`, `iyy`, `ixy` are central (taken about
      the centroid) and carry the sign of `area`.
    """
    error = _validate_error(error)

    if not isinstance(path, Path):
        path = Path(*path)
    if len(path) == 0:
        raise ValueError("Cannot compute area moments of an empty path.")

    subpaths = path.continuous_subpaths()
    for i, subpath in enumerate(subpaths):
        if not (subpath.iscontinuous()
                and subpath.start == subpath.end):
            raise ValueError(
                "area_moments() requires a closed path, but subpath %d "
                "is not closed (start=%s, end=%s)."
                % (i, subpath.start, subpath.end))

    # Integrate relative to the path's start point so that huge absolute
    # coordinates do not destroy precision through cancellation.
    origin = path[0].start

    totals = np.zeros(6)
    for subpath in subpaths:
        for seg in subpath:
            totals += _segment_integrals(seg, origin, error,
                                         min_depth, max_depth)
    area, mx, my, ixx_o, iyy_o, ixy_o = totals

    # Characteristic size of the path, for the zero-area threshold.
    xmin, xmax, ymin, ymax = path.bbox()
    scale = max(xmax - xmin, ymax - ymin)
    zero_tol = 100 * np.finfo(float).eps * max(scale, 1.0)**2
    if not np.isfinite(area) or abs(area) <= zero_tol:
        raise ValueError(
            "The path encloses zero (or numerically vanishing) signed "
            "area (area=%s); centroid and principal axes are undefined."
            % (area,))

    cx, cy = mx / area, my / area          # centroid relative to origin
    centroid = origin + complex(cx, cy)

    # Parallel axis theorem: shift the second moments from `origin` to
    # the centroid.  All quantities are signed, and the identities hold
    # algebraically regardless of the sign of `area`.
    ixx = ixx_o - area * cy * cy
    iyy = iyy_o - area * cx * cx
    ixy = ixy_o - area * cx * cy

    # Principal axes of the (sign-normalized) central moment tensor
    # T = [[ixx, -ixy], [-ixy, iyy]]: the moment about the axis through
    # the centroid along unit vector v is v^T T v.
    sign = 1.0 if area > 0 else -1.0
    tensor = sign * np.array([[ixx, -ixy], [-ixy, iyy]])
    eigvals, eigvecs = np.linalg.eigh(tensor)   # ascending eigenvalues
    i_min, i_max = float(eigvals[0]), float(eigvals[1])
    if np.isclose(i_min, i_max, rtol=1e-9, atol=zero_tol**2):
        principal_angle = 0.0     # every axis is principal (e.g. circle)
    else:
        major = eigvecs[:, 0]     # axis of least moment = greatest extent
        principal_angle = float(np.arctan2(major[1], major[0]))

    return AreaMoments(area=float(area),
                       centroid=centroid,
                       ixx=float(ixx),
                       iyy=float(iyy),
                       ixy=float(ixy),
                       principal_angle=principal_angle,
                       principal_moments=(i_min, i_max))
