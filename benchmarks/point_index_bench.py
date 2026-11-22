"""Correctness and performance checks for the Path.point / T2t arc-length
location index.

Run from the repository root:

    python3 -I benchmarks/point_index_bench.py

Everything here is test-only code; the package itself is imported normally.
The "old" reference implementations below are byte-for-byte copies of the
pre-index linear-scan logic (sequential cumulative additions included), so the
comparison checks that no public result changed, including float rounding.
"""

import math
import os
import pickle
import random
import sys
import time
import tracemalloc

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svgpathtools.path import (Path, Line, QuadraticBezier, CubicBezier, Arc,
                               BugException)


# --------------------------------------------------------------------------
# Reference (old) implementations: exact copies of the pre-change logic.
# --------------------------------------------------------------------------

def old_point(path, pos):
    if len(path._segments) == 0:
        raise ValueError("This path contains no segments!")
    if pos == 0.0:
        return path._segments[0].point(pos)
    if pos == 1.0:
        return path._segments[-1].point(pos)

    path._calc_lengths()
    segment_start = 0
    for index, segment in enumerate(path._segments):
        segment_end = segment_start + path._lengths[index]
        if segment_end >= pos:
            segment_pos = (pos - segment_start) / (
                segment_end - segment_start)
            return segment.point(segment_pos)
        segment_start = segment_end
    raise RuntimeError(
        "Something has gone wrong.  Could not compute Path.point({}) for "
        "path {}".format(pos, path))


def old_T2t(path, T):
    if T == 1:
        return len(path) - 1, 1
    if T == 0:
        return 0, 0
    path._calc_lengths()
    T0 = 0
    for seg_idx, seg_length in enumerate(path._lengths):
        T1 = T0 + seg_length
        if T1 >= T:
            t = (T - T0) / seg_length
            return seg_idx, t
        T0 = T1
    assert 0 <= T <= 1
    raise BugException


# --------------------------------------------------------------------------
# Fixed deterministic path construction.
# --------------------------------------------------------------------------

def assert_same(path, T):
    """Both point() and T2t() must produce identical results/exceptions."""
    try:
        old_val, old_exc = old_point(path, T), None
    except BaseException as e:  # noqa: BLE001
        old_val, old_exc = None, type(e)
    try:
        new_val, new_exc = path.point(T), None
    except BaseException as e:  # noqa: BLE001
        new_val, new_exc = None, type(e)
    assert (old_exc, old_val) == (new_exc, new_val), \
        "point divergence at T=%r: %r vs %r" % (T, old_val, new_val)
    try:
        old_t, old_exc = old_T2t(path, T), None
    except BaseException as e:  # noqa: BLE001
        old_t, old_exc = None, type(e)
    try:
        new_t, new_exc = path.T2t(T), None
    except BaseException as e:  # noqa: BLE001
        new_t, new_exc = None, type(e)
    assert (old_exc, old_t) == (new_exc, new_t), \
        "T2t divergence at T=%r: %r vs %r" % (T, old_t, new_t)


def make_mixed_path(n, seed=20261122):
    """A deterministic path of n wildly-varied-length segments mixing all four
    segment types, including some zero-length lines."""
    rng = random.Random(seed)
    segs = []
    cur = 0j
    for i in range(n):
        kind = i % 5
        # Occasionally emit a zero-length segment.
        if i % 97 == 13:
            segs.append(Line(cur, cur))
            continue
        if kind == 0:  # very short line
            end = cur + complex(rng.uniform(1e-9, 1e-6),
                                rng.uniform(-1e-6, 1e-6))
            segs.append(Line(cur, end))
        elif kind == 1:  # very long line
            end = cur + complex(rng.uniform(1e4, 1e6),
                                rng.uniform(-1e3, 1e3))
            segs.append(Line(cur, end))
        elif kind == 2:
            c1 = cur + complex(rng.uniform(-50, 50), rng.uniform(-50, 50))
            c2 = cur + complex(rng.uniform(50, 500), rng.uniform(-500, 500))
            end = cur + complex(rng.uniform(10, 800), rng.uniform(-800, 800))
            segs.append(CubicBezier(cur, c1, c2, end))
        elif kind == 3:
            c = cur + complex(rng.uniform(-40, 40), rng.uniform(-40, 40))
            end = cur + complex(rng.uniform(5, 400), rng.uniform(-400, 400))
            segs.append(QuadraticBezier(cur, c, end))
        else:
            end = cur + complex(rng.uniform(20, 300), rng.uniform(-300, 300))
            r = complex(abs(rng.uniform(10, 200)), abs(rng.uniform(10, 200)))
            try:
                segs.append(Arc(cur, radius=r, rotation=rng.uniform(-90, 90),
                                large_arc=rng.randrange(2),
                                sweep=rng.randrange(2), end=end))
            except AssertionError:
                segs.append(Line(cur, end))
        cur = segs[-1].end
    return Path(*segs)


def make_lines_path(n, seed=424242):
    """n line segments, including extreme length disparity; cheap enough to
    build at 10k scale for the timing benchmark."""
    rng = random.Random(seed)
    segs = []
    cur = 0j
    for i in range(n):
        if i % 1000 == 7:  # occasional giant segment
            step = complex(rng.uniform(1e6, 2e6), rng.uniform(-1e3, 1e3))
        elif i % 50 == 3:  # occasional zero-length segment
            step = 0j
        else:
            step = complex(rng.uniform(0.1, 10), rng.uniform(-10, 10))
        end = cur + step
        segs.append(Line(cur, end))
        cur = end
    return Path(*segs)


# --------------------------------------------------------------------------
# Checks
# --------------------------------------------------------------------------

def check_exact_equivalence():
    print("== correctness: fixed-path pointwise comparison vs old logic ==")
    path = make_mixed_path(500)

    # Include boundaries, interior random T values, and values deliberately
    # placed extremely close to 0 and 1.
    rng = random.Random(7)
    Ts = [0.0, 1.0]
    Ts += [rng.random() for _ in range(5000)]
    Ts += [5e-324, 1e-300, 1e-16, 1 - 1e-16, 1 - 1e-300,
           0.5 + rng.uniform(-1e-12, 1e-12)]
    # Every exact segment boundary is a critical case for bisect semantics.
    path._calc_lengths()
    Ts.extend(path._cum_lengths)

    mismatches = 0
    for T in Ts:
        try:
            p_old = old_point(path, T)
            old_exc = None
        except BaseException as e:  # noqa: BLE001 - compare exact type
            p_old, old_exc = None, type(e)
        try:
            p_new = path.point(T)
            new_exc = None
        except BaseException as e:  # noqa: BLE001
            p_new, new_exc = None, type(e)
        if old_exc is not None or new_exc is not None:
            if old_exc != new_exc:
                mismatches += 1
                print("  point EXCEPTION MISMATCH at T=%r: old=%s new=%s"
                      % (T, old_exc, new_exc))
        elif p_old != p_new:
            mismatches += 1
            print("  point MISMATCH at T=%r: old=%r new=%r"
                  % (T, p_old, p_new))
        try:
            t_old = old_T2t(path, T)
            old_exc = None
        except BaseException as e:  # noqa: BLE001
            t_old, old_exc = None, type(e)
        try:
            t_new = path.T2t(T)
            new_exc = None
        except BaseException as e:  # noqa: BLE001
            t_new, new_exc = None, type(e)
        if old_exc is not None or new_exc is not None:
            if old_exc != new_exc:
                mismatches += 1
                print("  T2t EXCEPTION MISMATCH at T=%r: old=%s new=%s"
                      % (T, old_exc, new_exc))
        elif t_old != t_new:
            mismatches += 1
            print("  T2t MISMATCH at T=%r: old=%r new=%r"
                  % (T, t_old, t_new))
    # t2T round trip for every segment.
    for idx in range(len(path)):
        for t in (0.0, 0.37, 1.0):
            T = path.t2T(idx, t)
            try:
                ref_val = old_point(path, T)
                old_exc = None
            except BaseException as e:  # noqa: BLE001
                ref_val, old_exc = None, type(e)
            try:
                new_val = path.point(T)
                new_exc = None
            except BaseException as e:  # noqa: BLE001
                new_val, new_exc = None, type(e)
            if (old_exc, ref_val) != (new_exc, new_val):
                mismatches += 1
                print("  t2T round-trip MISMATCH idx=%d t=%r" % (idx, t))

    # The cumulative index must be arithmetically identical to sequential
    # accumulation of _lengths (the old rounding order).
    rebuilt = []
    run = 0
    for L in path._lengths:
        run = run + L
        rebuilt.append(run)
    assert list(path._cum_lengths) == rebuilt, "cumulative rounding drift"
    print("  %d T values checked (%d point + %d T2t comparisons),"
          " mismatches: %d" % (len(Ts), len(Ts), len(Ts), mismatches))
    assert mismatches == 0
    return path


def check_edge_cases():
    print("== edge cases: empty path, zero-length, invalid T, disparity ==")

    # Empty path.
    empty = Path()
    try:
        empty.point(0.5)
        raise AssertionError("empty path should raise ValueError")
    except ValueError as e:
        assert "no segments" in str(e)
    assert empty.length() == 0

    # All-zero-length path: every non-shortcut T fails.  For T <= 0 the first
    # (zero-length) segment is selected and the division by the zero segment
    # length raises ZeroDivisionError; positive T values fall through and end
    # in RuntimeError.  The new implementation must reproduce both exactly.
    z = Path(Line(0j, 0j), Line(0j, 0j), Line(0j, 0j))
    assert z.point(0.0) == 0j and z.point(1.0) == 0j
    for bad in (-0.1, 0.5, 0.999, 1.1):
        try:
            old_point(z, bad)
            old_exc = None
        except BaseException as e:  # noqa: BLE001
            old_exc = type(e)
        try:
            z.point(bad)
            new_exc = None
        except BaseException as e:  # noqa: BLE001
            new_exc = type(e)
        assert old_exc == new_exc and old_exc is not None, \
            (bad, old_exc, new_exc)

    # Out-of-range finite T on a normal path: new implementation must raise
    # the same exception type as the old code.  (NaN is outside the documented
    # T domain of [0, 1] and comparison-based searches don't define an order
    # for it, so it is excluded from the compatibility matrix.)
    p = Path(Line(0, 10), Line(10, 20))
    p._calc_lengths()
    for bad in (-0.5, 1.0000000000000002, 2.0):
        try:
            old_point(p, bad)
            old_exc = None
        except BaseException as e:  # noqa: BLE001 - compare exact type below
            old_exc = type(e)
        try:
            p.point(bad)
            new_exc = None
        except BaseException as e:  # noqa: BLE001
            new_exc = type(e)
        assert old_exc == new_exc, (bad, old_exc, new_exc)

    # T2t out of range: old code returns the first segment for negative T
    # (its first cumulative end already satisfies `>= T`) and hits the final
    # assert for T > 1; the new implementation must behave identically.
    for bad in (-0.1, 1.5):
        try:
            old_res = old_T2t(p, bad)
            old_exc = None
        except BaseException as e:  # noqa: BLE001
            old_res, old_exc = None, type(e)
        try:
            new_res = p.T2t(bad)
            new_exc = None
        except BaseException as e:  # noqa: BLE001
            new_res, new_exc = None, type(e)
        assert (old_exc, old_res) == (new_exc, new_res), \
            (bad, old_exc, old_res, new_exc, new_res)

    # Extreme length disparity (1e-9 next to 1e6): every query still agrees.
    rng = random.Random(11)
    big = make_mixed_path(300, seed=99)
    for _ in range(2000):
        T = rng.random()
        try:
            old_val, old_exc = old_point(big, T), None
        except BaseException as e:  # noqa: BLE001
            old_val, old_exc = None, type(e)
        try:
            new_val, new_exc = big.point(T), None
        except BaseException as e:  # noqa: BLE001
            new_val, new_exc = None, type(e)
        assert (old_exc, old_val) == (new_exc, new_val), T
        try:
            old_t, old_exc = old_T2t(big, T), None
        except BaseException as e:  # noqa: BLE001
            old_t, old_exc = None, type(e)
        try:
            new_t, new_exc = big.T2t(T), None
        except BaseException as e:  # noqa: BLE001
            new_t, new_exc = None, type(e)
        assert (old_exc, old_t) == (new_exc, new_t), T
    print("  all edge cases match old behavior")


def check_invalidation():
    print("== invalidation: insert/delete/slice replace/control-point edit ==")
    path = make_mixed_path(120, seed=555)
    path._ensure_length_index()
    assert path._cum_lengths is not None

    # In-place control point change must invalidate lazily on next query.
    T = 0.42
    before = path.point(T)
    path[3].start = path[3].start + 100 + 100j
    assert path._cum_lengths is None, "listener did not invalidate index"
    after = path.point(T)
    assert after != before
    # Rebuilt index now matches a from-scratch reference path's arithmetic.
    ref = Path(*list(path))
    for _ in range(500):
        assert_same(ref, random.random())

    # Arc geometry edit: the path-level index must be invalidated and rebuilt
    # from the object's current length (Arc.length() itself re-hashes its
    # geometry, so it never serves a stale segment length).
    arc = Arc(0, 50 + 50j, 30, 0, 1, 100 + 0j)
    arc_path = Path(arc, Line(100, 200))
    arc_path.point(0.7)
    arc.radius = 70 + 20j
    assert arc_path._cum_lengths is None
    # Rebuilt via the public API, then compare against a fresh index built
    # from the same object state.
    rebuilt_vals = {tt: arc_path.point(tt)
                    for tt in (0.0, 0.1, 0.5, 0.9, 1.0)}
    ref_arc_path = Path(*list(arc_path))
    for tt, val in rebuilt_vals.items():
        assert val == old_point(ref_arc_path, tt), tt
    # And the length really changed versus the unmodified arc.
    orig = Path(Arc(0, 50 + 50j, 30, 0, 1, 100 + 0j), Line(100, 200))
    assert arc_path.length() != orig.length()

    # Structural mutations via MutableSequence interface.
    path.insert(0, Line(-100j, 0j))
    assert path._cum_lengths is None
    path.point(0.3)
    assert path._cum_lengths is not None
    del path[5]
    assert path._cum_lengths is None
    path.point(0.3)
    # Slice replacement goes through __setitem__/__delitem__/insert.
    path[2:5] = [Line(0j, 1j), Line(1j, 2j)]
    assert path._cum_lengths is None
    ref2 = Path(*list(path))
    for _ in range(300):
        assert_same(ref2, random.random())
    # append / extend / pop (MutableSequence built-ins).
    path.append(Line(path[-1].end, path[-1].end + 50j))
    assert path._cum_lengths is None
    path.point(0.99)
    path.extend([Line(path[-1].end, path[-1].end + 1j)])
    assert path._cum_lengths is None
    path.point(0.99)
    path.pop()
    assert path._cum_lengths is None
    print("  all invalidation paths rebuild correctly")


def check_pickle_and_memory():
    print("== pickle/deepcopy compatibility and index memory footprint ==")
    import copy
    path = make_lines_path(2000)
    path.point(0.5)  # populate index
    data = pickle.dumps(path)
    restored = pickle.loads(data)
    # Restored paths drop the runtime listener-bound index and rebuild it.
    assert restored._cum_lengths is None
    for T in (0.0, 0.25, 0.5, 0.75, 1.0):
        assert restored.point(T) == path.point(T)
    assert restored._cum_lengths is not None
    # An in-place segment edit after restoration must invalidate correctly.
    restored[0].end = restored[0].end + 1j
    assert restored._cum_lengths is None
    restored.point(0.5)

    # deepcopy: independent copy that rebuilds and re-binds its own segments.
    dup = copy.deepcopy(path)
    assert dup._cum_lengths is None
    assert dup.point(0.37) == path.point(0.37)
    dup[1].start = dup[1].start + 2j
    assert dup._cum_lengths is None
    assert path._cum_lengths is not None  # source untouched

    # Simulate a pickle written by the pre-index release: it only carries
    # _length/_lengths once warmed, and has none of the new attributes.
    legacy = make_lines_path(500)
    legacy._calc_lengths()
    legacy._cum_lengths = None
    for seg in legacy:  # old pickles predate _geom_version too
        if hasattr(seg, '_geom_version'):
            object.__delattr__(seg, '_geom_version')
    legacy_data = pickle.dumps(legacy)
    legacy_restored = pickle.loads(legacy_data)
    assert legacy_restored._cum_lengths is None
    ref = Path(*list(legacy_restored))
    rng = random.Random(3)
    for _ in range(1000):
        T = rng.random()
        assert legacy_restored.point(T) == old_point(ref, T)
    assert legacy_restored._cum_lengths is not None
    # A never-warmed (fully cold) old pickle as well.
    cold = make_lines_path(100)
    cold._length = cold._lengths = cold._cum_lengths = None
    cold_restored = pickle.loads(pickle.dumps(cold))
    for T in (0.1, 0.5, 0.9):
        assert cold_restored.point(T) == old_point(Path(*list(cold)), T)

    # Index memory: one compact C array of 8-byte doubles, i.e. 8
    # bytes/segment plus fixed array overhead; assert a bounded footprint of
    # at most 16 bytes/segment for any path size.
    n = len(path)
    idx_bytes = sys.getsizeof(path._cum_lengths)
    print("  index memory for n=%d: %d bytes (%.1f bytes/segment, "
          "budget 16/segment)" % (n, idx_bytes, idx_bytes / n))
    assert idx_bytes < 16 * n

    # Listener registry must not keep dead segments alive / must clean up.
    from svgpathtools.path import _geom_listeners
    before = len(_geom_listeners)
    tmp = Path(Line(0, 1), Line(1, 2))
    tmp.point(0.5)
    nkeys = len(_geom_listeners)
    assert nkeys >= before + 2
    del tmp
    import gc
    gc.collect()
    assert len(_geom_listeners) == before, "dead segment leaked in registry"
    print("  pickle round-trip and registry cleanup OK")


def benchmark_10k():
    print("== benchmark: 10k-segment path, 10k random point() queries ==")
    n = 10000
    path = make_lines_path(n)
    rng = random.Random(2026)
    Ts = [rng.random() for _ in range(n)]

    # Populate the shared length cache once (identical work for both impls).
    path._calc_lengths()
    assert len(path) == n

    # Correctness on the benchmark path itself.
    for T in Ts[:1000]:
        assert_same(path, T)

    # Timed runs, fresh cache state each time so the measured work is pure
    # location + evaluation.
    t0 = time.perf_counter()
    sink_old = 0j
    for T in Ts:
        sink_old += old_point(path, T)
    t_old = time.perf_counter() - t0

    t0 = time.perf_counter()
    sink_new = 0j
    for T in Ts:
        sink_new += path.point(T)
    t_new = time.perf_counter() - t0
    assert sink_old == sink_new

    # Allocation: peak bytes allocated while answering queries.
    tracemalloc.start()
    for T in Ts:
        old_point(path, T)
    _, peak_old = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    tracemalloc.start()
    for T in Ts:
        path.point(T)
    _, peak_new = tracemalloc.get_traced_memory()
    tracemalloc.stop()

    print("  old linear scan : %.4f s  peak alloc %d B" % (t_old, peak_old))
    print("  new bisect index: %.4f s  peak alloc %d B" % (t_new, peak_new))
    print("  speedup: %.1fx" % (t_old / t_new))
    # The scan is O(n) per query versus O(log n); on a 10k-segment path the
    # measured ratio is typically two orders of magnitude.  Require a
    # conservative lower bound so the check stays meaningful on slow CI.
    assert t_new < t_old / 25, "expected a substantial speedup at n=1e4"
    assert peak_new <= peak_old
    return t_old, t_new, peak_old, peak_new


def main():
    check_exact_equivalence()
    check_edge_cases()
    check_invalidation()
    check_pickle_and_memory()
    benchmark_10k()
    print("\nALL CHECKS PASSED")


if __name__ == "__main__":
    main()
