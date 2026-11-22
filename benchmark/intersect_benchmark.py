"""Benchmark: spatial-index-accelerated ``Path.intersect`` vs the original
nested-loop implementation.

Scenarios
---------
1. sparse/corner overlap: two large maps whose bounding boxes only overlap
   in a small corner region (the motivating case -- intersect previously
   inspected nearly every segment pair);
2. dense/overlapping maps: same-sized maps drawn on top of each other;
3. fully overlapping boxes: every segment shares one bbox (index degrades
   gracefully to brute force);
4. repeated queries: verifies the cached index is reused.

For every scenario the script records the segment counts, total possible
pairs, conservative candidate pairs, exact intersection counts, wall time
of both implementations, result equivalence, and peak Python heap
allocation (``tracemalloc``) of the accelerated path.

Run with::

    python3 benchmark/intersect_benchmark.py
"""

import os
import sys
import time
import tracemalloc
from random import Random

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from svgpathtools import CubicBezier, Line, Path, QuadraticBezier


def brute_force_intersect(path1, path2, justonemode=False, tol=1e-12):
    """The pre-optimization O(n*m) nested-loop intersection routine."""
    intersection_list = []
    for seg1 in path1:
        for seg2 in path2:
            if justonemode and intersection_list:
                return intersection_list[0]
            for t1, t2 in seg1.intersect(seg2, tol=tol):
                T1 = path1.t2T(seg1, t1)
                T2 = path2.t2T(seg2, t2)
                intersection_list.append(
                    ((T1, seg1, t1), (T2, seg2, t2)))
    if justonemode and intersection_list:
        return intersection_list[0]
    if intersection_list:
        pts = [item[0][1].point(item[0][2]) for item in intersection_list]
        indices2remove = set()
        for ind1 in range(len(pts)):
            for ind2 in range(ind1 + 1, len(pts)):
                if abs(pts[ind1] - pts[ind2]) < tol:
                    indices2remove.add(ind2)
        intersection_list = [inter for ind, inter in
                             enumerate(intersection_list)
                             if ind not in indices2remove]
    return intersection_list


def random_segments(rng, n, scale, offset=(0.0, 0.0)):
    """Lines and Bezier curves clustered inside a scale-by-scale box."""
    ox, oy = offset
    segs = []
    for _ in range(n):
        kind = rng.randint(0, 2)
        pts = [complex(rng.uniform(0, scale) + ox,
                       rng.uniform(0, scale) + oy) for _ in range(4)]
        if kind == 0:
            segs.append(Line(pts[0], pts[1]))
        elif kind == 1:
            segs.append(QuadraticBezier(pts[0], pts[1], pts[2]))
        else:
            segs.append(CubicBezier(pts[0], pts[1], pts[2], pts[3]))
    return segs


def time_run(fn, repeat=3):
    """Return (best_seconds, result) over `repeat` runs."""
    best = float('inf')
    result = None
    for _ in range(repeat):
        start = time.perf_counter()
        result = fn()
        best = min(best, time.perf_counter() - start)
    return best, result


def measure_memory(fn):
    """Return (peak_bytes, result) of Python-level heap allocations."""
    tracemalloc.start()
    try:
        result = fn()
        _, peak = tracemalloc.get_traced_memory()
    finally:
        tracemalloc.stop()
    return peak, result


def index_memory(path):
    """Deep-ish size estimate of a built spatial index in bytes."""
    idx = path.spatial_index()
    total = sys.getsizeof(idx) + sys.getsizeof(idx.bboxes)
    total += sum(sys.getsizeof(b) for b in idx.bboxes)
    total += sys.getsizeof(idx._cells)
    for key, bucket in idx._cells.items():
        total += sys.getsizeof(key) + sys.getsizeof(bucket)
        total += sum(sys.getsizeof(v) for v in bucket)
    return total, idx


def run_scenario(name, path1, path2, expect_degraded=None, repeat=3):
    n1, n2 = len(path1), len(path2)
    total_pairs = n1 * n2

    # Fresh indices for this measurement (the Path-level cache keeps them
    # across the repeated intersect calls below).
    path1.invalidate_spatial_index()
    path2.invalidate_spatial_index()
    t0 = time.perf_counter()
    idx1 = path1.spatial_index()
    t_build1 = time.perf_counter() - t0
    t0 = time.perf_counter()
    idx2 = path2.spatial_index()
    t_build2 = time.perf_counter() - t0
    t0 = time.perf_counter()
    candidates = idx1.candidate_pairs(idx2)
    t_query = time.perf_counter() - t0
    n_cand = len(candidates)

    t_fast, fast_result = time_run(
        lambda: path1.intersect(path2), repeat=repeat)
    t_slow, slow_result = time_run(
        lambda: brute_force_intersect(path1, path2), repeat=repeat)
    t_justone, justone_result = time_run(
        lambda: path1.intersect(path2, justonemode=True), repeat=repeat)
    justone_expected = brute_force_intersect(
        path1, path2, justonemode=True)

    peak_mem, _ = measure_memory(lambda: path1.intersect(path2))
    idx1_mem, built1 = index_memory(path1)
    idx2_mem, built2 = index_memory(path2)

    equivalent = fast_result == slow_result
    justone_ok = justone_result == justone_expected

    if expect_degraded is not None:
        assert built1.degraded == expect_degraded, \
            'index1 degraded=%r, expected %r' % (
                built1.degraded, expect_degraded)
        assert built2.degraded == expect_degraded

    speedup = t_slow / t_fast if t_fast > 1e-6 else None
    return {
        'name': name,
        'n1': n1, 'n2': n2,
        'total_pairs': total_pairs,
        'candidate_pairs': n_cand,
        'screened_pct': (100.0 * (1 - n_cand / total_pairs)
                         if total_pairs else 100.0),
        'intersections': len(fast_result),
        'equivalent': equivalent,
        'justonemode_ok': justone_ok,
        'degraded1': built1.degraded,
        'degrade_reason1': built1.degrade_reason,
        'degraded2': built2.degraded,
        'degrade_reason2': built2.degrade_reason,
        'time_index_build_s': t_build1 + t_build2,
        'time_candidate_query_s': t_query,
        'time_fast_s': t_fast,
        'time_bruteforce_s': t_slow,
        'time_justonemode_s': t_justone,
        'speedup': speedup,
        'peak_heap_bytes': peak_mem,
        'index1_bytes': idx1_mem,
        'index2_bytes': idx2_mem,
    }


def main():
    rng = Random(20261122)
    # Sparse maps are the motivating case and stay large; dense/overlap
    # scenarios use fewer segments because the library's exact Bezier
    # intersection solver (recursive subdivision) dominates the runtime
    # once most pairs survive screening.
    n_sparse = 2000
    n_dense = 60
    n_overlap = 80
    scale = 1000.0
    results = []

    # 1. Sparse: two maps that only meet in a small corner region.
    left = Path(*random_segments(rng, n_sparse, scale, offset=(0.0, 0.0)))
    right_segs = random_segments(rng, n_sparse, scale,
                                 offset=(scale * 5, scale * 5))
    # A single short bridge segment connects the two maps across the gap
    # near one corner: it overlaps only a handful of each map's segments,
    # so nearly all n*n pairs are screened, but real intersections exist.
    gap = scale * 5
    bridge = Line(complex(scale * 0.9, scale * 0.9),
                  complex(gap + scale * 0.1, gap + scale * 0.1))
    right_segs.append(bridge)
    right = Path(*right_segs)
    results.append(run_scenario('sparse corner overlap', left, right,
                                repeat=3))

    # 2. Dense: two equally large maps drawn on the same area.
    dense1 = Path(*random_segments(rng, n_dense, scale, offset=(0.0, 0.0)))
    dense2 = Path(*random_segments(rng, n_dense, scale, offset=(0.0, 0.0)))
    results.append(run_scenario('dense overlapping maps', dense1, dense2,
                                repeat=1))

    # 3. Fully overlapping bounding boxes: grid must degrade.  The lines
    #    share one bbox but are parallel and disjoint, so the runtime
    #    measures the degraded-mode screening itself rather than the
    #    pre-existing O(k^2) redundancy removal inside the exact routine
    #    (which is independent of this optimization and blows up when many
    #    pairs report the same coincident point).
    overlap_segs1 = [Line(0 + (k * 1e-6) * 1j, 1 + (k * 1e-6) * 1j)
                     for k in range(n_overlap)]
    overlap_segs2 = [Line(0 + 0.5j + (k * 1e-6) * 1j,
                          1 + 0.5j + (k * 1e-6) * 1j)
                     for k in range(n_overlap)]
    results.append(run_scenario(
        'fully overlapping bboxes',
        Path(*overlap_segs1), Path(*overlap_segs2),
        expect_degraded=True, repeat=2))

    # 4. Empty path edge case.
    empty = Path()
    empty_res = run_scenario(
        'empty path', empty,
        Path(*random_segments(rng, 100, scale)), repeat=1)
    results.append(empty_res)

    # 5. Index reuse across repeated queries.
    reused_before = left.spatial_index()
    left.intersect(right)
    left.intersect(right)
    reused_after = left.spatial_index()
    reuse_ok = reused_before is reused_after

    # 6. justonemode on the sparse maps was measured inside the scenarios.

    widths = (27, 8, 8, 12, 12, 10, 9, 10, 10, 9, 10)
    header = ('scenario'.ljust(widths[0]) +
              'n1'.rjust(widths[1]) + 'n2'.rjust(widths[2]) +
              'all pairs'.rjust(widths[3]) +
              'candidates'.rjust(widths[4]) +
              'screened%'.rjust(widths[5]) +
              'inters.'.rjust(widths[6]) +
              'fast s'.rjust(widths[7]) +
              'brute s'.rjust(widths[8]) +
              'speedup'.rjust(widths[9]) +
              'peak KiB'.rjust(widths[10]))
    lines = [header, '-' * len(header)]
    for r in results:
        speedup = 'N/A' if r['speedup'] is None \
            else ('%.1fx' % r['speedup'])
        lines.append(
            r['name'][:widths[0]].ljust(widths[0]) +
            str(r['n1']).rjust(widths[1]) +
            str(r['n2']).rjust(widths[2]) +
            str(r['total_pairs']).rjust(widths[3]) +
            str(r['candidate_pairs']).rjust(widths[4]) +
            ('%.2f' % r['screened_pct']).rjust(widths[5]) +
            str(r['intersections']).rjust(widths[6]) +
            ('%.4f' % r['time_fast_s']).rjust(widths[7]) +
            ('%.4f' % r['time_bruteforce_s']).rjust(widths[8]) +
            speedup.rjust(widths[9]) +
            str(r['peak_heap_bytes'] // 1024).rjust(widths[10]))

    lines.append('')
    lines.append('Index and screening timing (seconds):')
    tw = (27, 14, 16, 14, 14)
    thead = ('scenario'.ljust(tw[0]) +
             'index build'.rjust(tw[1]) +
             'pair query'.rjust(tw[2]) +
             'justonemode'.rjust(tw[3]) +
             'index KiB'.rjust(tw[4]))
    lines += [thead, '-' * len(thead)]
    for r in results:
        lines.append(
            r['name'][:tw[0]].ljust(tw[0]) +
            ('%.5f' % r['time_index_build_s']).rjust(tw[1]) +
            ('%.5f' % r['time_candidate_query_s']).rjust(tw[2]) +
            ('%.5f' % r['time_justonemode_s']).rjust(tw[3]) +
            str((r['index1_bytes'] + r['index2_bytes']) // 1024
                ).rjust(tw[4]))

    lines.append('')
    lines.append('Correctness and edge cases:')
    for r in results:
        lines.append(
            '  %-27s equivalent=%s justonemode=%s degraded=(%s,%s) %s'
            % (r['name'], r['equivalent'], r['justonemode_ok'],
               r['degraded1'], r['degraded2'],
               '[' + str(r['degrade_reason1']) + ']'
               if r['degraded1'] else ''))
    lines.append('  %-27s %s' % (
        'cached index reused', reuse_ok))
    lines.append('  %-27s index memory %d KiB + %d KiB (sparse maps)'
                 % ('', results[0]['index1_bytes'] // 1024,
                    results[0]['index2_bytes'] // 1024))

    report = '\n'.join(lines)
    print('\n' + report + '\n')

    out_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'artifacts')
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, 'intersect_benchmark.txt')
    with open(out_path, 'w') as fh:
        fh.write(report + '\n')
    print('report written to', out_path)

    # Machine-readable companion.
    import json
    json_path = os.path.join(out_dir, 'intersect_benchmark.json')
    with open(json_path, 'w') as fh:
        json.dump({'results': results,
                   'cached_index_reused': reuse_ok}, fh, indent=2)
    print('json written to', json_path)

    assert all(r['equivalent'] for r in results), \
        'accelerated results differ from brute force!'
    assert all(r['justonemode_ok'] for r in results)
    assert reuse_ok


if __name__ == '__main__':
    main()
