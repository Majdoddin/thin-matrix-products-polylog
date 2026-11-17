# Case enumeration of the proof of Corollary A.2: a grid tensor with a single high cell (slice rank q >= 2) and n
# outputs in distinct cells of a partial grid, of border rank at most n ("tight").
# Letter counts: dim A >= q + k, dim B >= q + l (k, l = other occupied row / column blocks), tightness: dim A, dim B <= n;
# the n - 1 low outputs fit in the (1+k) x (1+l) grid minus the high cell. alpha(T) = 2 ln q / (n ln n) > 1/4 iff
# q > n^(n/8).
# For each surviving (n, q) list the feasible (k, l), whether a side is forced saturated (q + k = n or q + l = n),
# and the max outputs per row / column (1 + l, 1 + k), for Theorem A.1 (<= 3 cells per row with dim B = n, or per
# column with dim A = n). Output columns: n, q, alpha(T), [(k, l, A saturated, B saturated, cells per row at most,
# cells per column at most, covered by Theorem A.1)], verdict.
# Usage: python3 one_high_cell_cases.py > ../outputs/one_high_cell_cases_out.txt
import math
for n in range(1, 40):
    for q in range(2, n + 1):
        if not 2 * math.log(q) / (n * math.log(n)) > 0.25:
            continue
        feas = [(k, l) for k in range(0, n - q + 1) for l in range(0, n - q + 1) if (1 + k) * (1 + l) >= n]
        if not feas:
            continue
        rows = []
        for k, l in feas:
            satA, satB = q + k == n, q + l == n
            ok24 = (satB and 1 + l <= 3) or (satA and 1 + k <= 3)
            rows.append((k, l, satA, satB, 1 + l, 1 + k, ok24))
        print(n, q, round(2 * math.log(q) / (n * math.log(n)), 4), rows, 'ALL covered by Theorem A.1' if all(r[-1] for r in rows) else 'NOT all covered')
