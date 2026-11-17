r"""Cross-check of the two formula implementations at the parameters of Theorem 1.2: t6_counts.counts(d) (Table 5 as
written) against t6_impl.formula (the formula that t6_impl.py compares with the executed counts), for d = 1, 2, 3,
B = 1 and B = 2. No algorithm is run.
Usage: python3 t6_formula_xcheck.py > ../outputs/t6_formula_xcheck_out.txt
"""
import sys
import t6_counts as RC, t6_impl as RA
T6 = RA.T6_kind(); T6t = RA.transpose_kind(T6)
for d in (1, 2, 3):
    for B in (1, 2):
        Lp = 3 * d; A = -(-3 * d // 2); L = 6 * d
        I = list(range(Lp)); II = list(range(Lp, L))
        rows = [(I, {0: A, 1: Lp - A}), (II, {0: d, 1: d, 2: d})]
        cols = [(I, {0: d, 1: d, 2: d}), (II, {0: A, 1: Lp - A})]
        cfg = dict(kinds=[T6] * Lp + [T6t] * Lp, rows=rows, cols=cols)
        V = RA.Alg(RA.GF(3, 1, [0]), d, 3)
        N0 = RC.counts(d)['N0']
        Rw = RA.words(rows) if d <= 2 else None
        rho0 = list(range(A)) + list(range(Lp, Lp + d)); kap0 = list(range(d)) + list(range(Lp, Lp + A))
        K = 2 * Lp * (Lp - 1)
        fa = RA.formula(cfg, cfg['kinds'], d, 3, 6, L, N0, N0, B, B, K, rho0, kap0, V)
        fc = RC.counts(d, B)['c']
        # t6_counts splits step 0 into three rows; t6_impl has one row '0 set-up'
        s0 = tuple(sum(fc[k][i] for k in fc if k.startswith('0')) for i in range(3))
        same = fa['0 set-up'] == s0 and all(fa[k] == fc[k] for k in fa if not k.startswith('0'))
        if Rw is not None:
            same &= len(Rw) == N0
        print('d=%d B=%d: N0=%d, formulas agree: %s' % (d, B, N0, same))
