r"""Checks of Lemma 5.2 (isolation) and Corollary 5.3 (hitting sets) on all inner sets S.

Setting (Section 5): V = E[x_1..x_d]/(x_i^3), E of characteristic 3, forms phi_l in E^d, |S| = 2d, and
c(S) = coefficient of prod_i u_i^2 in prod_{l in S} phi_l(u) = sum over f: S -> [d] with |f^{-1}(i)| = 2 for all i of
prod_l phi_{l, f(l)}.
Forms: phi_l = (1, z^l, z^{2l}, ..., z^{(d-1) l}), levels l = 0..L-1, i.e. phi_{l,i} = z^{w(l,i)} with the
strictly Monge weight w(l, i) = l (i - 1). Then c_S(z) = sum_f z^{W(f)}, W(f) = sum_l l (f(l) - 1), a polynomial in
F_3[z]. Lemma 5.2: the minimum of W over f is attained only by the non-increasing f, so
c_S(z) = z^{Wmin(S)} g_S(z) with g_S(0) = 1 and deg g_S = Delta(S) <= (L-1) floor(d^2/2). (The paper numbers the
levels 1..L; the shift changes W by the same constant for every f, so Delta(S) is the same.)
Part A (exact, over F_3[z]): every 2d-set S: the polynomial c_S, its lowest coefficient (must be 1, at Wmin(S)),
         Delta(S), max Delta against the bound.
Part B (exact, in GF(3^k)): evaluate c_S(z) at every z in E^*; per S the number of bad z (<= Delta(S) by root
         counting), the greedy hitting set (rounds), the z that are good for ALL S (one run suffices), and the bound
         log2 C(L,2d) + 1 for the greedy when q - 1 >= 2 Delta (Corollary 5.3(i)).
Usage: python3 char3_isolation.py  ->  ../outputs/char3_isolation_out.txt
"""
import itertools, math, os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gf3k import GF3k

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'char3_isolation_out.txt')
lines = []


def log(s):
    lines.append(s); print(s, flush=True)
    with open(OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


def assignments(d):
    """All exponent vectors alpha in {0..d-1}^{2d} with every value exactly twice (alpha_j = f(s_j) - 1)."""
    base = [i for i in range(d) for _ in range(2)]
    return np.array(sorted(set(itertools.permutations(base))), dtype=np.int64)


def polys(L, d, chunk=10000):
    """For all 2d-subsets S of {0..L-1} (sorted rows), the coefficient vectors (mod 3) of c_S(z), shape (nS, Wtot+1)."""
    Ss = np.array(list(itertools.combinations(range(L), 2 * d)), dtype=np.int64)
    A = assignments(d)                                   # (nf, 2d)
    Wtot = int((L - 1) * d * (d - 1))                    # W(f) <= (L-1) sum_j alpha_j = (L-1) d (d-1)
    nS = len(Ss)
    C = np.zeros((nS, Wtot + 1), dtype=np.int8)
    nmin = np.zeros(nS, dtype=np.int64)
    for s0 in range(0, nS, chunk):
        Sb = Ss[s0:s0 + chunk]
        W = Sb @ A.T                                     # (b, nf)
        b = len(Sb)
        flat = (np.arange(b)[:, None] * (Wtot + 1) + W).reshape(-1)
        cnt = np.bincount(flat, minlength=b * (Wtot + 1)).reshape(b, Wtot + 1)
        C[s0:s0 + b] = (cnt % 3).astype(np.int8)
        nmin[s0:s0 + b] = (W == W.min(axis=1)[:, None]).sum(axis=1)
    return Ss, A, C, nmin


def partA(L, d):
    t0 = time.time()
    Ss, A, C, n_at_min = polys(L, d)
    nS = len(Ss)
    nz = C != 0
    zero_poly = int((~nz.any(axis=1)).sum())
    low = nz.argmax(axis=1)
    high = C.shape[1] - 1 - nz[:, ::-1].argmax(axis=1)
    lowcoef = C[np.arange(nS), low]
    # predicted Wmin: anti-sorted f, i.e. alpha_j = d - ceil(j/2) for sorted s_1 < ... < s_{2d}
    alpha_min = np.array([d - (j + 2) // 2 for j in range(2 * d)], dtype=np.int64)
    alpha_max = np.array([(j + 2) // 2 - 1 for j in range(2 * d)], dtype=np.int64)
    Wmin = Ss @ alpha_min
    Wmax = Ss @ alpha_max
    # n_at_min: number of assignments attaining the minimum of W (over Z, before reduction mod 3)
    Delta = high - low
    bound = (L - 1) * (d * d // 2)
    log(f'(A) L={L} d={d}: {nS} sets S, {len(A)} assignments f each; c_S identically 0 mod 3: {zero_poly}; '
        f'lowest exponent = predicted Wmin on {int((low == Wmin).sum())}/{nS}; assignments at Wmin = 1 on '
        f'{int((n_at_min == 1).sum())}/{nS}; lowest coefficient = 1 on {int((lowcoef == 1).sum())}/{nS}; '
        f'highest exponent = predicted Wmax on {int((high == Wmax).sum())}/{nS}; max Delta(S) = {int(Delta.max())}, '
        f'bound (L-1) floor(d^2/2) = {bound}; Delta(S) <= bound on all: {bool((Delta <= bound).all())} '
        f'[{time.time() - t0:.1f}s]')
    return Ss, C, low, Delta


def partB(L, d, k, Ss, C, low, Delta, chunk=4000):
    t0 = time.time()
    F = GF3k(k, tables=False)
    q = F.q
    nS, Wl = C.shape
    # POW[W, e*k + i] = digit i of g^{e W}
    e = np.arange(q - 1)
    Wr = np.arange(Wl)
    expo = (Wr[:, None] * e[None, :]) % (q - 1)          # (Wl, q-1)
    POW = F.DIG[F.exp[expo]].astype(np.float32)          # (Wl, q-1, k)
    POW = POW.reshape(Wl, (q - 1) * k)
    bad = np.zeros((nS, q - 1), dtype=bool)
    Cf = C.astype(np.float32)
    for s0 in range(0, nS, chunk):
        val = Cf[s0:s0 + chunk] @ POW                    # exact small integers
        val = (np.rint(val).astype(np.int64) % 3).reshape(-1, q - 1, k)
        bad[s0:s0 + chunk] = ~(val != 0).any(axis=2)
    nbad = bad.sum(axis=1)
    root_ok = bool((nbad <= Delta).all())
    good_all = int((~bad.any(axis=0)).sum())
    frac = bad.mean()
    # greedy hitting set over E^* (deterministic: ties broken by the smallest exponent e)
    unc = np.ones(nS, dtype=bool); chosen = []
    while unc.any() and len(chosen) < q - 1:
        gain = (~bad[unc]).sum(axis=0)
        j = int(gain.argmax())
        if gain[j] == 0:
            break
        chosen.append(j); unc &= bad[:, j]
    # explicit schedule: z = g^0, g^1, ..., first Delta_max + 1 of them (need q - 1 >= Delta_max + 1)
    Dm = int(Delta.max())
    expl = None
    if q - 1 >= Dm + 1:
        expl_unc = bad[:, :Dm + 1].all(axis=1).sum()
        # how many of the explicit list are needed in order (first prefix that covers all)
        cov = np.cumsum(~bad, axis=1) > 0
        need = int(np.argmax(cov.all(axis=0))) + 1 if cov.all(axis=0).any() else None
        expl = (int(expl_unc), need)
    bound_g = math.floor(math.log2(math.comb(L, 2 * d))) + 1
    log(f'(B) L={L} d={d} GF(3^{k}) (q={q}, h={F.h}): bad (S, z) fraction {frac:.5f} (1/q = {1 / q:.5f}); max bad z per '
        f'S {int(nbad.max())} (<= Delta(S) for all S: {root_ok}); z good for ALL S: {good_all} of {q - 1}; greedy '
        f'hitting set: {len(chosen)} z (uncovered {int(unc.sum())}), exponents {chosen[:8]}; '
        f'explicit list g^0..g^Delta: ' + ('n/a (q too small)' if expl is None else
                                           f'uncovered {expl[0]}, prefix needed {expl[1]}') +
        f'; greedy bound log2 C(L,2d)+1 = {bound_g} (valid when q-1 >= 2 Delta = {2 * Dm}: {q - 1 >= 2 * Dm}) '
        f'[{time.time() - t0:.1f}s]')
    return bad


def main():
    log('Characteristic 3: Monge-weighted Vandermonde forms phi_l = (z^{l(i-1)})_i, l = 0..L-1 (char3_isolation.py)')
    cases = [(8, 1, [1, 2, 3]), (12, 2, [2, 3, 4]), (16, 2, [2, 3, 4, 5, 6, 7]), (12, 3, [3, 4, 5, 6]),
             (16, 3, [4, 5, 6]), (24, 3, [4, 5, 6]), (20, 4, [4, 5])]
    for L, d, ks in cases:
        Ss, C, low, Delta = partA(L, d)
        for k in ks:
            if len(Ss) * (3 ** k) > 3e8:
                log(f'(B) L={L} d={d} GF(3^{k}): skipped (size)'); continue
            partB(L, d, k, Ss, C, low, Delta)


if __name__ == '__main__':
    main()
