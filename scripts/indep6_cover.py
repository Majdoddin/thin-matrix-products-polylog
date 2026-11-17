r"""Independent check of Lemma 6.4 with an own construction of the path Pi^d (no code shared with t6_impl.py):
checks that consecutive members differ by one transposition applied on the right of one factor, that g changes by at
most 1 per step (for random sets), that the average of g over the members (s^k, s^k) is 2A/3, and that every
quadruple (r1, c1, r2, c2) (|r1| = A, |c1| = d, |r2| = d, |c2| = A in [L']) is covered: exhaustively for d = 1, 2, 3
(d = 3: 10584^2 = 1.1e8 quadruples, by bitmasks), by 200000 random quadruples for d = 4, 5.
Usage: python3 indep6_cover.py > ../outputs/indep6_cover_out.txt
"""
import itertools, random
import numpy as np
from fractions import Fraction
random.seed(11)

def path(Lp):
    comp = lambda f, g: tuple(f[g[l]] for l in range(Lp))
    s = tuple((l + 1) % Lp for l in range(Lp))
    P = [tuple(range(Lp))]
    for j in range(1, Lp):
        th = list(range(Lp)); th[j - 1], th[j] = j, j - 1
        P.append(comp(P[-1], tuple(th)))
    assert P[-1] == s
    sk = [tuple(range(Lp))]
    for _ in range(Lp): sk.append(comp(sk[-1], s))
    fam = []
    for k in range(Lp):
        fam += [(comp(sk[k], P[j]), sk[k]) for j in range(Lp - 1)]
        fam += [(sk[k + 1], comp(sk[k], P[j])) for j in range(Lp - 1)]
    # one right transposition between consecutive members: f_{i+1} = f_i o theta in exactly one factor
    inv = lambda f: tuple(sorted(range(Lp), key=lambda l: f[l]))
    for a, b in zip(fam, fam[1:]):
        moved = [sum(1 for l in range(Lp) if comp(inv(x), y)[l] != l) for x, y in zip(a, b)]
        assert sorted(moved) == [0, 2], moved
    diag = [i for i, f in enumerate(fam) if f[0] == f[1]]
    assert len(diag) == Lp
    return fam, diag

def gvec(fam, r, c, which):
    return np.array([len(r & {f[which][l] for l in c}) for f in fam])

for d in (1, 2, 3, 4, 5):
    Lp = 3 * d; A = (3 * d + 1) // 2
    fam, diag = path(Lp)
    sets = lambda k: [frozenset(x) for x in itertools.combinations(range(Lp), k)]
    # average over the diagonal members and step bound, for random quadruples
    for _ in range(200):
        r1, c1 = frozenset(random.sample(range(Lp), A)), frozenset(random.sample(range(Lp), d))
        r2, c2 = frozenset(random.sample(range(Lp), d)), frozenset(random.sample(range(Lp), A))
        g = gvec(fam, r1, c1, 0) + gvec(fam, r2, c2, 1)
        assert np.abs(np.diff(g)).max() <= 1
        assert Fraction(int(g[diag].sum()), Lp) == Fraction(2 * A, 3)
    if d <= 3:
        G1 = np.array([gvec(fam, r, c, 0) for r in sets(A) for c in sets(d)])
        G2 = np.array([gvec(fam, r, c, 1) for r in sets(d) for c in sets(A)])
        U1, U2 = np.unique(G1, axis=0), np.unique(G2, axis=0)
        K = len(fam); W = (K + 63) // 64
        def masks(G):
            out = np.zeros((len(G), d + 1, W), dtype=np.uint64)
            for a in range(d + 1):
                for i in range(K):
                    out[:, a, i // 64] |= ((G[:, i] == a).astype(np.uint64) << np.uint64(i % 64))
            return out
        M1, M2 = masks(U1), masks(U2)[:, ::-1, :]          # pair value a with d - a
        unc = 0
        for i in range(0, len(M1), 64):
            hit = (M1[i:i + 64, None] & M2[None]).any(axis=(2, 3))
            unc += int((~hit).sum())
        print('d=%d L=%d K=%d: exhaustive over %d x %d (r,c)-pairs (%d x %d distinct g-vectors): uncovered %d'
              % (d, 2 * Lp, K, len(G1), len(G2), len(U1), len(U2), unc), flush=True)
    else:
        unc = 0
        for _ in range(200000):
            r1, c1 = frozenset(random.sample(range(Lp), A)), frozenset(random.sample(range(Lp), d))
            r2, c2 = frozenset(random.sample(range(Lp), d)), frozenset(random.sample(range(Lp), A))
            g = gvec(fam, r1, c1, 0) + gvec(fam, r2, c2, 1)
            unc += int(not (g == d).any())
        print('d=%d L=%d K=%d: 200000 random quadruples, uncovered %d' % (d, 2 * Lp, len(fam), unc), flush=True)
print('structural checks (one right transposition per step, |step| <= 1, diagonal average 2A/3): passed')
