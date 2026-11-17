r"""Lemma 6.4 (covering) at the parameters of Theorem 1.2 (L' = 3d, A = ceil(3d/2)): every quadruple of high-block
sets (r1 of size A, c1 of size d, r2 of size d, c2 of size A in [L']) is covered by the path Pi^d, i.e. some
member (pi1, pi2) has |r1 cap pi1(c1)| + |r2 cap pi2(c2)| = d. Exhaustive at d = 2 (only the sets matter, not the
other blocks), random sample at d = 3, 4. Also the minimum number of hitting members.
Usage: python3 t6_cover_check.py > ../outputs/t6_cover_check_out.txt
"""
import itertools, random, sys
from t6_impl import cyc_family

def path(Lp):
    sk, pre = cyc_family(Lp)
    comp = lambda a, b: [a[b[l]] for l in range(Lp)]
    fam = []
    for k in range(Lp):
        for j in range(Lp - 1): fam.append((comp(sk[k], pre[j]), sk[k]))
        for j in range(Lp - 1): fam.append((sk[(k + 1) % Lp], comp(sk[k], pre[j])))
    return fam

def img(pi, S): return sum(1 << pi[l] for l in range(len(pi)) if S >> l & 1)
def sets(n, k): return [sum(1 << i for i in c) for c in itertools.combinations(range(n), k)]

random.seed(3)
for d in (2, 3, 4):
    Lp = 3 * d; A = -(-3 * d // 2); fam = path(Lp)
    R1, C1, R2, C2 = sets(Lp, A), sets(Lp, d), sets(Lp, d), sets(Lp, A)
    im1 = {c: [img(f[0], c) for f in fam] for c in C1}; im2 = {c: [img(f[1], c) for f in fam] for c in C2}
    if d == 2:
        quads = itertools.product(R1, C1, R2, C2); mode = 'exhaustive'
    else:
        quads = ((random.choice(R1), random.choice(C1), random.choice(R2), random.choice(C2)) for _ in range(200000)); mode = 'random 200000'
    n = bad = 0; minhit = 10 ** 9
    for r1, c1, r2, c2 in quads:
        a, b = im1[c1], im2[c2]
        hits = sum(1 for k in range(len(fam)) if bin(r1 & a[k]).count('1') + bin(r2 & b[k]).count('1') == d)
        n += 1; bad += hits == 0; minhit = min(minhit, hits)
    print('d=%d L=%d K=%d %s: %d quadruples, uncovered %d, min hitting members %d' % (d, 2 * Lp, len(fam), mode, n, bad, minhit), flush=True)
