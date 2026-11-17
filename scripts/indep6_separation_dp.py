r"""Independent exact (not random) check of Lemma 6.2 (separation) at the parameters of Theorem 1.2, L' = 3d,
A = ceil(3d/2), for d = 1..D.
Claim checked: for every output string w of the band types and every pair (u, v) with R(u) of the row type and C(v)
of the column type, a nonzero coefficient prod_l gamma_l(u_l, v_l, w_l) forces a monomial of T6 (T6') at EVERY level.
Reduction: the type conditions are counts per kind of level, so a violation exists iff, for one kind, some multiset of
output letters allowed by the band types (a 2x3 table of cell counts with row sums (A, 3d-A) and column sums (d,d,d)) can
be matched by a choice of one product monomial per level, at least one of them an error monomial, with row-block counts of
the A letters (A, 3d-A) and column-block counts of the B letters (d,d,d). In T6 coordinates the kind-II condition is the
same as the kind-I one (the left string carries the y letters there, the right string the x letters), so one DP suffices.
Controls: (1) drop the column-type condition on v; (2) drop the row-type condition on u. Both must show violations.
The product monomials are expanded from ../data/T6_decomposition.json by this script.
Usage: python3 indep6_separation_dp.py [D] > ../outputs/indep6_separation_dp_out.txt   (D = 12 by default)
"""
import json, itertools, os, sys
from functools import lru_cache
p = 3
rec = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'T6_decomposition.json')))
terms = rec['terms']
P = {}
for a, b, c in terms:
    for (x, cx), (y, cy), (z, cz) in itertools.product(a.items(), b.items(), c.items()):
        P[(x, y, z)] = (P.get((x, y, z), 0) + cx * cy * cz) % p
P = {k: v for k, v in P.items() if v}
T = {('o0', 'y0', 'w00'), ('p0', 'q0', 'w00'), ('x0', 'r0', 'w00'), ('x0', 'y1', 'z01'), ('x0', 'y2', 'z02'),
     ('x1', 'y0', 'z10'), ('x1', 'y1', 'z11'), ('x1', 'y2', 'z12')}
rb = {'x0': 0, 'p0': 0, 'o0': 0, 'x1': 1}
cb = {'y0': 0, 'q0': 0, 'r0': 0, 'y1': 1, 'y2': 2}
cells = [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)]
zof = {(0, 0): 'w00', (0, 1): 'z01', (0, 2): 'z02', (1, 0): 'z10', (1, 1): 'z11', (1, 2): 'z12'}
# options per output letter: (row block of A letter, column block of B letter, is error monomial)
opts = {z: sorted({(rb[s], cb[t], int((s, t, z) not in T)) for (s, t, zz) in P if zz == z}) for z in zof.values()}

def tables(d, A):
    Lp = 3 * d
    for n00 in range(0, d + 1):
        for n01 in range(0, d + 1):
            n02 = A - n00 - n01
            if not (0 <= n02 <= d): continue
            n10, n11, n12 = d - n00, d - n01, d - n02
            if min(n10, n11, n12) < 0: continue
            yield {(0, 0): n00, (0, 1): n01, (0, 2): n02, (1, 0): n10, (1, 1): n11, (1, 2): n12}

def violates(d, A, tab, use_row=True, use_col=True):
    """is there a choice with >= 1 error monomial meeting the count conditions?"""
    seq = []
    for c in cells:
        seq += [zof[c]] * tab[c]
    Lp = len(seq)
    @lru_cache(maxsize=None)
    def go(i, r0, c0, c1, flag):
        if i == Lp:
            ok = flag and (not use_row or r0 == A) and (not use_col or (c0 == d and c1 == d))
            return ok
        for (r, c, e) in opts[seq[i]]:
            nr0, nc0, nc1 = r0 + (r == 0), c0 + (c == 0), c1 + (c == 1)
            if use_row and nr0 > A: continue
            if use_col and (nc0 > d or nc1 > d): continue
            if go(i + 1, nr0, nc0, nc1, flag or e):
                return True
        return False
    return go(0, 0, 0, 0, 0)

print('options per output letter (row block of A letter, column block of B letter, error?):')
for z, o in opts.items():
    print('  %s: %s' % (z, o))
Dmax = int(sys.argv[1]) if len(sys.argv) > 1 else 12
for d in range(1, Dmax + 1):
    A = -(-3 * d // 2)
    tabs = list(tables(d, A))
    v = sum(violates(d, A, t) for t in tabs)
    c1 = sum(violates(d, A, t, use_col=False) for t in tabs)
    c2 = sum(violates(d, A, t, use_row=False) for t in tabs)
    print('d=%2d L=%3d A=%2d: output cell tables %3d, tables with a violation %d | controls: no column type %d, no row type %d'
          % (d, 6 * d, A, len(tabs), v, c1, c2), flush=True)
