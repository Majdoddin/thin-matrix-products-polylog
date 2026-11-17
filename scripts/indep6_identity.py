r"""Independent symbolic check of the core identity of Proposition 6.3 (and of its q = p^k form, Section 7) for a
GENERIC d x d matrix Phi (entries are indeterminates), as a polynomial identity over F_p:
  prod_{l=1}^d sum_{e=0}^{q-1} phi_l(xi)^e phi_l(eta)^{q-1-e}  ==  det(Phi)^{q-1} * prod_{i=1}^d sum_{e=0}^{q-1} xi_i^e eta_i^{q-1-e}
in F_p[Phi][xi, eta]/(xi_i^q, eta_i^q), phi_l(xi) = sum_i Phi[l,i] xi_i. The left side is the per-level expansion of
sum_e a(e)(xi) a((q-1)-e)(eta); the right side, after the pairing (xi^T (x) eta^U -> x_T y_{(q-1)-U}), is
det^{q-1} sum_T x_T y_T. Cases: (p, q, d) = (3,3,2), (3,3,3), (3,3,4) [Proposition 6.3], (2,2,3) [Proposition 3.2],
(2,4,2), (3,9,2), (5,5,2) [q = p^k and p > 3, Section 7]. Control: (3,3,2) with slice coefficients (1, 2, 1).
Usage: python3 indep6_identity.py > ../outputs/indep6_identity_out.txt
"""
import itertools, sys
from math import comb

def run(p, q, d, slice_coef=None):
    nv = d * d
    def key(P, X, Y): return (P, X, Y)
    def mul(f, g):
        out = {}
        for (P1, X1, Y1), c1 in f.items():
            for (P2, X2, Y2), c2 in g.items():
                X = tuple(a + b for a, b in zip(X1, X2))
                if max(X) >= q: continue
                Y = tuple(a + b for a, b in zip(Y1, Y2))
                if max(Y) >= q: continue
                P = tuple(a + b for a, b in zip(P1, P2))
                k = (P, X, Y); out[k] = (out.get(k, 0) + c1 * c2) % p
        return {k: c for k, c in out.items() if c}
    def add(f, g):
        out = dict(f)
        for k, c in g.items(): out[k] = (out.get(k, 0) + c) % p
        return {k: c for k, c in out.items() if c}
    zP = (0,) * nv; zX = (0,) * d
    one = {(zP, zX, zX): 1}
    def ent(l, i): return tuple(int(j == l * d + i) for j in range(nv))
    def unit(i): return tuple(int(j == i) for j in range(d))
    lhs = one
    for l in range(d):
        px = {(ent(l, i), unit(i), zX): 1 for i in range(d)}
        py = {(ent(l, i), zX, unit(i)): 1 for i in range(d)}
        powx = [one]; powy = [one]
        for e in range(1, q): powx.append(mul(powx[-1], px)); powy.append(mul(powy[-1], py))
        fac = {}
        for e in range(q):
            c = 1 if slice_coef is None else slice_coef[e]
            term = {k: (v * c) % p for k, v in mul(powx[e], powy[q - 1 - e]).items()}
            fac = add(fac, term)
        lhs = mul(lhs, fac)
    # det(Phi) by Leibniz
    det = {}
    for perm in itertools.permutations(range(d)):
        sgn = 1
        for i in range(d):
            for j in range(i + 1, d):
                if perm[i] > perm[j]: sgn = -sgn
        P = [0] * nv
        for l in range(d): P[l * d + perm[l]] += 1
        k = (tuple(P), zX, zX); det[k] = (det.get(k, 0) + sgn) % p
    det = {k: c for k, c in det.items() if c}
    rhs = one
    for _ in range(q - 1): rhs = mul(rhs, det)
    for i in range(d):
        fac = {(zP, tuple(e * int(j == i) for j in range(d)), tuple((q - 1 - e) * int(j == i) for j in range(d))): 1 for e in range(q)}
        rhs = mul(rhs, fac)
    return lhs == rhs, len(lhs), len(rhs)

for case in ((3, 3, 2), (3, 3, 3), (2, 2, 3), (2, 4, 2), (3, 9, 2), (5, 5, 2), (3, 3, 4)):
    ok, nl, nr = run(*case)
    print('p=%d q=%d d=%d: identity holds as a polynomial identity: %s (terms %d / %d)' % (case + (ok, nl, nr)), flush=True)
ok, nl, nr = run(3, 3, 2, slice_coef=(1, 2, 1))
print('control p=3 q=3 d=2 with slice coefficients (1, 2, 1): identity holds: %s' % ok)
