r"""Second implementation: independent checks of the signed decomposition of Lemma 2.1 (expansion over Z), Lemma
2.3 (the recursion computes Mult with the coefficients of T4 - E3), Lemma 3.1 with the signed decomposition, and
Proposition 5.1 (the characteristic-3 reading). Output: ../outputs/indep3_lemmas_out.txt.
"""
import itertools, os, sys
import numpy as np
from indep3_gf import GF

out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'indep3_lemmas_out.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()

# letters: left x0,x1,p0 = 0,1,2; right y0,y1,q0 = 0,1,2; output z0,z1,z2,w0 = 0,1,2,3
T4 = np.zeros((3, 3, 4), dtype=np.int64)
for (s, t, o) in [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 2, 3), (2, 1, 3)]:
    T4[s, t, o] = 1
E3 = np.zeros((3, 3, 4), dtype=np.int64)
for (s, t, o) in [(1, 0, 0), (2, 1, 1), (1, 2, 2)]:
    E3[s, t, o] = 1

def forms(sign):
    # signed (sign=-1): Lemma 2.1; plus (sign=+1): the characteristic-2 form
    PHI = np.array([[1, sign, 0], [1, 0, sign], [0, 1, 0], [1, 0, 0]])
    PSI = np.array([[1, 0, 0], [0, 1, 0], [1, 0, sign], [1, 1, 0]])
    CHI = np.array([[1, 0, 0, sign], [0, 1, 0, sign], [0, 0, 1, sign], [0, 0, 0, 1]])
    return PHI, PSI, CHI

say('== Lemma 2.1, signed form (expansion over Z of the rank-4 sum into the 3x3x4 coefficient tensor)')
for sign, name in [(-1, 'signed'), (1, 'plus-sign')]:
    PHI, PSI, CHI = forms(sign)
    G = np.einsum('ls,lt,lo->sto', PHI, PSI, CHI)
    say(name, ': equals T4 - E3:', np.array_equal(G, T4 - E3), '; equals T4 + E3:', np.array_equal(G, T4 + E3))
    if sign == 1:
        X = G - T4 - E3
        say('  plus-sign excess entries (s,t,o,coef):', [(int(s), int(t), int(o), int(X[s, t, o])) for s, t, o in zip(*np.nonzero(X))],
            ' (x0y0w0=(0,0,3), x0y1w0=(0,1,3), x1y0w0=(1,0,3))')
        say('  plus-sign = T4+E3 mod 2:', np.array_equal(G % 2, (T4 + E3) % 2), '; = T4-E3 mod 3:',
            np.array_equal(G % 3, (T4 - E3) % 3))

def encode(arr, M, L):
    # apply M (rows x cols) on every axis; integer arithmetic
    for ax in range(L):
        arr = np.moveaxis(np.tensordot(M, np.moveaxis(arr, ax, 0), axes=(1, 0)), 0, ax)
    return arr

def recursion(a, b, sign, L):
    PHI, PSI, CHI = forms(sign)
    A = encode(a, PHI, L); B = encode(b, PSI, L)
    return encode(A * B, CHI.T, L)

say('== Lemma 2.3: recursion = Mult with gamma = coefficients of T4 - E3 (brute force over all u, v, w at L = 3)')
rng = np.random.default_rng(7)
L = 3
for sign in (-1, 1):
    a = rng.integers(-4, 5, size=(3,) * L); b = rng.integers(-4, 5, size=(3,) * L)
    R = recursion(a, b, sign, L)
    PHI, PSI, CHI = forms(sign)
    gam = np.einsum('ls,lt,lo->sto', PHI, PSI, CHI)
    M = np.zeros((4,) * L, dtype=np.int64)
    for w in itertools.product(range(4), repeat=L):
        s = 0
        for u in itertools.product(range(3), repeat=L):
            for v in itertools.product(range(3), repeat=L):
                c = 1
                for l in range(L):
                    c *= gam[u[l], v[l], w[l]]
                    if c == 0: break
                if c: s += c * a[u] * b[v]
        M[w] = s
    say('sign', sign, ': recursion equals brute-force Mult on all', 4 ** L, 'outputs:', np.array_equal(R, M))

def lstring(r, P, L):
    return tuple(2 if l in P else (1 if l in r else 0) for l in range(L))
def rstring(c, Q, L):
    return tuple(2 if l in Q else (1 if l in c else 0) for l in range(L))
def ostring(r, c, L):
    return tuple(3 if (l in r and l in c) else (2 if l in r else (1 if l in c else 0)) for l in range(L))

say('== Lemma 3.1 (signed decomposition, over Z): Mult[(r,c)] = sum_{P subset S} a[(r,P)] b[(c,S-P)] for all |r|=a0, |c|=b0')
for (L, a0, b0) in [(5, 2, 3), (6, 3, 3), (6, 2, 4), (7, 3, 4), (8, 4, 4), (8, 3, 5)]:
    for sign in (-1, 1):
        a = np.zeros((3,) * L, dtype=np.int64); b = np.zeros((3,) * L, dtype=np.int64)
        for r in itertools.combinations(range(L), a0):
            for k in range(a0 + 1):
                for P in itertools.combinations(r, k):
                    a[lstring(set(r), set(P), L)] = rng.integers(-9, 10)
        for c in itertools.combinations(range(L), b0):
            for k in range(b0 + 1):
                for Q in itertools.combinations(c, k):
                    b[rstring(set(c), set(Q), L)] = rng.integers(-9, 10)
        R = recursion(a, b, sign, L)
        nbad = nbad3 = nbad2 = ntot = 0
        for r in itertools.combinations(range(L), a0):
            for c in itertools.combinations(range(L), b0):
                rs, cs = set(r), set(c); S = sorted(rs & cs)
                tot = 0
                for k in range(len(S) + 1):
                    for P in itertools.combinations(S, k):
                        tot += a[lstring(rs, set(P), L)] * b[rstring(cs, set(S) - set(P), L)]
                val = R[ostring(rs, cs, L)]
                ntot += 1; nbad += (val != tot); nbad3 += ((val - tot) % 3 != 0); nbad2 += ((val - tot) % 2 != 0)
        say(f'L={L} a0={a0} b0={b0} {"signed" if sign < 0 else "plus  "}: outputs {ntot}, wrong over Z {nbad}, '
            f'wrong mod 3 {nbad3}, wrong mod 2 {nbad2}')

say('== Proposition 5.1: sum_P <X,a(P)> <Y,a(S-P)>prime = c(S) X.Y, over GF(3^k), random forms, all 2d-subsets')

def vmul_form(F, v, phi, d):
    """v in V = E[x]/(x^3) (length 3^d, index sum T_i 3^i) times the linear form sum_i phi_i x_i"""
    res = np.zeros_like(v)
    for idx in range(3 ** d):
        c = int(v[idx])
        if not c: continue
        for i in range(d):
            ti = (idx // 3 ** i) % 3
            if ti < 2:
                j = idx + 3 ** i
                res[j] = F.ADD[res[j], F.MUL[c, phi[i]]]
    return res

def cS_bruteforce(F, phis, d):
    """sum over f: S -> [d] with all fibres of size 2 of prod phi_{l,f(l)} (independent of V)"""
    n = len(phis); tot = 0
    for f in itertools.product(range(d), repeat=n):
        if all(f.count(i) == 2 for i in range(d)):
            p = 1
            for l in range(n):
                p = int(F.MUL[p, phis[l][f[l]]])
            tot = int(F.ADD[tot, p])
    return tot

for k, d, L0, trials in [(1, 1, 4, 3), (1, 2, 6, 3), (1, 3, 7, 2), (2, 2, 7, 3), (2, 3, 8, 2), (4, 2, 7, 3), (4, 3, 9, 2), (2, 4, 9, 2), (5, 3, 8, 1)]:
    F = GF(k)
    nS = nbad = nbad_unsigned = nzero = nc_mismatch = 0
    for trial in range(trials):
        phis = [list(rng.integers(0, F.q, size=d)) for _ in range(L0)]
        if trial == 0 and d >= 2:
            phis = [list(rng.integers(0, 3, size=d)) for _ in range(L0)]  # one trial with forms over F_3: some c(S) = 0
        cache = {}
        def aP(P):
            if P in cache: return cache[P]
            if not P:
                v = np.zeros(3 ** d, dtype=np.int64); v[0] = 1
            else:
                v = vmul_form(F, aP(P[:-1]), phis[P[-1]], d)
            cache[P] = v; return v
        idxs = list(range(3 ** d))
        Tof = [tuple((t // 3 ** i) % 3 for i in range(d)) for t in idxs]
        sig = [(-1) ** sum(1 for x in T if x == 1) for T in Tof]
        comp = [sum((2 - T[i]) * 3 ** i for i in range(d)) for T in Tof]
        for S in itertools.combinations(range(L0), 2 * d):
            X = rng.integers(0, F.q, size=3 ** d); Y = rng.integers(0, F.q, size=3 ** d)
            lhs = lhs_u = 0
            for kk in range(2 * d + 1):
                for P in itertools.combinations(S, kk):
                    Pc = tuple(l for l in S if l not in P)
                    vP, vQ = aP(P), aP(Pc)
                    left = 0
                    for t in idxs:
                        left = int(F.ADD[left, F.MUL[X[t], vP[t]]])
                    right = right_u = 0
                    for t in idxs:
                        term = int(F.MUL[Y[t], vQ[comp[t]]])
                        right_u = int(F.ADD[right_u, term])
                        right = int(F.ADD[right, term if sig[t] == 1 else F.NEG[term]])
                    lhs = int(F.ADD[lhs, F.MUL[left, right]])
                    lhs_u = int(F.ADD[lhs_u, F.MUL[left, right_u]])
            cS = int(aP(S)[3 ** d - 1])
            cB = cS_bruteforce(F, [phis[l] for l in S], d)
            xy = 0
            for t in idxs:
                xy = int(F.ADD[xy, F.MUL[X[t], Y[t]]])
            rhs = int(F.MUL[cS, xy])
            nS += 1; nbad += (lhs != rhs); nbad_unsigned += (lhs_u != rhs); nzero += (cS == 0); nc_mismatch += (cS != cB)
    say(f'GF(3^{k}) d={d} levels={L0}: sets x trials {nS}, identity wrong {nbad}, c(S)=a(S)_(2..2) vs brute-force f-sum '
        f'mismatches {nc_mismatch}, c(S)=0 on {nzero}; control unsigned pairing wrong {nbad_unsigned}')
out.close()
