r"""Independent end-to-end check of single read-out entries at the EXACT layout of Theorem 1.2 (L' = 3d kind-I
levels T6, L' kind-II levels T6', A = ceil(3d/2), row and column types of Section 6), d = 2, 3, 4 (L = 12, 18, 24),
over GF(27) = F_3[x]/(x^3 + 2x + 2) (a different modulus from that of t6_impl.py), with random distinct nodes. Shares
no code with t6_impl.py: own field, own algebra V, own diagonal path Pi^d, own strings.
For a few row words R_k (row type) and column words C_k (column type), and the first member pi of Pi^d with
|S(R_1, pi C_1)| = d, the arrays A[(R_k, e)] = <x_k, a(e)>, B[(pi C_k, f)] = <y_k, a(f)>^ (|supp| <= d) are formed and
  M[w] = sum_{u, v} prod_l gamma_l(u_l, v_l, w_l) A[u] B[v],   gamma_l = coefficient in the FULL six-term product (T + E)
is evaluated by brute force over all pairs (this is what the decoding yields, Lemma 2.3) at w = w(R_k, pi C_k') for
every pair (k, k') with |S| = d. Checks: (i) the nonzero coefficients are exactly the pairs of Lemma 6.2;
(ii) M[w] = prod_{l<l' in S} (t_l - t_l')^2 * <x_k, y_k'> (Proposition 6.3);
(iii) control: the same with <y, v> in place of <y, v>^ must fail.
Usage: python3 indep6_entry.py > ../outputs/indep6_entry_out.txt   (about 1 GB of memory at d = 4)
"""
import json, itertools, os, random, sys
import numpy as np
p = 3
# ---------------------------------------------------------------- GF(27), x^3 = x + 1  (x^3 + 2x + 2 = 0)
def pmul(a, b):
    c = [0] * 5
    for i in range(3):
        for j in range(3):
            c[i + j] += a[i] * b[j]
    c[2] += c[4]; c[1] += c[4]          # x^4 = x^2 + x
    c[1] += c[3]; c[0] += c[3]          # x^3 = x + 1
    return [v % 3 for v in c[:3]]
dig = lambda n: [n % 3, (n // 3) % 3, n // 9]
num = lambda v: v[0] + 3 * v[1] + 9 * v[2]
MUL = np.array([[num(pmul(dig(a), dig(b))) for b in range(27)] for a in range(27)], dtype=np.int64)
ADD = np.array([[num([(x + y) % 3 for x, y in zip(dig(a), dig(b))]) for b in range(27)] for a in range(27)], dtype=np.int64)
SUB = np.array([[num([(x - y) % 3 for x, y in zip(dig(a), dig(b))]) for b in range(27)] for a in range(27)], dtype=np.int64)
assert all(sum(1 for b in range(27) if MUL[a, b] == 1) == 1 for a in range(1, 27)), 'not a field'
COORD = np.array([dig(n) for n in range(27)], dtype=np.int64)
def gsum(arr, axis=-1):
    c = COORD[arr].sum(axis=axis - 1 if axis < 0 else axis) % 3      # coordinates summed
    return c[..., 0] + 3 * c[..., 1] + 9 * c[..., 2]
# ---------------------------------------------------------------- the six-term product
rec = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'T6_decomposition.json')))
terms = rec['terms']
Pc = {}
for a, b, c in terms:
    for (x, cx), (y, cy), (z, cz) in itertools.product(a.items(), b.items(), c.items()):
        Pc[(x, y, z)] = (Pc.get((x, y, z), 0) + cx * cy * cz) % p
Xl = ['x0', 'p0', 'o0', 'x1']; Yl = ['y0', 'q0', 'r0', 'y1', 'y2']
Zl = ['w00', 'z01', 'z02', 'z10', 'z11', 'z12']
zof = {(0, 0): 'w00', (0, 1): 'z01', (0, 2): 'z02', (1, 0): 'z10', (1, 1): 'z11', (1, 2): 'z12'}
xlet = {(0, 0): 'x0', (0, 1): 'p0', (0, 2): 'o0', (1, 0): 'x1'}
ylet = {(0, 0): 'y0', (0, 1): 'q0', (0, 2): 'r0', (1, 0): 'y1', (2, 0): 'y2'}
# gamma tables: kind I index [x letter][y letter][z]; kind II: left letter y, right letter x: same coefficient
G1 = np.zeros((4, 5, 6), dtype=np.int64)
for (x, y, z), c in Pc.items():
    G1[Xl.index(x), Yl.index(y), Zl.index(z)] = c

def run(d, nwords, seed):
    rng = random.Random(seed)
    Lp = 3 * d; L = 2 * Lp; A = -(-3 * d // 2); D = 3 ** d
    nodes = rng.sample(range(27), L)
    mons = list(itertools.product(range(3), repeat=d)); midx = {T: i for i, T in enumerate(mons)}
    dual = np.array([midx[tuple(2 - t for t in T)] for T in mons])
    def vmul(u, v):
        out = np.zeros(D, dtype=np.int64)
        for i, T in enumerate(mons):
            if u[i] == 0: continue
            for j, U in enumerate(mons):
                if v[j] == 0: continue
                W = tuple(a + b for a, b in zip(T, U))
                if max(W) <= 2:
                    k = midx[W]; out[k] = ADD[out[k], MUL[u[i], v[j]]]
        return out
    one = np.zeros(D, dtype=np.int64); one[midx[(0,) * d]] = 1
    phi = []
    for l in range(L):
        v = np.zeros(D, dtype=np.int64); pw = 1
        for i in range(d):
            v[midx[tuple(int(k == i) for k in range(d))]] = pw; pw = MUL[pw, nodes[l]]
        phi.append([one, v, vmul(v, v)])
    cache = {(): one}
    def a_of(e):                      # e: sorted tuple of (level, value)
        if e not in cache:
            cache[e] = vmul(a_of(e[:-1]), phi[e[-1][0]][e[-1][1]])
        return cache[e]
    # words
    def rand_row():
        w1 = [0] * A + [1] * (Lp - A); rng.shuffle(w1)
        w2 = [0] * d + [1] * d + [2] * d; rng.shuffle(w2)
        return tuple(w1 + w2)
    def rand_col():
        w1 = [0] * d + [1] * d + [2] * d; rng.shuffle(w1)
        w2 = [0] * A + [1] * (Lp - A); rng.shuffle(w2)
        return tuple(w1 + w2)
    # own diagonal path
    s = [(l + 1) % Lp for l in range(Lp)]
    comp = lambda f, g: [f[g[l]] for l in range(Lp)]
    Pj = [list(range(Lp))]
    for j in range(1, Lp):
        th = list(range(Lp)); th[j - 1], th[j] = j, j - 1
        Pj.append(comp(Pj[-1], th))
    assert Pj[Lp - 1] == s
    sk = [list(range(Lp))]
    for k in range(1, Lp + 1): sk.append(comp(sk[-1], s))
    path = []
    for k in range(Lp):
        path += [(comp(sk[k], Pj[j]), sk[k]) for j in range(Lp - 1)]
        path += [(sk[k + 1], comp(sk[k], Pj[j])) for j in range(Lp - 1)]
    assert len(path) == L * (L - 2) // 2
    def act(pi, C):
        p1, p2 = pi; out = [None] * L
        for l in range(Lp): out[p1[l]] = C[l]
        for l in range(Lp): out[Lp + p2[l]] = C[Lp + l]
        return tuple(out)
    rho = lambda R: [l for l in range(L) if R[l] == 0]
    kap = lambda C: [l for l in range(L) if C[l] == 0]
    Rw = [rand_row() for _ in range(nwords)]; Cw = [rand_col() for _ in range(nwords)]
    pi = next(q for q in path if len(set(rho(Rw[0])) & set(kap(act(q, Cw[0])))) == d)
    Cp = [act(pi, C) for C in Cw]
    xs = [np.array([rng.randrange(27) for _ in range(D)]) for _ in range(nwords)]
    ys = [np.array([rng.randrange(27) for _ in range(D)]) for _ in range(nwords)]
    def exps(levels):
        out = [()]
        for j in range(1, d + 1):
            for supp in itertools.combinations(levels, j):
                for vals in itertools.product((1, 2), repeat=j):
                    out.append(tuple(zip(supp, vals)))
        return out
    # left strings: letter index per level (x letters at kind I, y letters at kind II)
    U, Ua = [], []
    for k, R in enumerate(Rw):
        for e in exps(rho(R)):
            ed = dict(e)
            U.append([Xl.index(xlet[(R[l], ed.get(l, 0))]) if l < Lp else Yl.index(ylet[(R[l], ed.get(l, 0))]) for l in range(L)] + [k, hash(e)])
            Ua.append(gsum(MUL[xs[k], a_of(e)]))
    V, Vb, Vc = [], [], []
    for k, C in enumerate(Cp):
        for f in exps(kap(C)):
            fd = dict(f)
            V.append([Yl.index(ylet[(C[l], fd.get(l, 0))]) if l < Lp else Xl.index(xlet[(C[l], fd.get(l, 0))]) for l in range(L)] + [k])
            af = a_of(f)
            Vb.append(gsum(MUL[ys[k][dual], af]))      # <y, a(f)>^ = sum_T y_{2-T} a(f)_T
            Vc.append(gsum(MUL[ys[k], af]))            # control: <y, a(f)>
    U = np.array(U); V = np.array(V); Ua = np.array(Ua); Vb = np.array(Vb); Vc = np.array(Vc)
    res = dict(d=d, L=L, words=nwords, left=len(U), right=len(V), targets=0, bad_support=0, bad_value=0, ctrl_ok=0)
    for k in range(nwords):
        for k2 in range(nwords):
            S = sorted(set(rho(Rw[k])) & set(kap(Cp[k2])))
            if len(S) != d: continue
            res['targets'] += 1
            R, C = Rw[k], Cp[k2]
            G = np.ones((len(U), len(V)), dtype=np.int64)
            for l in range(L):
                z = Zl.index(zof[(R[l], C[l])] if l < Lp else zof[(C[l], R[l])])
                if l < Lp: G = (G * G1[U[:, l][:, None], V[:, l][None, :], z]) % 3
                else:      G = (G * G1[V[:, l][None, :], U[:, l][:, None], z]) % 3
            # (i) support = true pairs: same words, e on S, f = 2 - e on S, f = 0 off S
            nz = np.argwhere(G)
            for iu, iv in nz:
                u, v = U[iu], V[iv]
                ok = u[L] == k and v[L] == k2
                if ok:
                    for l in range(L):
                        if l in S:
                            eu = (Xl if l < Lp else Yl)[u[l]]; fv = (Yl if l < Lp else Xl)[v[l]]
                            ex = {'x0': 0, 'p0': 1, 'o0': 2, 'y0': 0, 'q0': 1, 'r0': 2}
                            ok &= ex.get(eu, -9) + ex.get(fv, -9) == 2
                        else:
                            eu = (Xl if l < Lp else Yl)[u[l]]; fv = (Yl if l < Lp else Xl)[v[l]]
                            ok &= eu in ('x0', 'x1', 'y0', 'y1', 'y2') and fv in ('x0', 'x1', 'y0', 'y1', 'y2')
                    ok &= G[iu, iv] == 1
                res['bad_support'] += int(not ok)
            if len(nz) != 3 ** d: res['bad_support'] += 1
            # (ii) value
            def contract(Bv):
                W = (G @ COORD[Bv]) % 3                       # sum_v G[u,v] B[v], coordinates
                Wn = W[:, 0] + 3 * W[:, 1] + 9 * W[:, 2]
                return gsum(MUL[Ua, Wn])
            cS = 1
            for a_, b_ in itertools.combinations(S, 2):
                df = SUB[nodes[a_], nodes[b_]]; cS = MUL[cS, MUL[df, df]]
            want = MUL[cS, gsum(MUL[xs[k], ys[k2]])]
            res['bad_value'] += int(contract(Vb) != want)
            res['ctrl_ok'] += int(contract(Vc) == want)
    return res

if __name__ == '__main__':
    for d, nw, seeds in ((2, 6, range(4)), (3, 3, range(3)), (4, 2, range(1))):
        for sd in seeds:
            r = run(d, nw, 100 * d + sd)
            print('d=%(d)d L=%(L)d words=%(words)d left strings=%(left)d right strings=%(right)d: targets with |S|=d %(targets)d, '
                  'support not the true pairs %(bad_support)d, wrong values %(bad_value)d, control (wrong pairing) correct %(ctrl_ok)d' % r, flush=True)
