r"""The algorithm of Section 6 (Theorem 1.2) for a base tensor given by a decomposition with a grading, run on
T6 / T6' (characteristic 3) and on T4 (characteristic 2).

The properties checked for each base tensor (Section 7, "What a better tensor needs"):
(P1) the product of the terms is T + E with every monomial of T at coefficient 1; the grading h is 0 exactly on the
     monomials of T, >= 1 on those of E, and constant on the row blocks and on the column blocks;
(P2) every monomial of T whose output lies in cell (i, j) has its left letter in row block i and its right letter in
     column block j, and the outputs lie in distinct cells;
(P3) the high slice is sum_{e<q} alpha_e beta_{q-1-e} (q = 3 for T6, 2 for T4) and every other slice is the product
     of the outer letters of its cell.
Every field operation of the algorithm is counted as executed (GF(p^k) arithmetic on arrays; each elementwise
call counts its number of field elements); the counts are compared with the closed formula of Table 5 (Table 2 for
T4), evaluated for the configuration's parameters. Every entry of XY is checked against the textbook product, and
every output of every run at |S| = d is checked as well. Controls: no division by c(S); a wrong right pairing.
Separation check (Lemma 6.2): Mult by the n-term decomposition (T + E) against Mult by the decomposition of T
into its monomials (true terms only), at every output whose row and column words are of the band types; and the
number of outputs outside the band types where the two differ (the error terms do occur there).

Usage: python3 t6_impl.py CONFIG [CONFIG ...]   (configs at the bottom; sep:CONFIG runs the separation check,
ctrl:CONFIG the wrong-pairing control). Output to stdout.
"""
import sys, json, time, itertools
import numpy as np
from math import comb

# ----------------------------------------------------------------------------------------------- finite fields
class GF:
    def __init__(self, p, k, f):
        """GF(p^k) = F_p[x]/(x^k + f[k-1] x^{k-1} + ... + f[0]); elements are int8 arrays with the k coordinates
        on axis 0."""
        self.p, self.k, self.f, self.q = p, k, f, p ** k
        self.ops = {'mul': 0, 'add': 0, 'inv': 0}
        self.elems = [np.array([(i // p ** j) % p for j in range(k)], dtype=np.int8) for i in range(self.q)]
        self.mul_on = True

    def zeros(self, shape):
        return np.zeros((self.k,) + tuple(shape), dtype=np.int8)

    def add(self, a, b):
        r = ((a.astype(np.int16) + b) % self.p).astype(np.int8)
        self.ops['add'] += r[0].size
        return r

    def sub(self, a, b):
        r = ((a.astype(np.int16) - b) % self.p).astype(np.int8)
        self.ops['add'] += r[0].size
        return r

    def neg(self, a):
        r = ((-a.astype(np.int16)) % self.p).astype(np.int8)
        self.ops['add'] += r[0].size
        return r

    def mul(self, a, b, count=True):
        k, p, f = self.k, self.p, self.f
        c = [None] * (2 * k - 1)
        for i in range(k):
            ai = a[i].astype(np.int16)
            for j in range(k):
                t = ai * b[j]
                c[i + j] = t if c[i + j] is None else c[i + j] + t
        for t in range(2 * k - 2, k - 1, -1):
            for i in range(k):
                if f[i]:
                    c[t - k + i] = c[t - k + i] - f[i] * c[t]
        r = np.stack([c[i] % p for i in range(k)]).astype(np.int8)
        if count:
            self.ops['mul'] += r[0].size
        return r

    def inv(self, a):
        for e in self.elems[1:]:
            if np.array_equal(self.mul(a, e, count=False), self.elems[1]):
                self.ops['inv'] += 1
                return e.copy()
        raise ZeroDivisionError

    def is_zero(self, a):
        return not a.any()

    def rand(self, shape, rng):
        return rng.integers(0, self.p, size=(self.k,) + tuple(shape)).astype(np.int8)

    def lin(self, M, xs):
        """F_p-linear combination sum_s M[s] xs[s] (M entries in F_p, only +-1 occur for p <= 3), counted:
        starts from a term with coefficient 1 (free copy), then one add/sub per further nonzero term; a negation
        if no coefficient is 1."""
        terms = [(s, int(c) % self.p) for s, c in enumerate(M) if int(c) % self.p]
        assert all(c in (1, self.p - 1) for _, c in terms), 'only +-1 coefficients supported'
        terms.sort(key=lambda t: t[1] != 1)
        if not terms:
            return self.zeros(xs[0].shape[1:])
        s0, c0 = terms[0]
        acc = xs[s0].copy() if c0 == 1 else self.neg(xs[s0])
        for s, c in terms[1:]:
            acc = self.add(acc, xs[s]) if c == 1 else self.sub(acc, xs[s])
        return acc


FIELDS = {
    'GF4': (2, 2, [1, 1]),        # x^2 + x + 1
    'GF8': (2, 3, [1, 1, 0]),     # x^3 + x + 1
    'GF16': (2, 4, [1, 1, 0, 0]),  # x^4 + x + 1
    'GF9': (3, 2, [1, 0]),        # x^2 + 1
    'GF27': (3, 3, [1, 2, 0]),    # x^3 + 2x + 1 = x^3 - x + 1
}

# ----------------------------------------------------------------------------------------------- base tensors
def T6_kind():
    """T6 with its six F_3 terms and grading (Lemma 6.1; the same as ../data/T6_decomposition.json, compared by indep6_decomposition.py)."""
    Al = ['x0', 'x1', 'o0', 'p0']; Bl = ['y0', 'y1', 'y2', 'q0', 'r0']; Cl = ['w00', 'z01', 'z02', 'z10', 'z11', 'z12']
    terms = [[{"o0": 1, "p0": 2, "x0": 2, "x1": 1}, {"y0": 1}, {"w00": 1, "z01": 1, "z02": 1, "z10": 2}],
             [{"p0": 1, "x0": 2, "x1": 2}, {"q0": 1, "y0": 1, "y1": 1, "y2": 2}, {"w00": 1, "z01": 1, "z02": 2, "z10": 1, "z11": 1}],
             [{"x0": 1, "x1": 2}, {"r0": 1, "y0": 2, "y1": 1}, {"w00": 1, "z02": 2, "z10": 1, "z11": 2, "z12": 2}],
             [{"x0": 1, "x1": 1}, {"q0": 1, "r0": 1, "y0": 2, "y1": 1}, {"w00": 1, "z02": 2, "z10": 1}],
             [{"p0": 1, "x0": 1, "x1": 2}, {"y1": 1, "y2": 2}, {"w00": 2, "z01": 2, "z02": 1, "z10": 2, "z11": 2, "z12": 1}],
             [{"x0": 1}, {"r0": 1, "y0": 2, "y2": 2}, {"w00": 2, "z10": 2, "z12": 1}]]
    T = [('o0', 'y0', 'w00'), ('p0', 'q0', 'w00'), ('x0', 'r0', 'w00'), ('x0', 'y1', 'z01'), ('x0', 'y2', 'z02'),
         ('x1', 'y0', 'z10'), ('x1', 'y1', 'z11'), ('x1', 'y2', 'z12')]
    h = {'o0': -3, 'p0': -3, 'x0': -3, 'x1': -5, 'q0': 4, 'r0': 4, 'y0': 4, 'y1': -2, 'y2': -1,
         'w00': -1, 'z01': 5, 'z02': 4, 'z10': 1, 'z11': 7, 'z12': 6}
    rowletter = {(0, 0): 'x0', (0, 1): 'p0', (0, 2): 'o0', (1, 0): 'x1'}
    colletter = {(0, 0): 'y0', (0, 1): 'q0', (0, 2): 'r0', (1, 0): 'y1', (2, 0): 'y2'}
    cell = {(0, 0): 'w00', (0, 1): 'z01', (0, 2): 'z02', (1, 0): 'z10', (1, 1): 'z11', (1, 2): 'z12'}
    return mk_kind('T6', 3, Al, Bl, Cl, terms, T, h, rowletter, colletter, cell, (0, 0), 3)


def T4_kind():
    """T4 with the decomposition of Lemma 2.1 (over Z, read in characteristic 2) and the grading of
    Remark 2.2 (h = 0 on T4, 1 on E3)."""
    Al = ['x0', 'x1', 'p0']; Bl = ['y0', 'y1', 'q0']; Cl = ['z0', 'z1', 'z2', 'w0']
    terms = [[{'x0': 1, 'x1': -1}, {'y0': 1}, {'z0': 1, 'w0': -1}],
             [{'x0': 1, 'p0': -1}, {'y1': 1}, {'z1': 1, 'w0': -1}],
             [{'x1': 1}, {'y0': 1, 'q0': -1}, {'z2': 1, 'w0': -1}],
             [{'x0': 1}, {'y0': 1, 'y1': 1}, {'w0': 1}]]
    T = [('x0', 'y0', 'z0'), ('x0', 'y1', 'z1'), ('x1', 'y0', 'z2'), ('x1', 'q0', 'w0'), ('p0', 'y1', 'w0')]
    h = {'x0': 0, 'x1': 1, 'p0': 1, 'y0': 0, 'y1': 1, 'q0': 1, 'z0': 0, 'z1': -1, 'z2': -1, 'w0': -2}
    rowletter = {(0, 0): 'x0', (1, 0): 'x1', (1, 1): 'p0'}
    colletter = {(0, 0): 'y0', (1, 0): 'y1', (1, 1): 'q0'}
    cell = {(0, 0): 'z0', (0, 1): 'z1', (1, 0): 'z2', (1, 1): 'w0'}
    return mk_kind('T4', 2, Al, Bl, Cl, terms, T, h, rowletter, colletter, cell, (1, 1), 2)


def mk_kind(name, p, Al, Bl, Cl, terms, T, h, rowletter, colletter, cell, high, q):
    n = len(terms)
    Phi = np.array([[t[0].get(s, 0) % p for s in Al] for t in terms])
    Psi = np.array([[t[1].get(s, 0) % p for s in Bl] for t in terms])
    X = np.array([[t[2].get(z, 0) % p for z in Cl] for t in terms])
    # true-terms decomposition: one rank-one term per monomial of T
    PhiT = np.array([[int(s == m[0]) for s in Al] for m in T])
    PsiT = np.array([[int(s == m[1]) for s in Bl] for m in T])
    XT = np.array([[int(z == m[2]) for z in Cl] for m in T])
    return dict(name=name, p=p, Al=Al, Bl=Bl, Cl=Cl, Phi=Phi, Psi=Psi, X=X, PhiT=PhiT, PsiT=PsiT, XT=XT, T=T, h=h,
                rowletter=rowletter, colletter=colletter, cell=cell, high=high, q=q, n=n, terms=terms)


def transpose_kind(K):
    cell = {(j, i): z for (i, j), z in K['cell'].items()}
    return dict(K, name=K['name'] + "'", Al=K['Bl'], Bl=K['Al'], Phi=K['Psi'], Psi=K['Phi'], PhiT=K['PsiT'],
                PsiT=K['PhiT'], T=[(b, a, z) for a, b, z in K['T']], rowletter=K['colletter'],
                colletter=K['rowletter'], cell=cell, high=(K['high'][1], K['high'][0]),
                terms=[[t[1], t[0], t[2]] for t in K['terms']])


def check_properties(K):
    """(P1) product of the terms = T + E, weight 0 exactly on T, >= 1 on E, h block-constant; (P2), (P3)."""
    p = K['p']; P = {}
    for a, b, c in K['terms']:
        for (x, cx), (y, cy), (z, cz) in itertools.product(a.items(), b.items(), c.items()):
            P[(x, y, z)] = (P.get((x, y, z), 0) + cx * cy * cz) % p
    P = {m: c for m, c in P.items() if c}
    h = K['h']; w = lambda m: h[m[0]] + h[m[1]] + h[m[2]]
    T = set(K['T'])
    ok1 = all(P.get(m) == 1 for m in T) and all((w(m) == 0) == (m in T) for m in P) and all(w(m) >= 0 for m in P)
    rb = {s: i for (i, e), s in K['rowletter'].items()}; cb = {s: j for (j, e), s in K['colletter'].items()}
    blockconst = all(len({h[s] for s in rb if rb[s] == i}) == 1 for i in set(rb.values())) and \
                 all(len({h[s] for s in cb if cb[s] == j}) == 1 for j in set(cb.values()))
    # (P2) every monomial of T with output of cell (i,j) has A letter in block i, B letter in block j
    cellof = {z: ij for ij, z in K['cell'].items()}
    ok2 = all(rb[a] == cellof[z][0] and cb[b] == cellof[z][1] for a, b, z in K['T']) and len(K['cell']) == K['n']
    # (P3) high slice sum_e alpha_e beta_{q-1-e}; low slices outer letters only
    q = K['q']; i0, j0 = K['high']; ok3 = True
    for (i, j), z in K['cell'].items():
        sl = sorted((a, b) for a, b, zz in K['T'] if zz == z)
        if (i, j) == (i0, j0):
            want = sorted((K['rowletter'][(i, e)], K['colletter'][(j, q - 1 - e)]) for e in range(q))
        else:
            want = [(K['rowletter'][(i, 0)], K['colletter'][(j, 0)])]
        ok3 &= sl == want
    return dict(P1_tight=ok1, P1_blockconst=blockconst, P2=ok2, P3=ok3, nE=len(P) - len(T))


# ----------------------------------------------------------------------------------------------- the algebra V
class Alg:
    """V = E[xi_1..xi_d]/(xi_i^q), dense coordinates over all T in {0..q-1}^d."""
    def __init__(self, F, d, q):
        self.F, self.d, self.q = F, d, q
        self.mons = list(itertools.product(range(q), repeat=d))
        self.idx = {T: i for i, T in enumerate(self.mons)}
        self.deg = np.array([sum(T) for T in self.mons])
        self.bydeg = [np.array([i for i, T in enumerate(self.mons) if sum(T) == k]) for k in range((q - 1) * d + 1)]
        self.dual = np.array([self.idx[tuple(q - 1 - t for t in T)] for T in self.mons])


# ----------------------------------------------------------------------------------------------- relabelings
def cyc_family(Lp):
    """the paper's Pi_L on Lp levels: s^k o theta_1 o ... o theta_j, theta_j = (j-1, j) (0-indexed)"""
    s = [(l + 1) % Lp for l in range(Lp)]
    def comp(a, b): return [a[b[l]] for l in range(Lp)]   # (a o b)(l) = a(b(l))
    pre = [list(range(Lp))]
    for j in range(1, Lp):
        th = list(range(Lp)); th[j - 1], th[j] = th[j], th[j - 1]
        pre.append(comp(pre[-1], th))
    assert pre[Lp - 1] == s
    sk = [list(range(Lp))]
    for k in range(1, Lp):
        sk.append(comp(sk[-1], s))
    return sk, pre


def family(kind_groups):
    """relabelings as permutations of all L levels. One group: Pi_L. Two groups of equal size: the diagonal path."""
    if len(kind_groups) == 1:
        Lp = len(kind_groups[0])
        sk, pre = cyc_family(Lp)
        return [[None] and [sk[k][x] for x in pre[j]] for k in range(Lp) for j in range(Lp - 1)]
    g1, g2 = kind_groups; Lp = len(g1); assert len(g2) == Lp
    sk, pre = cyc_family(Lp)
    def comp(a, b): return [a[b[l]] for l in range(Lp)]
    fam = []
    for k in range(Lp):
        for j in range(Lp - 1):
            fam.append((comp(sk[k], pre[j]), sk[k]))
        for j in range(Lp - 1):
            fam.append((sk[(k + 1) % Lp], comp(sk[k], pre[j])))
    # consecutive members differ by one transposition on the right of one factor (checked)
    def ntrans(a, b): return sum(1 for l in range(Lp) if a[l] != b[l])
    for i in range(len(fam)):
        x, y = fam[i], fam[(i + 1) % len(fam)]
        assert ntrans(x[0], y[0]) + ntrans(x[1], y[1]) == 2, 'not one transposition'
    return [[g1[a] - g1[0] for a in x] + [Lp + b for b in y] for x, y in fam]


# ----------------------------------------------------------------------------------------------- the algorithm
def words(groups):
    """all words with prescribed block counts per group: groups = [(levels, {block: count})]"""
    parts = []
    for lev, cnt in groups:
        base = sorted(sum([[b] * c for b, c in cnt.items()], []))
        assert len(base) == len(lev)
        parts.append(sorted(set(itertools.permutations(base))))
    out = []
    for combo in itertools.product(*parts):
        w = {}
        for (lev, _), part in zip(groups, combo):
            for l, b in zip(lev, part):
                w[l] = b
        out.append(tuple(w[l] for l in range(len(w))))
    return out


def exps(levels, d, q):
    """all exponent functions on the given levels with |supp| <= d, values 1..q-1 on the support"""
    out = [()]
    for j in range(1, d + 1):
        for supp in itertools.combinations(levels, j):
            for vals in itertools.product(range(1, q), repeat=j):
                out.append(tuple(zip(supp, vals)))
    return out


def run_config(cfg, out):
    t0 = time.time()
    rng = np.random.default_rng(cfg.get('seed', 1))
    F = GF(*FIELDS[cfg['field']])
    kinds = cfg['kinds']; L = len(kinds); p = F.p
    q = kinds[0]['q']; d = cfg['d']; n = kinds[0]['n']
    V = Alg(F, d, q)
    Rw = words(cfg['rows']); Cw = words(cfg['cols'])
    N0r, N0c = len(Rw), len(Cw)
    Br, Bc = cfg['bands']
    Nr, Nc = Br * N0r, Bc * N0c
    D = V.q ** d if False else q ** d
    fam = family(cfg['groups'])
    Kf = len(fam)
    hi = lambda l: kinds[l]['high']
    rho = lambda R: [l for l in range(L) if R[l] == hi(l)[0]]
    kap = lambda C: [l for l in range(L) if C[l] == hi(l)[1]]
    print('config %s: kinds %s, L=%d, d=%d, D=%d, field %s, row words %d, column words %d, bands %dx%d, K=%d'
          % (cfg['name'], '-'.join(k['name'] for k in kinds), L, d, D,
             cfg['field'], N0r, N0c, Br, Bc, Kf), file=out, flush=True)
    for K in {k['name']: k for k in kinds}.values():
        print('  properties of %s: %s' % (K['name'], check_properties(K)), file=out)
    # inputs
    Xm = F.rand((Nr, D), rng); Ym = F.rand((D, Nc), rng)
    ops0 = dict(F.ops)
    # ---- step 0: nodes, powers, forms
    assert L <= F.q or d == 1
    nodes = [F.elems[l % F.q] for l in range(L)]
    powr = []
    for l in range(L):
        pw = [F.elems[1].copy(), nodes[l].copy()][:d]
        for i in range(2, d):
            pw.append(F.mul(pw[-1], nodes[l]))
        powr.append(pw)
    # ---- step 0: table a(e) for all e on [L] with 1 <= |supp| <= d, a(e) = a(e - delta_max) * phi_max
    one = F.zeros((len(V.mons),)); one[:, V.idx[(0,) * d]] = F.elems[1]
    table = {(): one}
    shift = {}
    for i in range(d):
        src = [V.idx[T] for T in V.mons if T[i] <= q - 2]
        dst = [V.idx[T[:i] + (T[i] + 1,) + T[i + 1:]] for T in V.mons if T[i] <= q - 2]
        shift[i] = (np.array(src), np.array(dst))
    for e in sorted(exps(range(L), d, q)[1:], key=lambda e: (sum(v for _, v in e), e)):
        m = e[-1][0]
        prev = tuple((l, v - (l == m)) for l, v in e if not (l == m and v == 1))
        a = table[prev]; k = sum(v for _, v in prev)
        res = F.zeros((len(V.mons),)); hit = np.zeros(len(V.mons), bool)
        for i in range(d):
            src, dst = shift[i]
            sel = V.deg[src] == k            # only the monomials of degree k (all positions counted)
            s_, t_ = src[sel], dst[sel]
            if len(s_) == 0:
                continue
            prod = F.mul(a[:, s_], powr[m][i][:, None])
            new = ~hit[t_]
            res[:, t_[new]] = prod[:, new]
            if (~new).any():
                res[:, t_[~new]] = F.add(res[:, t_[~new]], prod[:, ~new])
            hit[t_] = True
        table[e] = res
    # ---- step 0: divisors c(S)^{-1}
    cinv = {}
    if d == 1:
        for l in range(L):
            cinv[(l,)] = F.elems[1].copy()
    else:
        for S in itertools.combinations(range(L), d):
            det = None
            for l, l2 in itertools.combinations(S, 2):
                diff = F.sub(nodes[l2], nodes[l])
                det = diff if det is None else F.mul(det, diff)
            c = det if q == 2 else F.mul(det, det)
            cinv[S] = F.inv(c)
    ops_setup = {k: F.ops[k] - ops0[k] for k in F.ops}
    # ---- covering table (bit operations, not counted)
    rho_r = [frozenset(rho(R)) for R in Rw]
    def relabel(C, pi):
        Cn = [None] * L
        for l in range(L):
            Cn[pi[l]] = C[l]
        return tuple(Cn)
    Cpi = [[relabel(C, pi) for C in Cw] for pi in fam]
    kap_c = [[frozenset(kap(C)) for C in row] for row in Cpi]
    first = {}
    uncovered = 0
    for nu in range(N0r):
        for mu in range(N0c):
            for k, pi in enumerate(fam):
                if len(rho_r[nu] & kap_c[k][mu]) == d:
                    first[(nu, mu)] = k; break
            else:
                uncovered += 1
    print('  covering: %d of %d label pairs uncovered; relabelings used as first cover: %d of %d'
          % (uncovered, N0r * N0c, len(set(first.values())), Kf), file=out, flush=True)
    assert uncovered == 0
    # string spaces
    Ashape = [len(kinds[l]['Al']) for l in range(L)]; Bshape = [len(kinds[l]['Bl']) for l in range(L)]
    Aidx = [{s: i for i, s in enumerate(kinds[l]['Al'])} for l in range(L)]
    Bidx = [{s: i for i, s in enumerate(kinds[l]['Bl'])} for l in range(L)]
    Cidx = [{s: i for i, s in enumerate(kinds[l]['Cl'])} for l in range(L)]
    def astring(R, e):
        ed = dict(e)
        return tuple(Aidx[l][kinds[l]['rowletter'][(R[l], ed.get(l, 0))]] for l in range(L))
    def bstring(C, f):
        fd = dict(f)
        return tuple(Bidx[l][kinds[l]['colletter'][(C[l], fd.get(l, 0))]] for l in range(L))
    def ostring(R, C):
        return tuple(Cidx[l][kinds[l]['cell'][(R[l], C[l])]] for l in range(L))
    # ---- step 1 reading (counted), vectorised per degree
    def read(vecs, labels, levfun, right):
        """vecs: (k, nlab, D) the rows X_I (or columns Y_J) of one band, labels: words; returns dict string -> value"""
        jobs = []                                        # (label index, e)
        for i, W in enumerate(labels):
            for e in exps(levfun(W), d, q):
                jobs.append((i, e))
        vals = {}
        bydeg = {}
        for i, e in jobs:
            bydeg.setdefault(sum(v for _, v in e), []).append((i, e))
        for kdeg, js in bydeg.items():
            mons = V.bydeg[kdeg]
            av = np.stack([table[e][:, mons] for _, e in js], axis=1)           # (k, nj, M)
            ii = np.array([i for i, _ in js])
            xv = vecs[:, ii][:, :, V.dual[mons]] if (right and not cfg.get('control_pairing')) else vecs[:, ii][:, :, mons]
            pr = F.mul(xv, av)
            acc = pr[:, :, 0]
            for c in range(1, len(mons)):
                acc = F.add(acc, pr[:, :, c])
            for t, (i, e) in enumerate(js):
                vals[(i, e)] = acc[:, t]
        return vals
    # ---- step 2 encoding (counted)
    def ratio(M):
        c = sum(max(sum(1 for x in row if x % p) - 1, 0) for row in M)
        return c, M.shape[0] - M.shape[1]
    def encode(arr, mats):
        order = sorted(range(L), key=lambda l: -ratio(mats[l])[0] / max(ratio(mats[l])[1], 1e-9))
        for l in order:
            M = mats[l]
            xs = [np.take(arr, s, axis=l + 1) for s in range(M.shape[1])]
            ys = [F.lin(M[lam], xs) for lam in range(M.shape[0])]
            arr = np.stack(ys, axis=l + 1)
        return arr
    def decode(arr, mats):
        for l in range(L):
            M = mats[l].T
            xs = [np.take(arr, s, axis=l + 1) for s in range(M.shape[1])]
            ys = [F.lin(M[z], xs) for z in range(M.shape[0])]
            arr = np.stack(ys, axis=l + 1)
        return arr
    PhiL = [kinds[l]['Phi'] for l in range(L)]; PsiL = [kinds[l]['Psi'] for l in range(L)]
    XL = [kinds[l]['X'] for l in range(L)]
    ops_steps = {}
    def tick(name, before):
        ops_steps[name] = {k: ops_steps.get(name, {}).get(k, 0) + F.ops[k] - before[k] for k in F.ops}
    # rows: read and encode once per row band
    encA = []
    for b in range(Br):
        o = dict(F.ops)
        vals = read(Xm[:, b * N0r:(b + 1) * N0r, :], Rw, rho, False)
        tick('1 reading', o)
        A = F.zeros(Ashape)
        for (i, e), v in vals.items():
            A[(slice(None),) + astring(Rw[i], e)] = v
        o = dict(F.ops)
        encA.append(encode(A, PhiL))
        tick('2 encoding', o)
    result = F.zeros((Nr, Nc)); done = np.zeros((Nr, Nc), bool)
    nodiv_wrong = 0; nodiv_tot = 0; allrun_checked = 0; allrun_wrong = 0
    XY = None
    # textbook product, uncounted
    F.mul_on = False
    o = dict(F.ops)
    XY = F.zeros((Nr, Nc))
    for t in range(D):
        XY = ((XY.astype(np.int16) + F.mul(Xm[:, :, t][:, :, None], Ym[:, t, :][:, None, :], count=False)) % p).astype(np.int8)
    F.ops = o
    for g in range(Bc):
        Yb = np.transpose(Ym[:, :, g * N0c:(g + 1) * N0c], (0, 2, 1))          # (k, N0c, D)
        for kpi, pi in enumerate(fam):
            o = dict(F.ops)
            vals = read(Yb, Cpi[kpi], kap, True)
            tick('1 reading', o)
            Bt = F.zeros(Bshape)
            for (i, f), v in vals.items():
                Bt[(slice(None),) + bstring(Cpi[kpi][i], f)] = v
            o = dict(F.ops)
            encB = encode(Bt, PsiL)
            tick('2 encoding', o)
            for b in range(Br):
                o = dict(F.ops)
                leaf = F.mul(encA[b], encB)
                tick('3 leaf products', o)
                o = dict(F.ops)
                Mlt = decode(leaf, XL)
                tick('4 decoding', o)
                # read-out (counted) for the pairs whose first cover is this relabeling
                o = dict(F.ops)
                for nu in range(N0r):
                    for mu in range(N0c):
                        S = tuple(sorted(rho_r[nu] & kap_c[kpi][mu]))
                        if len(S) != d:
                            continue
                        val = Mlt[(slice(None),) + ostring(Rw[nu], Cpi[kpi][mu])]
                        I, J = b * N0r + nu, g * N0c + mu
                        if first[(nu, mu)] == kpi:
                            result[:, I, J] = F.mul(val, cinv[S])
                            done[I, J] = True
                            nodiv_tot += 1
                            nodiv_wrong += int(not np.array_equal(val, XY[:, I, J]))
                        else:                         # every other output at |S| = d, checked, not counted
                            F.mul_on = False
                            oo = dict(F.ops)
                            chk = F.mul(val, cinv[S], count=False)
                            F.ops = oo
                            allrun_checked += 1
                            allrun_wrong += int(not np.array_equal(chk, XY[:, I, J]))
                tick('5 read-out', o)
    wrong = int((result != XY).any(axis=0).sum())
    print('  all %d entries computed: %s; wrong entries: %d' % (Nr * Nc, bool(done.all()), wrong), file=out)
    print('  other outputs at |S| = d in all runs checked: %d, wrong: %d' % (allrun_checked, allrun_wrong), file=out)
    print('  control (no division by c(S)): wrong %d of %d' % (nodiv_wrong, nodiv_tot), file=out)
    ops_steps['0 set-up'] = ops_setup
    # ---- formula (Table 5, general parameters)
    form = formula(cfg, kinds, d, q, n, L, N0r, N0c, Br, Bc, Kf, rho(Rw[0]), kap(Cw[0]), V)
    allok = True
    for name in sorted(form):
        got = ops_steps.get(name, {'mul': 0, 'add': 0, 'inv': 0})
        ok = (got['mul'], got['add'], got['inv']) == form[name]
        allok &= ok
        print('  %-16s counted (mul, add, inv) = %s  formula = %s  %s' % (name, (got['mul'], got['add'], got['inv']), form[name], 'EQUAL' if ok else 'DIFFERENT'), file=out)
    tot = sum(sum(v) for v in form.values())
    print('  total %d operations, %.2f per entry, = %.4f L^6 per entry; counts equal formula: %s; %.1f s'
          % (tot, tot / (Nr * Nc), tot / (Nr * Nc) / L ** 6, allok, time.time() - t0), file=out, flush=True)
    return dict(name=cfg['name'], wrong=wrong, entries=Nr * Nc, counts_equal=allok, allrun_wrong=allrun_wrong)


def P_prod(V, k):
    """products when a degree-k element (all monomials of degree k) is multiplied by a linear form"""
    return sum(1 for T in V.mons if sum(T) == k for i in range(V.d) if T[i] <= V.q - 2)


def formula(cfg, kinds, d, q, n, L, N0r, N0c, Br, Bc, Kf, rho0, kap0, V):
    Mk = [len(V.bydeg[k]) for k in range((q - 1) * d + 1)]
    f = {}
    f['0 set-up'] = None
    pw = L * max(d - 2, 0)
    tm = ta = 0
    for j in range(1, d + 1):
        for vals in itertools.product(range(1, q), repeat=j):
            k = sum(vals); cnt = comb(L, j)
            pp = P_prod(V, k - 1)
            tm += cnt * pp; ta += cnt * (pp - Mk[k])
    if d >= 2:
        dm, da, di = comb(L, d) * (comb(d, 2) - 1 + (q == 3)), comb(L, d) * comb(d, 2), comb(L, d)
    else:
        dm = da = di = 0
    f['0 set-up'] = (pw + tm + dm, ta + da, di)
    def Rm(nlev, minus):
        s = 0
        for j in range(d + 1):
            for vals in itertools.product(range(1, q), repeat=j):
                s += comb(nlev, j) * (Mk[sum(vals)] - minus)
        return s
    f['1 reading'] = (Br * N0r * Rm(len(rho0), 0) + Kf * Bc * N0c * Rm(len(kap0), 0),
                      Br * N0r * Rm(len(rho0), 1) + Kf * Bc * N0c * Rm(len(kap0), 1), 0)
    def enc(mats):
        p = kinds[0]['p']
        cs = [(sum(max(sum(1 for x in row if x % p) - 1, 0) for row in M), M.shape[1]) for M in mats]
        order = sorted(range(L), key=lambda l: -cs[l][0] / max(n - cs[l][1], 1e-9))
        tot = 0
        for pos, l in enumerate(order):
            fib = n ** pos
            for l2 in order[pos + 1:]:
                fib *= cs[l2][1]
            tot += cs[l][0] * fib
        return tot
    f['2 encoding'] = (0, Br * enc([kinds[l]['Phi'] for l in range(L)]) + Kf * Bc * enc([kinds[l]['Psi'] for l in range(L)]), 0)
    f['3 leaf products'] = (Kf * Br * Bc * n ** L, 0, 0)
    dec = sum(sum(max(sum(1 for x in col if x % kinds[l]['p']) - 1, 0) for col in kinds[l]['X'].T) * n ** (L - 1) for l in range(L))
    f['4 decoding'] = (0, Kf * Br * Bc * dec, 0)
    f['5 read-out'] = (Br * N0r * Bc * N0c, 0, 0)
    return f


# ----------------------------------------------------------------------------------------------- separation check
def separation(cfg, out):
    """Lemma 6.2: Mult by the n-term decomposition (T + E) equals Mult by the monomial decomposition of T at every
    output whose words are of the band types; random A, B on ALL strings of the band types (every exponent
    function, not only |supp| <= d). Field: F_p is enough (the statement is over every ring)."""
    t0 = time.time()
    rng = np.random.default_rng(7)
    kinds = cfg['kinds']; L = len(kinds); p = kinds[0]['p']
    F = GF(p, 1, [0] if p == 2 else [0])
    F.ops = {'mul': 0, 'add': 0, 'inv': 0}
    Rw = set(words(cfg['rows'])); Cw = set(words(cfg['cols']))
    Ashape = [len(k['Al']) for k in kinds]; Bshape = [len(k['Bl']) for k in kinds]
    A = F.zeros(Ashape); B = F.zeros(Bshape)
    rb = [{s: i for (i, e), s in k['rowletter'].items()} for k in kinds]
    cb = [{s: j for (j, e), s in k['colletter'].items()} for k in kinds]
    for u in itertools.product(*[range(s) for s in Ashape]):
        if tuple(rb[l][kinds[l]['Al'][u[l]]] for l in range(L)) in Rw:
            A[(slice(None),) + u] = rng.integers(0, p)
    for v in itertools.product(*[range(s) for s in Bshape]):
        if tuple(cb[l][kinds[l]['Bl'][v[l]]] for l in range(L)) in Cw:
            B[(slice(None),) + v] = rng.integers(0, p)
    def mult(PhiK, PsiK, XK):
        a, b = A, B
        for l in range(L):
            M = kinds[l][PhiK]; xs = [np.take(a, s, axis=l + 1) for s in range(M.shape[1])]
            a = np.stack([F.lin(M[r], xs) for r in range(M.shape[0])], axis=l + 1)
            M = kinds[l][PsiK]; xs = [np.take(b, s, axis=l + 1) for s in range(M.shape[1])]
            b = np.stack([F.lin(M[r], xs) for r in range(M.shape[0])], axis=l + 1)
        c = F.mul(a, b)
        for l in range(L):
            M = kinds[l][XK].T; xs = [np.take(c, s, axis=l + 1) for s in range(M.shape[1])]
            c = np.stack([F.lin(M[r], xs) for r in range(M.shape[0])], axis=l + 1)
        return c[0]
    M1 = mult('Phi', 'Psi', 'X'); M2 = mult('PhiT', 'PsiT', 'XT')
    cellof = [{z: ij for ij, z in k['cell'].items()} for k in kinds]
    inband = np.zeros(M1.shape, bool)
    for w in itertools.product(*[range(len(k['Cl'])) for k in kinds]):
        R = tuple(cellof[l][kinds[l]['Cl'][w[l]]][0] for l in range(L)); C = tuple(cellof[l][kinds[l]['Cl'][w[l]]][1] for l in range(L))
        inband[w] = R in Rw and C in Cw
    diff = M1 != M2
    print('separation %s: outputs of the band types %d, differing %d; outputs outside %d, differing %d (control); '
          'nonzero true values in band %d; %.1f s'
          % (cfg['name'], inband.sum(), (diff & inband).sum(), (~inband).sum(), (diff & ~inband).sum(),
             ((M2 != 0) & inband).sum(), time.time() - t0), file=out, flush=True)
    return int((diff & inband).sum())


# ----------------------------------------------------------------------------------------------- configurations
def configs():
    T6 = T6_kind(); T6t = transpose_kind(T6); T4 = T4_kind()
    C = {}
    # theorem parameters, d = 1: L' = 3, L = 6, A = 2
    C['thm_d1'] = dict(name='thm_d1 (theorem parameters, d=1)', kinds=[T6] * 3 + [T6t] * 3, d=1, field='GF9',
                       rows=[([0, 1, 2], {0: 2, 1: 1}), ([3, 4, 5], {0: 1, 1: 1, 2: 1})],
                       cols=[([0, 1, 2], {0: 1, 1: 1, 2: 1}), ([3, 4, 5], {0: 2, 1: 1})],
                       groups=[[0, 1, 2], [3, 4, 5]], bands=(2, 2))
    # symmetric layout, d = 2, L' = 4 (smaller than the theorem's L' = 6; block counts chosen so that the mean is 2)
    C['sym_d2'] = dict(name='sym_d2 (T6/T6\', L=8, d=2)', kinds=[T6] * 4 + [T6t] * 4, d=2, field='GF9',
                       rows=[([0, 1, 2, 3], {0: 2, 1: 2}), ([4, 5, 6, 7], {0: 2, 1: 1, 2: 1})],
                       cols=[([0, 1, 2, 3], {0: 2, 1: 1, 2: 1}), ([4, 5, 6, 7], {0: 2, 1: 2})],
                       groups=[[0, 1, 2, 3], [4, 5, 6, 7]], bands=(2, 2))
    # T6 alone (rectangular tiles), L = 7, d = 2
    C['rect_d2'] = dict(name='rect_d2 (T6 only, L=7, d=2)', kinds=[T6] * 7, d=2, field='GF9',
                        rows=[(list(range(7)), {0: 4, 1: 3})], cols=[(list(range(7)), {0: 4, 1: 2, 2: 1})],
                        groups=[list(range(7))], bands=(1, 1))
    # T6 alone, L = 9, d = 3 (all nine elements of GF(9) are nodes)
    C['rect_d3'] = dict(name='rect_d3 (T6 only, L=9, d=3)', kinds=[T6] * 9, d=3, field='GF9',
                        rows=[(list(range(9)), {0: 6, 1: 3})], cols=[(list(range(9)), {0: 5, 1: 2, 2: 2})],
                        groups=[list(range(9))], bands=(1, 1))
    # symmetric, L = 10, d = 3, GF(27)
    C['sym_d3'] = dict(name='sym_d3 (T6/T6\', L=10, d=3)', kinds=[T6] * 5 + [T6t] * 5, d=3, field='GF27',
                       rows=[([0, 1, 2, 3, 4], {0: 3, 1: 2}), ([5, 6, 7, 8, 9], {0: 3, 1: 1, 2: 1})],
                       cols=[([0, 1, 2, 3, 4], {0: 3, 1: 1, 2: 1}), ([5, 6, 7, 8, 9], {0: 3, 1: 2})],
                       groups=[[0, 1, 2, 3, 4], [5, 6, 7, 8, 9]], bands=(1, 1))
    # T4 in characteristic 2 through the same code (property-form check)
    C['t4_L8'] = dict(name='t4_L8 (T4, char 2, L=8, d=2)', kinds=[T4] * 8, d=2, field='GF8',
                      rows=[(list(range(8)), {0: 4, 1: 4})], cols=[(list(range(8)), {0: 4, 1: 4})],
                      groups=[list(range(8))], bands=(2, 2))
    C['t4_L10'] = dict(name='t4_L10 (T4, char 2, L=10, d=2)', kinds=[T4] * 10, d=2, field='GF16',
                       rows=[(list(range(10)), {0: 5, 1: 5})], cols=[(list(range(10)), {0: 5, 1: 5})],
                       groups=[list(range(10))], bands=(1, 1))
    C['t4_sym'] = dict(name='t4_sym (T4/T4\', L=8, d=2)', kinds=[T4] * 4 + [transpose_kind(T4)] * 4, d=2, field='GF8',
                       rows=[([0, 1, 2, 3], {0: 2, 1: 2}), ([4, 5, 6, 7], {0: 2, 1: 2})],
                       cols=[([0, 1, 2, 3], {0: 2, 1: 2}), ([4, 5, 6, 7], {0: 2, 1: 2})],
                       groups=[[0, 1, 2, 3], [4, 5, 6, 7]], bands=(1, 1))
    return C


if __name__ == '__main__':
    C = configs()
    out = sys.stdout
    for name in sys.argv[1:]:
        if name.startswith('sep:'):
            separation(C[name[4:]], out)
        elif name.startswith('ctrl:'):                 # control: the left pairing <y, v> on the right side
            cfg = dict(C[name[5:]], control_pairing=True, name=C[name[5:]]['name'] + ' CONTROL wrong pairing')
            try:
                r = run_config(cfg, out)
            except AssertionError as e:
                print('control run stopped:', e, file=out)
            print('RESULT', json.dumps(r), file=out, flush=True)
        else:
            r = run_config(C[name], out)
            print('RESULT', json.dumps(r), file=out, flush=True)
