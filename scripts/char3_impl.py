r"""The algorithm of Section 5 (Theorem 1.1(b),(c)): the full product N_r x D x N_c over a field of characteristic 3
from T4 and the signed decomposition of Lemma 2.1 (called P3 below), with the forms phi_l = (z^{l(i-1)})_i of
Lemma 5.2 and a deterministic covering of the inner sets by several values of z (Corollary 5.3); instrumented
operation count against the exact formula; checked against direct multiplication. Port of char2_impl.py with these
changes:
  * E = GF(3^k) (gf3k.py), elements as base-3 digit vectors; dense arrays as k digit planes (int8), the
    transforms act plane by plane (their coefficients are 0, +1, -1 in F_3);
  * P3 with its signs (exact over Z, Lemma 2.1): terms (x0 - x1) y0 (z0 - w0), (x0 - p0) y1 (z1 - w0),
    x1 (y0 - q0) (z2 - w0), x0 (y0 + y1) w0, sum = T4 - (x1y0z0 + p0y1z1 + x1q0z2);
  * rows of size a, columns of size b (a = b allowed), inner sets of size s = floor(a b / L) = 2d, D = 3^d, inner
    index T in {0,1,2}^d (t = sum T_i 3^i), V = E[x_1..x_d]/(x_i^3);
  * left value <X_r, a(P)> = sum_{|T| = |P|} X[r,T] a(P)_T; right value <Y_c, a(Q)>' = sum_{|T| = 2d-|Q|} Y[T,c]
    sigma(T) a(Q)_{2-T}, sigma(T) = (-1)^{#{i: T_i = 1}} (signed table a'(Q)_U = sigma(U) a(Q)_U, set-up);
  * the entry is read at w = (r, pi(c)) of a tile-run (z, pi) with |r cap pi(c)| = 2d and c_z(S) != 0 and divided by
    c_z(S) = a(S) (the coefficient of prod x_i^2), Proposition 5.1;
  * forms phi_l = (z^{l (i-1)})_{i=1..d}, levels l = 0..L-1, z from a list Z (zmode 'greedy': greedy hitting set
    over E^*; 'explicit': Z = (g^0, ..., g^Delta), Delta = max_S deg g_S, no search; 'single:<e>': Z = (g^e,) as a
    control).
Schedules: 'first' (each pair at its first (z, pi), z-major), and optionally 'all' (every (z, pi), z in Z).
Every class output of every executed tile-run is checked: c_z(S) != 0: value = XY; c_z(S) = 0: raw value = 0.
Controls (flag): 'nodiv' (no division; always reported), 'unsigned' (right pairing without sigma), 'char2P3' (P3 with
plus signs, the characteristic-2 form), zmode 'single:<e>' (one z: uncovered pairs).
Usage: python3 char3_impl.py L a b B inputs(F3|E) k zmode(greedy|explicit|single:<e>) sched(first|all)
       [control(none|unsigned|char2P3)] [seed]   ->  ../outputs/char3_impl_<tag>_out.txt
"""
import sys, os, time, math, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gf3k import GF3k

CNT = {'mul': 0, 'add': 0, 'neg': 0, 'inv': 0}

# P3 with signs (rows = terms 1..4).  Left letters (x0, x1, p0), right letters (y0, y1, q0), outputs (z0, z1, z2, w0).
PHI_S = np.array([[1, -1, 0], [1, 0, -1], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
PSI_S = np.array([[1, 0, 0], [0, 1, 0], [1, 0, -1], [1, 1, 0]], dtype=np.int64)
CHI_S = np.array([[1, 0, 0, -1], [0, 1, 0, -1], [0, 0, 1, -1], [0, 0, 0, 1]], dtype=np.int64)
# the characteristic-2 form (plus signs), for the control
PHI_2 = np.abs(PHI_S); PSI_2 = np.abs(PSI_S); CHI_2 = np.abs(CHI_S)


def check_P3():
    """Expand the four terms over Z and compare with T4 - E3 (signed) / T4 + E3 + 2(...) (unsigned)."""
    L_, R_, O_ = ['x0', 'x1', 'p0'], ['y0', 'y1', 'q0'], ['z0', 'z1', 'z2', 'w0']
    def expand(P, Q, C):
        t = {}
        for lam in range(4):
            for i in range(3):
                for j in range(3):
                    for o in range(4):
                        c = P[lam, i] * Q[lam, j] * C[lam, o]
                        if c:
                            key = L_[i] + R_[j] + O_[o]
                            t[key] = t.get(key, 0) + c
        return {k: v for k, v in t.items() if v}
    T4 = {'x0y0z0': 1, 'x0y1z1': 1, 'x1y0z2': 1, 'x1q0w0': 1, 'p0y1w0': 1}
    E3 = {'x1y0z0': 1, 'p0y1z1': 1, 'x1q0z2': 1}
    want = dict(T4); [want.__setitem__(k, -1) for k in E3]
    s = expand(PHI_S, PSI_S, CHI_S)
    u = expand(PHI_2, PSI_2, CHI_2)
    diff_u = {k: u.get(k, 0) - (T4.get(k, 0) + E3.get(k, 0)) for k in set(u) | set(T4) | set(E3)}
    return s == want, {k: v for k, v in diff_u.items() if v}


def popcount_table(L):
    t = np.zeros(1 << L, dtype=np.int64)
    for m in range(1, 1 << L):
        t[m] = t[m >> 1] + (m & 1)
    return t


def relabel_family(L):
    """Pi_L of Lemma 3.3, as in char2_impl.py."""
    def compose(p, q):
        return tuple(p[q[l]] for l in range(L))
    ident = tuple(range(L))
    trans = []
    for j in range(L - 1, 0, -1):
        t = list(range(L)); t[0], t[j] = t[j], t[0]
        trans.append(tuple(t))
    prod = ident
    for t in trans:
        prod = compose(prod, t)
    assert prod == tuple((l + 1) % L for l in range(L))
    fam, cur = [], ident
    for k in range(L):
        for t in trans:
            fam.append(cur)
            cur = compose(cur, t)
    assert cur == ident
    return fam


def perm_masks(perm, masks, L):
    out = np.zeros_like(masks)
    for l in range(L):
        out |= ((masks >> l) & 1) << perm[l]
    return out


def spread(masks, L, base):
    out = np.zeros(np.shape(masks), dtype=np.int64)
    for l in range(L):
        out += ((masks >> l) & 1) * (base ** (L - 1 - l))
    return out


def tri(d, i):
    """M3(d, i): number of T in {0,1,2}^d with |T| = i."""
    return sum(1 for T in itertools.product(range(3), repeat=d) if sum(T) == i)


def table_pairs(d, i):
    """(source T of degree i-1, variable j with T_j < 2, target T + e_j) for the step a(P) = a(P - top) phi_top."""
    out = []
    for T in itertools.product(range(3), repeat=d):
        if sum(T) != i - 1:
            continue
        t = sum(T[j] * 3 ** j for j in range(d))
        for j in range(d):
            if T[j] < 2:
                out.append((t, j, t + 3 ** j))
    return out


def n_minus(d, i):
    return sum(1 for T in itertools.product(range(3), repeat=d) if sum(T) == i and sum(1 for x in T if x == 1) % 2)


def formula(L, a, b, d, B, nz, nzpi, ngood_inv, ndiv=None):
    """Exact count for a schedule with nz values of z (left arrays, tables) and nzpi tile-run pairs (z, pi)."""
    N0r, N0c = math.comb(L, a), math.comb(L, b)
    f = {}
    f['forms_mul'] = nz * (((L - 2) + L * max(d - 2, 0)) if d >= 2 else 0)
    tm = ta = tn = 0
    for i in range(1, 2 * d + 1):
        npairs = len(table_pairs(d, i))
        tm += math.comb(L, i) * npairs
        ta += math.comb(L, i) * (npairs - tri(d, i))
    for i in range(0, 2 * d + 1):
        tn += math.comb(L, i) * n_minus(d, i)
    f['table_mul'] = nz * tm; f['table_add'] = nz * ta; f['sign_neg'] = nz * tn
    f['inv'] = ngood_inv
    Rl = sum(math.comb(a, i) * tri(d, i) for i in range(0, min(2 * d, a) + 1))
    Sl = sum(math.comb(a, i) for i in range(0, min(2 * d, a) + 1))
    Rr = sum(math.comb(b, i) * tri(d, i) for i in range(0, min(2 * d, b) + 1))
    Sr = sum(math.comb(b, i) for i in range(0, min(2 * d, b) + 1))
    f['readL_mul'] = nz * B * N0r * Rl; f['readL_add'] = nz * B * N0r * (Rl - Sl)
    f['readR_mul'] = nzpi * B * N0c * Rr; f['readR_add'] = nzpi * B * N0c * (Rr - Sr)
    f['enc_add'] = (nz * B + nzpi * B) * 2 * (4 ** L - 3 ** L)
    f['leaf_mul'] = nzpi * B * B * 4 ** L
    f['dec_add'] = nzpi * B * B * 3 * L * 4 ** (L - 1)
    f['div_mul'] = B * N0r * B * N0c if ndiv is None else ndiv
    f['mul'] = f['forms_mul'] + f['table_mul'] + f['readL_mul'] + f['readR_mul'] + f['leaf_mul'] + f['div_mul']
    f['add'] = f['table_add'] + f['readL_add'] + f['readR_add'] + f['enc_add'] + f['dec_add']
    f['neg'] = f['sign_neg']
    f['total'] = f['mul'] + f['add'] + f['neg'] + f['inv']
    return f


class Algo:
    def __init__(self, L, a, b, B, k, ctrl):
        self.L, self.a, self.b, self.B, self.ctrl = L, a, b, B, ctrl
        s = (a * b) // L
        assert s % 2 == 0 and s >= 2, 'need floor(ab/L) even'
        self.d = d = s // 2
        self.D = 3 ** d
        self.F = F = GF3k(k)
        self.q = F.q
        self.MULf = F.MUL.astype(np.uint8 if F.q <= 256 else np.uint16).reshape(-1)
        self.DIGT = np.ascontiguousarray(F.DIG.T)               # (k, q) int8
        self.P3 = F.P3
        self.POP = popcount_table(L)
        D = self.D
        self.Tdeg = np.array([sum((t // 3 ** j) % 3 for j in range(d)) for t in range(D)], dtype=np.int64)
        self.Tsig = np.array([(-1) ** sum(1 for j in range(d) if (t // 3 ** j) % 3 == 1) for t in range(D)])
        self.Tsets = [np.nonzero(self.Tdeg == i)[0] for i in range(2 * d + 1)]
        self.comp = (D - 1) - np.arange(D)
        if ctrl == 'char2P3':
            self.PHI, self.PSI, self.CHI = PHI_2, PSI_2, CHI_2
        else:
            self.PHI, self.PSI, self.CHI = PHI_S, PSI_S, CHI_S
        self.DEC = self.CHI.T.copy()

    # ---- field helpers (counted where the algorithm does the operation) ----
    def gmul(self, u, v):
        CNT['mul'] += int(np.broadcast(u, v).size)
        q = self.q
        idx = u.astype(np.int64) * q + v.astype(np.int64)
        return self.MULf[idx].astype(np.int64)

    def planes(self, idx):
        """(..., ) element indices -> (k, ...) digit planes int8."""
        return self.DIGT[:, idx]

    def index(self, pl):
        """(k, ...) digit planes (any ints) -> element indices."""
        r = np.zeros(pl.shape[1:], dtype=np.int64)
        for i in range(pl.shape[0]):
            r += (pl[i].astype(np.int64) % 3) * int(self.P3[i])
        return r

    def sum_last(self, prods):
        """field sum over the last axis of an array of element indices (counted: n-1 additions per sum)."""
        n = prods.shape[-1]
        CNT['add'] += int(np.prod(prods.shape[:-1])) * (n - 1)
        pl = self.planes(prods).astype(np.int64).sum(axis=-1)
        return self.index(pl)

    def transform(self, pl, Mat, sin):
        """Apply Mat (rows new letters, cols old letters, entries 0/+1/-1) along every axis; pl (k, sin^L) int8."""
        L = self.L; k = pl.shape[0]
        sout = Mat.shape[0]
        arr = pl
        for l in range(L):
            A = arr.reshape(k, sout ** l, sin, sin ** (L - l - 1))
            out = np.empty((k, sout ** l, sout, sin ** (L - l - 1)), dtype=np.int8)
            for t in range(sout):
                js = sorted([j for j in range(sin) if Mat[t, j]], key=lambda j: -Mat[t, j])   # a +1 entry first
                assert Mat[t, js[0]] == 1
                acc = A[:, :, js[0], :].copy()
                for j in js[1:]:
                    if Mat[t, j] == 1:
                        acc += A[:, :, j, :]
                    else:
                        acc -= A[:, :, j, :]
                    CNT['add'] += acc.size // k
                if len(js) > 1:
                    np.remainder(acc, 3, out=acc)
                out[:, :, t, :] = acc
            arr = out.reshape(k, -1)
        return arr

    # ---- set-up per z ----
    def forms(self, z):
        L, d, F = self.L, self.d, self.F
        phi = np.zeros((L, d), dtype=np.int64)
        if d == 1:
            phi[:, 0] = 1
            return phi
        tl = [1, z]
        for l in range(2, L):
            tl.append(F.mul(tl[-1], z)); CNT['mul'] += 1
        for l in range(L):
            phi[l, 0] = 1
            phi[l, 1] = tl[l]
            for i in range(2, d):
                phi[l, i] = F.mul(int(phi[l, i - 1]), tl[l]); CNT['mul'] += 1
        return phi

    def tables(self, phi):
        """a(P) for all P subset [L], |P| <= 2d (dense, length D), the signed table, and c(S) = a(S)_{top}."""
        L, d, D = self.L, self.d, self.D
        POP = self.POP
        AP = np.zeros((1 << L, D), dtype=np.int64)
        AP[0, 0] = 1
        pairs = {i: np.array(table_pairs(d, i), dtype=np.int64) for i in range(1, 2 * d + 1)}
        order = sorted([m for m in range(1, 1 << L) if POP[m] <= 2 * d], key=lambda m: POP[m])
        for P in order:
            i = int(POP[P]); top = P.bit_length() - 1
            prev = AP[P & ~(1 << top)]
            pr = pairs[i]
            prods = self.gmul(prev[pr[:, 0]], phi[top, pr[:, 1]])
            tg = pr[:, 2]
            ntg = len(np.unique(tg))
            CNT['add'] += len(pr) - ntg
            acc = np.zeros((self.F.k, D), dtype=np.int64)
            np.add.at(acc.T, tg, self.planes(prods).T.astype(np.int64))
            AP[P] = self.index(acc)
        # signed table a'(Q)_U = sigma(U) a(Q)_U
        APs = AP.copy()
        neg_cols = np.nonzero(self.Tsig == -1)[0]
        for i in range(0, 2 * d + 1):
            Ps = np.nonzero(POP <= 2 * d)[0]
        rowsP = np.nonzero(POP <= 2 * d)[0]
        for Pm in rowsP:
            nc = neg_cols[self.Tdeg[neg_cols] == POP[Pm]]
            APs[Pm, nc] = self.F.neg[APs[Pm, nc]]
            CNT['neg'] += len(nc)
        cS = np.zeros(1 << L, dtype=np.int64)
        Smask = np.nonzero(POP == 2 * d)[0]
        cS[Smask] = AP[Smask, D - 1]
        return AP, APs, cS

    # ---- reading ----
    def reading(self, vecs, masks, AP, right, unsigned=False):
        """vecs: (n, D) rows of X (or columns of Y transposed) for labels masks (n,); returns (k, 3^L) planes."""
        L, d, D = self.L, self.d, self.D
        sz = int(self.POP[masks[0]])
        bits = np.array([[l for l in range(L) if (int(m) >> l) & 1] for m in masks], dtype=np.int64)
        arr = np.zeros(3 ** L, dtype=np.int64)
        base = spread(masks, L, 3)
        for i in range(0, min(2 * d, sz) + 1):
            combos = list(itertools.combinations(range(sz), i))
            Pm = np.zeros((len(masks), len(combos)), dtype=np.int64)
            for j, cmb in enumerate(combos):
                for pos in cmb:
                    Pm[:, j] |= np.int64(1) << bits[:, pos]
            if right:
                Ts = self.Tsets[2 * d - i]
                coef = AP[Pm][:, :, self.comp[Ts]]
            else:
                Ts = self.Tsets[i]
                coef = AP[Pm][:, :, Ts]
            xv = vecs[:, Ts][:, None, :]
            prods = self.gmul(np.broadcast_to(xv, coef.shape), coef)
            val = self.sum_last(prods)
            idx = base[:, None] + spread(Pm, L, 3)
            arr[idx.reshape(-1)] = val.reshape(-1)
        return self.planes(arr).copy()


def main():
    L = int(sys.argv[1]); a = int(sys.argv[2]); b = int(sys.argv[3]); B = int(sys.argv[4]); inputs = sys.argv[5]
    k = int(sys.argv[6]); zmode = sys.argv[7]; sched = sys.argv[8]
    ctrl = sys.argv[9] if len(sys.argv) > 9 else 'none'
    seed = int(sys.argv[10]) if len(sys.argv) > 10 else 1
    tag = f'L{L}a{a}b{b}B{B}{inputs}k{k}{zmode.replace(":", "")}{sched}' + ('' if ctrl == 'none' else ctrl) + (f's{seed}' if seed != 1 else '')
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', f'char3_impl_{tag}_out.txt')
    lines = []

    def log(s):
        lines.append(s); print(s, flush=True)
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
    t0 = time.time()
    al = Algo(L, a, b, B, k, ctrl)
    F, d, D, q, POP = al.F, al.d, al.D, al.q, al.POP
    okS, diffU = check_P3()
    log(f'Full product, characteristic 3: L={L} a={a} b={b} s=2d={2 * d} d={d} D=3^d={D} B={B} inputs={inputs} '
        f'E=GF(3^{k}) (h={F.h}, g=X) zmode={zmode} schedule={sched} control={ctrl} seed={seed}')
    log(f'P3 check over Z: signed form = T4 - E3: {okS}; plus-sign form minus (T4 + E3) = {diffU}')
    rng = np.random.default_rng(seed)
    labr = np.array([m for m in range(1 << L) if POP[m] == a], dtype=np.int64)
    labc = np.array([m for m in range(1 << L) if POP[m] == b], dtype=np.int64)
    N0r, N0c = len(labr), len(labc)
    Nr, Nc = B * N0r, B * N0c
    hi = 3 if inputs == 'F3' else q
    X = rng.integers(0, hi, size=(Nr, D)).astype(np.int64)
    Y = rng.integers(0, hi, size=(D, Nc)).astype(np.int64)
    # direct product (not counted)
    XYp = np.zeros((F.k, Nr, Nc), dtype=np.int64)
    for T in range(D):
        pr = F.MUL[X[:, T][:, None], Y[T, :][None, :]]
        XYp += F.DIG[pr].transpose(2, 0, 1)
    XY = al.index(XYp)
    log(f'N_r = {Nr} (N0r = C(L,a) = {N0r}), N_c = {Nc} (N0c = C(L,b) = {N0c}); direct product computed')
    # ---- covering: the list Z (preprocessing; counted separately) ----
    CNT0 = dict(CNT)
    Smasks = np.nonzero(POP == 2 * d)[0]
    if d == 1:
        Zlist = [1]
        cover_note = 'd = 1: forms (1), c(S) = 1 for every S; Z = (1)'
        Delta = 0
    else:
        # Delta(S) = deg g_S from the sorted formula (proof of Lemma 5.2)
        def DeltaS(m):
            s_ = [l for l in range(L) if (m >> l) & 1]
            return sum(s_[j] * (2 * ((j + 2) // 2) - d - 1) for j in range(2 * d))
        Delta = max(DeltaS(int(m)) for m in Smasks)
        if zmode == 'explicit':
            assert q - 1 >= Delta + 1, 'field too small for the explicit list'
            Zlist = [F.pow(F.g, e) for e in range(Delta + 1)]
            cover_note = f'explicit Z = (g^0, ..., g^{Delta}), {len(Zlist)} values, no search'
        elif zmode.startswith('single'):
            e0 = int(zmode.split(':')[1]) if ':' in zmode else 1
            Zlist = [F.pow(F.g, e0)]
            cover_note = f'single z = g^{e0} (control: pairs whose every covering relabeling meets a bad S stay uncovered)'
        else:
            # greedy hitting set over all of E^*: c_z(S) for every z (counted as covering preprocessing)
            allz = [int(F.exp[e]) for e in range(q - 1)]
            good = np.zeros((len(allz), len(Smasks)), dtype=bool)
            for zi, z in enumerate(allz):
                phi = al.forms(z)
                _, _, cS = al.tables(phi)
                good[zi] = cS[Smasks] != 0
            unc = np.ones(len(Smasks), dtype=bool); Zlist = []
            while unc.any():
                gain = (good[:, unc]).sum(axis=1)
                j = int(gain.argmax()); assert gain[j] > 0
                Zlist.append(allz[j]); unc &= ~good[j]
            cover_note = (f'greedy hitting set over E^* ({q - 1} candidates): Z = {Zlist} (logs '
                          f'{[int(F.log[z]) for z in Zlist]}), {len(Zlist)} values; bound log2 C(L,2d)+1 = '
                          f'{math.floor(math.log2(math.comb(L, 2 * d))) + 1} (valid as q-1 >= 2 Delta = {2 * Delta}: '
                          f'{q - 1 >= 2 * Delta})')
    cover_cnt = {kk: CNT[kk] - CNT0[kk] for kk in CNT}
    for kk in CNT:
        CNT[kk] = CNT0[kk]
    log(f'covering: max_S Delta(S) = {Delta}; {cover_note}; preprocessing field operations {cover_cnt}')
    # ---- set-up for every z in Z (counted) ----
    setup = []
    for z in Zlist:
        c_before = dict(CNT)
        phi = al.forms(z)
        AP, APs, cS = al.tables(phi)
        goodS = cS[Smasks] != 0
        inv = np.zeros(1 << L, dtype=np.int64)
        inv[Smasks[goodS]] = F.inv_tab[cS[Smasks[goodS]]]
        CNT['inv'] += int(goodS.sum())
        setup.append(dict(z=z, AP=AP, APs=APs, cS=cS, inv=inv, nbad=int((~goodS).sum()),
                          cnt={kk: CNT[kk] - c_before[kk] for kk in CNT}))
    log('set-up per z: ' + '; '.join(f'z={s_["z"]} (log {int(F.log[s_["z"]])}): {s_["nbad"]} of {len(Smasks)} sets '
                                     f'S bad' for s_ in setup))
    uncovS = np.ones(len(Smasks), dtype=bool)
    for s_ in setup:
        uncovS &= s_['cS'][Smasks] == 0
    log(f'sets S bad for every z in Z: {int(uncovS.sum())}')
    # ---- covering table (bit operations): first (z, pi) per label pair, z-major ----
    fam = relabel_family(L)
    K = len(fam)
    pcs = [perm_masks(p, labc, L) for p in fam]
    first = np.full((N0r, N0c), -1, dtype=np.int64)
    for zi, s_ in enumerate(setup):
        for pk in range(K):
            inter = labr[:, None] & pcs[pk][None, :]
            hit = (POP[inter] == 2 * d) & (s_['cS'][inter] != 0) & (first < 0)
            first[hit] = zi * K + pk
    uncovered = int((first < 0).sum())
    used = sorted(set(first[first >= 0].reshape(-1).tolist()))
    log(f'Pi_L: {K} relabelings; covering table over {len(Zlist)} x {K} (z, pi): {uncovered} of {N0r * N0c} label '
        f'pairs uncovered; first-covering (z, pi) runs: {len(used)}; per z: '
        f'{[sum(1 for u in used if u // K == zi) for zi in range(len(Zlist))]}; bit-operation table '
        f'{len(Zlist) * K * N0r * N0c} popcount/lookups')
    if sched == 'all':
        runs = [(zi, pk) for zi in range(len(Zlist)) for pk in range(K)]
    else:
        runs = [(u // K, u % K) for u in used]
    zs_used = sorted(set(zi for zi, _ in runs))
    # ---- the algorithm ----
    result = np.zeros((Nr, Nc), dtype=np.int64); done = np.zeros((Nr, Nc), dtype=bool)
    rawres = np.zeros((Nr, Nc), dtype=np.int64)
    wrong_good = checked_good = wrong_bad = checked_bad = 0
    tA = time.time()
    setup_cnt = {kk: sum(setup[zi]['cnt'][kk] for zi in zs_used) for kk in CNT}
    algo_before = dict(CNT)
    nrun = 0
    for zi in zs_used:
        s_ = setup[zi]
        encL = []
        for bi in range(B):
            pl = al.reading(X[bi * N0r:(bi + 1) * N0r, :], labr, s_['AP'], right=False)
            encL.append(al.transform(pl, al.PHI, 3))
        for (zj, pk) in runs:
            if zj != zi:
                continue
            pc = pcs[pk]
            encR = []
            APr = s_['AP'] if ctrl == 'unsigned' else s_['APs']
            for bj in range(B):
                pl = al.reading(Y[:, bj * N0c:(bj + 1) * N0c].T.copy(), pc, APr, right=True)
                encR.append(al.transform(pl, al.PSI, 3))
            inter = labr[:, None] & pc[None, :]
            nu, mu = np.nonzero(POP[inter] == 2 * d)
            S = inter[nu, mu]
            widx = 2 * spread(labr[nu], L, 4) + spread(pc[mu], L, 4)
            gS = s_['cS'][S] != 0
            isfirst = first[nu, mu] == zi * K + pk
            for bi in range(B):
                ia = al.index(encL[bi])
                for bj in range(B):
                    ib = al.index(encR[bj])
                    prod = al.gmul(ia, ib)
                    dec = al.transform(al.planes(prod), al.DEC, 4)
                    raw = al.index(dec[:, widx])
                    I = bi * N0r + nu; J = bj * N0c + mu
                    save = CNT['mul']
                    val = F.MUL[raw, s_['inv'][S]]
                    CNT['mul'] = save
                    wrong_good += int((val[gS] != XY[I[gS], J[gS]]).sum()); checked_good += int(gS.sum())
                    wrong_bad += int((raw[~gS] != 0).sum()); checked_bad += int((~gS).sum())
                    CNT['mul'] += int(isfirst.sum())
                    result[I[isfirst], J[isfirst]] = val[isfirst]
                    rawres[I[isfirst], J[isfirst]] = raw[isfirst]
                    done[I[isfirst], J[isfirst]] = True
            nrun += 1
            if nrun % 10 == 0:
                print(f'  run {nrun}/{len(runs)}  {time.time() - tA:.1f}s', flush=True)
    algo_cnt = {kk: CNT[kk] - algo_before[kk] for kk in CNT}
    tB = time.time()
    log(f'end to end: {int(done.sum())} of {Nr * Nc} entries computed; wrong entries of the assembled product: '
        f'{int((result != XY).sum())} (of which not computed: {int((~done).sum())}); class outputs of every executed '
        f'tile-run: good S {checked_good} checked, {wrong_good} wrong; bad S (c_z(S) = 0) {checked_bad} checked, '
        f'{wrong_bad} with raw value != 0')
    log(f'time {tB - tA:.1f}s for {len(runs)} (z, pi) runs x {B * B} band pairs')
    tot = {kk: setup_cnt[kk] + algo_cnt[kk] for kk in CNT}
    ninv = sum(len(Smasks) - setup[zi]['nbad'] for zi in zs_used)
    fm = formula(L, a, b, d, B, len(zs_used), len(runs), ninv, ndiv=B * B * int((first >= 0).sum()))
    eq = (tot['mul'], tot['add'], tot['neg'], tot['inv']) == (fm['mul'], fm['add'], fm['neg'], fm['inv'])
    T = sum(tot.values())
    log(f'counted (schedule {sched}, {len(zs_used)} z, {len(runs)} (z, pi) runs): mul {tot["mul"]}, add {tot["add"]}, '
        f'neg {tot["neg"]}, inv {tot["inv"]}, total {T}; formula mul {fm["mul"]}, add {fm["add"]}, neg {fm["neg"]}, '
        f'inv {fm["inv"]}, total {fm["total"]}; equal: {eq}; per entry {T / (Nr * Nc):.2f}; trivial 2D-1 = {2 * D - 1}')
    log('   formula parts: ' + ', '.join(f'{kk}={fm[kk]}' for kk in
                                        ['forms_mul', 'table_mul', 'table_add', 'sign_neg', 'inv', 'readL_mul',
                                         'readL_add', 'readR_mul', 'readR_add', 'enc_add', 'leaf_mul', 'dec_add',
                                         'div_mul']))
    if sched == 'first':
        fa = formula(L, a, b, d, B, len(Zlist), len(Zlist) * K, len(Zlist) * len(Smasks) - sum(s_['nbad'] for s_ in setup))
        log(f'formula for schedule all ({len(Zlist)} z x {K} relabelings): total {fa["total"]}, per entry '
            f'{fa["total"] / (Nr * Nc):.2f}')
    log(f'control (no division by c_z(S)): {int((rawres[done] != XY[done]).sum())} of {int(done.sum())} computed '
        f'entries wrong (0 expected only when every c_z(S) used is 1)')
    log(f'total wall time {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
