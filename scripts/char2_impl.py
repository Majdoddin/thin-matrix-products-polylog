r"""The algorithm of Section 4 (Theorem 1.1(a)): the full product of an N x D and a D x N matrix over a field of
characteristic 2 from the tensor T4 and its four-term decomposition (Lemma 2.1), vectorised, with every field
operation counted as executed, compared with the exact count of Table 2, and checked against direct multiplication.

Algorithm (Section 4; Lemma 3.1, Proposition 3.2, Lemma 3.3):
  * L even, a = L/2 (any a is accepted), d = floor(a^2/L), D = 2^d, N0 = C(L, a), N = B N0;
  * field GF(16) (gf2k.py), nodes t_l = l + 1, Vandermonde forms phi_l = (1, t_l, ..., t_l^{d-1});
  * rows of band i labelled by the a-subsets r_nu of [L], columns of band j by c_mu; relabelings Pi_L (the
    transposition path through the cyclic shifts, Lemma 3.3); column mu of band j sits at pi(c_mu) in the tile-run
    (i, j, pi), that is, the run of the recursion for the band pair (i, j) and the relabeling pi;
  * left array  a_i[(r, P)]     = <X_row, a(P)>,        P subset r,  |P| <= d   (string: p0 on P, x1 on r\P);
    right array b_{j,pi}[(c', Q)] = <Y_col, a(Q)^dual>, Q subset c', |Q| <= d, c' = pi(c_mu);
  * encodings by the 4 x 3 matrices of the decomposition's terms along every axis, leaf products (4^L), decoding by
    the 4 x 4 matrix term -> covered outputs along every axis; the entry (row, col) is read at w = (r, c') and divided
    by det(Phi_S) = prod_{l<l' in S} (t_l + t_l'), S = r cap c', in the first relabeling with |S| = d.
Two schedules: 'all' runs every pi in Pi_L (K = L(L-1), the count of Theorem 1.1(a)); 'first' runs only the
relabelings that are the first covering one for some label pair (computed once, shared by all band pairs; Remark
4.3(iv)). In the check every "class output" of every tile-run (every pair at intersection d in every relabeling, not
only the first-covering ones) is compared with XY. Counted operations (GF(2^k) operations): mul, add (XOR), inv; and
separately the one-time covering table (popcounts of L-bit words). The dense transforms are counted as executed
(XORs x slice size). Control: the assembled product without the division by det(Phi_S).
Usage: python3 char2_impl.py L B inputs(F2|GF16) [seed]   ->  ../outputs/char2_impl_L{L}B{B}{inputs}_out.txt
"""
import sys, os, time, math, itertools
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gf2k import GF

F = GF(4)
TMf = F.T.astype(np.uint8).reshape(-1)        # flat 16 x 16 multiplication table
CNT = {'mul': 0, 'add': 0, 'inv': 0}


def gmul(u, v):
    """elementwise product of uint8 arrays (counted)."""
    CNT['mul'] += int(np.broadcast(u, v).size)
    return TMf[(u.astype(np.intp) << 4) | v.astype(np.intp)]


def popcount_table(L):
    t = np.zeros(1 << L, dtype=np.int64)
    for m in range(1, 1 << L):
        t[m] = t[m >> 1] + (m & 1)
    return t


# The decomposition of Lemma 2.1 in characteristic 2: term x left letter (x0,x1,p0), term x right letter (y0,y1,q0),
# and the decoding matrix output z x term (z0,z1,z2,w0) x (terms 1..4).
PHI = np.array([[1, 1, 0], [1, 0, 1], [0, 1, 0], [1, 0, 0]], dtype=np.int64)
PSI = np.array([[1, 0, 0], [0, 1, 0], [1, 0, 1], [1, 1, 0]], dtype=np.int64)
CHI = np.array([[1, 0, 0, 1], [0, 1, 0, 1], [0, 0, 1, 1], [0, 0, 0, 1]], dtype=np.int64)
DEC = CHI.T.copy()


def transform(flat, L, Mat, sin):
    """Apply Mat (rows = new letters, cols = old letters) along every axis of an array of shape (sin,)*L given flat
    (C order, axis 0 = level 0 most significant).  XOR as addition, counted as executed."""
    sout = Mat.shape[0]
    arr = flat
    for l in range(L):
        A = arr.reshape(sout ** l, sin, sin ** (L - l - 1))
        out = np.empty((sout ** l, sout, sin ** (L - l - 1)), dtype=np.uint8)
        for t in range(sout):
            js = [j for j in range(sin) if Mat[t, j]]
            acc = A[:, js[0], :].copy()
            for j in js[1:]:
                acc ^= A[:, j, :]
                CNT['add'] += acc.size
            out[:, t, :] = acc
        arr = out.reshape(-1)
    return arr


def relabel_family(L):
    """Pi_L of Lemma 3.3: s^k o t_1 o ... o t_j, s = (0 L-1) o ... o (0 1)."""
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
    """sum_l bit_l(m) base^(L-1-l)."""
    out = np.zeros(masks.shape, dtype=np.int64)
    for l in range(L):
        out += ((masks >> l) & 1) * (base ** (L - 1 - l))
    return out


def formula(L, a, d, B, K):
    """Exact operation count of Table 2 (GF(2^k) operations)."""
    N0 = math.comb(L, a); N = B * N0
    R = math.comb(a + d, d)                       # sum_i C(a,i) C(d,i)
    S0 = sum(math.comb(a, i) for i in range(d + 1))
    f = {}
    f['table_mul'] = sum(math.comb(L, i) * i * math.comb(d, i) for i in range(1, d + 1))
    f['table_add'] = sum(math.comb(L, i) * (i - 1) * math.comb(d, i) for i in range(1, d + 1))
    c2 = math.comb(d, 2)
    f['det_add'] = math.comb(L, d) * c2
    f['det_mul'] = math.comb(L, d) * max(c2 - 1, 0)
    f['det_inv'] = math.comb(L, d)
    f['read_mul'] = B * N0 * R + B * K * N0 * R
    f['read_add'] = (B + B * K) * N0 * (R - S0)
    f['enc_add'] = (B + B * K) * 2 * (4 ** L - 3 ** L)
    f['leaf_mul'] = B * B * K * 4 ** L
    f['dec_add'] = B * B * K * 3 * L * 4 ** (L - 1)
    f['div_mul'] = N * N
    f['mul'] = f['table_mul'] + f['det_mul'] + f['read_mul'] + f['leaf_mul'] + f['div_mul']
    f['add'] = f['table_add'] + f['det_add'] + f['read_add'] + f['enc_add'] + f['dec_add']
    f['inv'] = f['det_inv']
    f['total'] = f['mul'] + f['add'] + f['inv']
    f['per_entry'] = f['total'] / (N * N)
    return f


def main():
    L = int(sys.argv[1]); B = int(sys.argv[2]); inputs = sys.argv[3]
    seed = int(sys.argv[4]) if len(sys.argv) > 4 else 1
    a = L // 2
    d = (a * a) // L
    D = 1 << d
    assert L <= 16 and d >= 1
    out_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', f'char2_impl_L{L}B{B}{inputs}_out.txt')
    lines = []

    def log(s):
        lines.append(s); print(s, flush=True)
        with open(out_path, 'w') as fh:
            fh.write('\n'.join(lines) + '\n')
    t0 = time.time()
    rng = np.random.default_rng(seed)
    POP = popcount_table(L)
    labels = np.array([m for m in range(1 << L) if POP[m] == a], dtype=np.int64)
    N0 = len(labels); N = B * N0
    log(f'Full product, characteristic 2: L={L} a={a} d={d} D={D} N0=C(L,a)={N0} B={B} N={N} inputs={inputs} seed={seed} '
        f'field GF(16), t_l=l+1')
    hi = 2 if inputs == 'F2' else 16
    X = rng.integers(0, hi, size=(N, D)).astype(np.uint8)
    Y = rng.integers(0, hi, size=(D, N)).astype(np.uint8)
    # direct product (not counted)
    XY = np.zeros((N, N), dtype=np.uint8)
    for T in range(D):
        XY ^= TMf[(X[:, T].astype(np.intp)[:, None] << 4) | Y[T, :].astype(np.intp)[None, :]]
    # ---------- setup (counted) ----------
    tl = [l + 1 for l in range(L)]
    phi = np.array([[F.pow(tl[l], i) for i in range(d)] for l in range(L)], dtype=np.uint8)
    # table a(P) for |P| <= d, as dense coefficient vectors of length D (index = bitmask T)
    Ps = [m for m in range(1 << L) if POP[m] <= d]
    AP = np.zeros((1 << L, D), dtype=np.uint8)
    AP[0, 0] = 1
    for P in sorted(Ps, key=lambda m: POP[m]):
        if P == 0:
            continue
        top = P.bit_length() - 1
        prev = AP[P & ~(1 << top)]
        v = np.zeros(D, dtype=np.uint8)
        first = np.ones(D, dtype=bool)
        for T in np.nonzero(POP[np.arange(D)] == POP[P] - 1)[0]:
            for i in range(d):
                if (T >> i) & 1:
                    continue
                prod = gmul(np.array([prev[T]]), np.array([phi[top, i]]))[0]
                tgt = T | (1 << i)
                if first[tgt]:
                    v[tgt] = prod; first[tgt] = False
                else:
                    v[tgt] ^= prod; CNT['add'] += 1
        AP[P] = v
    # det(Phi_S)^{-1} for all d-sets S
    detinv = np.zeros(1 << L, dtype=np.uint8)
    for S in range(1 << L):
        if POP[S] != d:
            continue
        lev = [l for l in range(L) if (S >> l) & 1]
        dv = None
        for x, y in itertools.combinations(lev, 2):
            s = tl[x] ^ tl[y]; CNT['add'] += 1
            if dv is None:
                dv = s
            else:
                dv = int(gmul(np.array([dv], dtype=np.uint8), np.array([s], dtype=np.uint8))[0])
        if dv is None:
            dv = 1
        detinv[S] = F.inv(dv); CNT['inv'] += 1
    cnt_setup = dict(CNT)
    # covering table (one-time; bit operations, not field operations)
    fam = relabel_family(L)
    K_all = len(fam)
    first = np.full((N0, N0), -1, dtype=np.int64)
    pc_cache = []
    for k, p in enumerate(fam):
        pc = perm_masks(p, labels, L)
        pc_cache.append(pc)
        hit = (POP[labels[:, None] & pc[None, :]] == d) & (first < 0)
        first[hit] = k
    popcount_ops = K_all * N0 * N0
    assert (first >= 0).all(), 'covering failed'
    used = sorted(set(first.reshape(-1).tolist()))
    log(f'Pi_L: {K_all} relabelings (distinct: {len(set(fam))}); every label pair covered; first-covering '
        f'relabelings: {len(used)} distinct, max index {max(used)}; covering table {popcount_ops} popcounts '
        f'(one-time, bit operations)')
    # string-index helpers
    sp4_lab = spread(labels, L, 4)
    # sub-masks of each label with |P| <= d, by size
    sub_by_size = []
    for i in range(d + 1):
        combos = list(itertools.combinations(range(a), i))
        sub_by_size.append(combos)
    Tsets = [np.nonzero(POP[np.arange(D)] == i)[0] for i in range(d + 1)]
    comp = (D - 1) ^ np.arange(D)

    def label_bits(m):
        return [l for l in range(L) if (m >> l) & 1]
    # positions of the bits of each label (for subset enumeration)
    lab_bits = np.array([label_bits(int(m)) for m in labels], dtype=np.int64)          # (N0, a)

    def subsets_of_labels(masks_bits, i):
        """For each label (row of masks_bits: its a levels) all i-subsets as masks: shape (N0, C(a,i))."""
        combos = sub_by_size[i]
        if i == 0:
            return np.zeros((masks_bits.shape[0], 1), dtype=np.int64)
        out = np.zeros((masks_bits.shape[0], len(combos)), dtype=np.int64)
        for j, cmb in enumerate(combos):
            for pos in cmb:
                out[:, j] |= np.int64(1) << masks_bits[:, pos]
        return out

    def reading(vecs, masks_bits, masks, right):
        """vecs: (N0, D) rows of X (or columns of Y, transposed); returns flat (3,)*L array of the band."""
        arr = np.zeros(3 ** L, dtype=np.uint8)
        base = spread(masks, L, 3)
        for i in range(d + 1):
            Pm = subsets_of_labels(masks_bits, i)                  # (N0, C(a,i))
            Ts = Tsets[i]                                          # monomials of degree i
            coef = AP[Pm][:, :, (comp[Tsets[d - i]] if right else Ts)]   # (N0, C(a,i), C(d,i))
            xv = vecs[:, (Tsets[d - i] if right else Ts)][:, None, :]   # (N0, 1, C(d,i))
            prods = gmul(np.broadcast_to(xv, coef.shape), coef)
            val = np.bitwise_xor.reduce(prods, axis=2)
            CNT['add'] += prods.shape[0] * prods.shape[1] * (prods.shape[2] - 1)
            idx = base[:, None] + spread(Pm, L, 3)
            arr[idx.reshape(-1)] = val.reshape(-1)
        return arr
    # ---------- the algorithm ----------
    schedules = {'all': list(range(K_all)), 'first': used}
    cnt_sched = {}
    wrong_class = 0; checked_class = 0
    result = np.zeros((N, N), dtype=np.uint8); done = np.zeros((N, N), dtype=bool)
    rawres = np.zeros((N, N), dtype=np.uint8)
    tA = time.time()
    # left encodings per row band
    CNT0 = dict(CNT)
    encL = []
    for bi in range(B):
        arr = reading(X[bi * N0:(bi + 1) * N0, :], lab_bits, labels, right=False)
        encL.append(transform(arr, L, PHI, 3))
    left_cnt = {k: CNT[k] - CNT0[k] for k in CNT}
    per_pi = {}
    for k in range(K_all):
        p = fam[k]
        pc = pc_cache[k]
        pc_bits = np.array([label_bits(int(m)) for m in pc], dtype=np.int64)
        c_before = dict(CNT)
        encR = []
        for bj in range(B):
            arr = reading(Y[:, bj * N0:(bj + 1) * N0].T.copy(), pc_bits, pc, right=True)
            encR.append(transform(arr, L, PSI, 3))
        inter = POP[labels[:, None] & pc[None, :]]
        nu, mu = np.nonzero(inter == d)
        widx = 2 * sp4_lab[nu] + spread(pc, L, 4)[mu]
        S = labels[nu] & pc[mu]
        isfirst = first[nu, mu] == k
        for bi in range(B):
            for bj in range(B):
                prod = gmul(encL[bi], encR[bj])
                mult = transform(prod, L, DEC, 4)
                raw = mult[widx]
                # check every class output of this tile-run (not counted)
                save = CNT['mul']
                val = TMf[(raw.astype(np.intp) << 4) | detinv[S].astype(np.intp)]
                CNT['mul'] = save
                I = bi * N0 + nu; J = bj * N0 + mu
                wrong_class += int((val != XY[I, J]).sum()); checked_class += len(val)
                # the algorithm's division: once per entry, at its first covering relabeling
                CNT['mul'] += int(isfirst.sum())
                result[I[isfirst], J[isfirst]] = val[isfirst]
                rawres[I[isfirst], J[isfirst]] = raw[isfirst]
                done[I[isfirst], J[isfirst]] = True
        per_pi[k] = {kk: CNT[kk] - c_before[kk] for kk in CNT}
        if k % 10 == 0 or k == K_all - 1:
            print(f'  relabeling {k + 1}/{K_all}  {time.time() - tA:.1f}s', flush=True)
    tB = time.time()
    # counts per schedule: setup + left + sum over the schedule's relabelings
    for name, ks in schedules.items():
        c = {kk: cnt_setup[kk] + left_cnt[kk] + sum(per_pi[k][kk] for k in ks) for kk in CNT}
        # the division multiplications happen only in first-covering relabelings, which both schedules contain
        cnt_sched[name] = c
    log(f'end to end: {int(done.sum())} of {N * N} entries computed; wrong entries of the assembled product: '
        f'{int((result != XY).sum())}; every class output of every tile-run checked: {checked_class} checks, '
        f'{wrong_class} wrong')
    log(f'time {tB - tA:.1f}s for {K_all} relabelings x {B * B} band pairs')
    for name, K in (('all', K_all), ('first', len(used))):
        fm = formula(L, a, d, B, K)
        c = cnt_sched[name]
        tot = c['mul'] + c['add'] + c['inv']
        log(f'schedule {name} (K = {K}): counted mul {c["mul"]}, add {c["add"]}, inv {c["inv"]}, total {tot}; '
            f'formula mul {fm["mul"]}, add {fm["add"]}, inv {fm["inv"]}, total {fm["total"]}; '
            f'equal: {(c["mul"], c["add"], c["inv"]) == (fm["mul"], fm["add"], fm["inv"])}; '
            f'per entry {tot / (N * N):.2f}; trivial N^2 D (mul+add) per entry {2 * D}')
        log('   formula parts: ' + ', '.join(f'{kk}={fm[kk]}' for kk in
                                            ['table_mul', 'table_add', 'det_add', 'det_mul', 'det_inv', 'read_mul',
                                             'read_add', 'enc_add', 'leaf_mul', 'dec_add', 'div_mul']))
    # control: no division by det
    log(f'control (no division by det(Phi_S)): {int((rawres != XY).sum())} of {N * N} entries wrong '
        f'(expected 0 only when d = 1, where det = 1)')
    log(f'total wall time {time.time() - t0:.1f}s')


if __name__ == '__main__':
    main()
