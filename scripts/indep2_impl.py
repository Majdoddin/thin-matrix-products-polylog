r"""Second implementation of the algorithm of Section 4 (steps 0-5, Theorem 1.1(a)), written from the text only,
with operation counting, against direct multiplication.

Usage: python3 indep2_impl.py OUTFILE CASE [CASE ...]
CASE = N,m,Fn,seed,relab,schedule,classcheck   e.g. 1296,3,1,7,adjacent,all,1
  Fn: F = GF(2^Fn) (1 = F_2); the algorithm computes in E = F if |F| >= L, else in GF(2^(Fn k)), k = ceil(log2 L);
  relab: adjacent (transpositions (j-1 j)) | star (transpositions (0 L-j)): two factorizations of the cyclic shift
  in Lemma 3.3; schedule: all (K = L(L-1) relabelings, as in the theorem) | used (only first-covering ones; same
  outputs, fewer tile-runs; for speed at L = 12); classcheck: 1 = check every output at intersection d
  of every tile-run (all duplicates), 0 = only the read-out.
"""
import sys, time, itertools
import numpy as np
from math import comb
from indep2_gf import GF, embedding
from indep2_core import (axis_transform, PHI_ROWS, PSI_ROWS, CHIT_ROWS, left_index, right_index,
                          out_index, subsets, popcount)

def compose(f, g):
    return [f[g[l]] for l in range(len(g))]

def transposition(L, i, j):
    t = list(range(L)); t[i], t[j] = t[j], t[i]; return t

def relabelings(L, relab):
    s = [(l + 1) % L for l in range(L)]
    if relab == 'adjacent':
        ts = [transposition(L, j - 1, j) for j in range(1, L)]
    else:
        ts = [transposition(L, 0, L - j) for j in range(1, L)]
    prod = list(range(L))
    for t in ts:
        prod = compose(prod, t)
    assert prod == s
    seq = []
    sk = list(range(L))
    for k in range(L):
        cur = sk
        seq.append(cur)
        for j in range(L - 2):
            cur = compose(cur, ts[j])
            seq.append(cur)
        sk = compose(sk, s)
    assert len(seq) == L * (L - 1)
    return seq

def apply_mask(pi, c):
    m = 0
    for l in range(len(pi)):
        if (c >> l) & 1:
            m |= 1 << pi[l]
    return m

def formula(L, B, K):
    a, d = L // 2, L // 4
    N0 = comb(L, a)
    R = comb(a + d, d); S0 = sum(comb(a, i) for i in range(d + 1))
    f = {}
    f['table_mul'] = sum(comb(L, i) * i * comb(d, i) for i in range(1, d + 1))
    f['table_add'] = sum(comb(L, i) * (i - 1) * comb(d, i) for i in range(1, d + 1))
    f['det_mul'] = comb(L, d) * max(comb(d, 2) - 1, 0)
    f['det_add'] = comb(L, d) * comb(d, 2)
    f['inv'] = comb(L, d)
    f['read_mul'] = (B + B * K) * N0 * R
    f['read_add'] = (B + B * K) * N0 * (R - S0)
    f['enc_add'] = (B + B * K) * 2 * (4 ** L - 3 ** L)
    f['leaf_mul'] = B * B * K * 4 ** L
    f['dec_add'] = B * B * K * 3 * L * 4 ** (L - 1)
    f['div_mul'] = (B * N0) ** 2
    return f

def run(N, m, Fn, seed, relab, schedule, classcheck, p):
    t0 = time.time()
    rng = np.random.default_rng(seed)
    F = GF(Fn)
    q = (m - 1).bit_length()          # ceil(log2 m)
    L, a, d = 4 * q, 2 * q, q
    D = 1 << d
    assert d == (a * a) // L and D >= m
    if F.q >= L:
        E = F
    else:
        k = (L - 1).bit_length()      # ceil(log2 L)
        E = GF(Fn * k)
    emb = embedding(F, E)
    N0 = comb(L, a)
    B = -(-N // N0)
    Np = B * N0
    hyp = (N >= 256 and 2 <= m and 16 * m ** 4 <= N)
    p(f'\n### case N={N} m={m} F=GF(2^{Fn}) E=GF(2^{E.n}) seed={seed} relab={relab} schedule={schedule}')
    p(f'L={L} a={a} d={d} D={D} N0={N0} B={B} N\'={Np}; theorem hypotheses (N>=256, 2<=m<=N^(1/4)/2): {hyp}')
    ops = dict.fromkeys(['table_mul', 'table_add', 'det_mul', 'det_add', 'inv', 'read_mul', 'read_add',
                         'enc_add', 'leaf_mul', 'dec_add', 'div_mul'], 0)
    mul = E.mul_t
    X = rng.integers(0, F.q, (N, m)); Y = rng.integers(0, F.q, (m, N))
    Xp = np.zeros((Np, D), dtype=np.uint8); Xp[:N, :m] = emb[X]
    Yp = np.zeros((D, Np), dtype=np.uint8); Yp[:m, :N] = emb[Y]
    # distinct t_l in E (random choice)
    t = [int(v) for v in rng.choice(E.q, L, replace=False)]
    phis = [[E.pow(t[l], i) for i in range(d)] for l in range(L)]   # phi_l = sum_i t_l^(i-1) x_i
    full = D - 1
    # ---- step 0(i): a(P) for all P subset [L], |P| <= d, as a(P - max P) * phi_{max P}
    aP = {0: np.zeros(D, dtype=np.uint8)}
    aP[0][0] = 1
    for size in range(1, d + 1):
        for Pt in itertools.combinations(range(L), size):
            P = sum(1 << l for l in Pt)
            lmax = max(Pt)
            f = aP[P ^ (1 << lmax)]
            g = np.zeros(D, dtype=np.uint8)
            contrib = np.zeros(D, dtype=np.int64)
            for T in range(D):
                if popcount(T) != size - 1:
                    continue
                for i in range(d):
                    if not (T >> i) & 1:
                        g[T | (1 << i)] ^= mul[f[T], phis[lmax][i]]
                        ops['table_mul'] += 1
                        contrib[T | (1 << i)] += 1
            ops['table_add'] += int(np.sum(np.maximum(contrib - 1, 0)))
            aP[P] = g
    # ---- step 0(ii): det(Phi_S)^(-1) for all d-sets S
    detinv = {}
    for St in itertools.combinations(range(L), d):
        S = sum(1 << l for l in St)
        det = None
        for i1 in range(d):
            for i2 in range(i1 + 1, d):
                sm = t[St[i1]] ^ t[St[i2]]; ops['det_add'] += 1
                if det is None:
                    det = sm
                else:
                    det = int(mul[det, sm]); ops['det_mul'] += 1
        if det is None:
            det = 1
        detinv[S] = E.inv(det); ops['inv'] += 1
    # ---- step 0(iii): relabelings and covering table (bit operations, not counted)
    labels = [sum(1 << l for l in c) for c in itertools.combinations(range(L), a)]
    seq = relabelings(L, relab)
    K_all = len(seq)
    lab = np.array(labels, dtype=np.int64)
    PCs = []
    cover = np.full((N0, N0), -1, dtype=np.int64)
    for kk, pi in enumerate(seq):
        pc = np.array([apply_mask(pi, c) for c in labels], dtype=np.int64)
        PCs.append(pc)
        inter = np.bitwise_count(lab[:, None] & pc[None, :])
        newly = (cover < 0) & (inter == d)
        cover[newly] = kk
    assert np.all(cover >= 0), 'some pair uncovered'
    runs = list(range(K_all)) if schedule == 'all' else sorted(set(np.unique(cover).tolist()))
    K = len(runs)
    pos = {kk: ii for ii, kk in enumerate(runs)}
    cover_pos = np.vectorize(pos.get)(cover)
    # output index and det^-1 per (nu, mu) under its first covering relabeling (vectorised;
    # output digit at level l is 2 [l in r] + [l in c]: z0 = 0, z1 = 1, z2 = 2, w0 = 3)
    def spread(x):
        y = np.zeros_like(x)
        for l in range(L):
            y += ((x >> l) & 1) * (4 ** l)
        return y
    def oidx(r, c):
        return 2 * spread(r) + spread(c)
    detinv_arr = np.zeros(1 << L, dtype=np.uint8)
    for S_, v_ in detinv.items():
        detinv_arr[S_] = v_
    PCarr = np.array(PCs)                          # (K_all, N0)
    cmat = PCarr[cover, np.arange(N0)[None, :]]    # column set of (nu, mu) in its covering run
    widx = oidx(lab[:, None], cmat)
    dinv = detinv_arr[lab[:, None] & cmat]
    for _ in range(200):                           # spot check against the scalar index
        nu, mu = rng.integers(0, N0, 2)
        assert widx[nu, mu] == out_index(labels[nu], int(cmat[nu, mu]), L)
    class_cache = {}
    def class_data(kk):
        if kk in class_cache:
            return class_cache[kk]
        pc = PCs[kk]
        inter = np.bitwise_count(lab[:, None] & pc[None, :])
        nus, mus = np.nonzero(inter == d)
        cs = pc[mus]
        val = (nus, mus, oidx(lab[nus], cs), detinv_arr[lab[nus] & cs])
        if L <= 8:
            class_cache[kk] = val
        return val
    p(f'set-up done {time.time()-t0:.1f}s; relabelings run K={K} of {K_all}')
    # ---- step 1: reading, left arrays (one per row band), vectorised over bands
    A = np.zeros((B, 3 ** L), dtype=np.uint8)
    for nu, r in enumerate(labels):
        rows = Xp[nu::N0]                         # rows i*N0 + nu, i = 0..B-1
        for P in subsets(r):
            sz = popcount(P)
            if sz > d:
                continue
            Ts = [T for T in range(D) if popcount(T) == sz]
            val = np.zeros(B, dtype=np.uint8)
            for T in Ts:
                val ^= mul[rows[:, T], aP[P][T]]
            ops['read_mul'] += B * len(Ts); ops['read_add'] += B * (len(Ts) - 1)
            A[:, left_index(r, P, L)] = val
    # ---- step 2: left encodings
    PhiA, adds = axis_transform(A, L, 3, PHI_ROWS); ops['enc_add'] += adds
    del A
    XYp = np.zeros((Np, Np), dtype=np.uint8)      # direct product over E, for checking
    for T in range(D):
        XYp ^= mul[Xp[:, T][:, None], Yp[T, :][None, :]]
    result = np.zeros((Np, Np), dtype=np.uint8)
    nodiv = np.zeros((Np, Np), dtype=np.uint8)
    class_checked = 0; class_wrong = 0
    # ---- per column band j
    batch = K if L <= 8 else 1
    for j in range(B):
        for b0 in range(0, K, batch):
            chunk = runs[b0:b0 + batch]
            Barr = np.zeros((len(chunk), 3 ** L), dtype=np.uint8)
            for ci, kk in enumerate(chunk):
                pc = PCs[kk]
                for mu in range(N0):
                    c = int(pc[mu])
                    col = Yp[:, j * N0 + mu]
                    for Q in subsets(c):
                        sz = popcount(Q)
                        if sz > d:
                            continue
                        Ts = [T for T in range(D) if popcount(T) == d - sz]
                        val = 0
                        for T in Ts:
                            val ^= int(mul[col[T], aP[Q][full ^ T]])
                        ops['read_mul'] += len(Ts); ops['read_add'] += len(Ts) - 1
                        Barr[ci, right_index(c, Q, L)] = val
            PsiB, adds = axis_transform(Barr, L, 3, PSI_ROWS); ops['enc_add'] += adds
            del Barr
            for i in range(B):
                # ---- step 3: leaf products; step 4: decoding
                prod = mul[PhiA[i][None, :], PsiB]
                ops['leaf_mul'] += prod.size
                Mult, adds = axis_transform(prod, L, 4, CHIT_ROWS); ops['dec_add'] += adds
                del prod
                # ---- step 5: read-out with division, for (nu, mu) whose first covering run is in chunk
                for ci, kk in enumerate(chunk):
                    sel = (cover == kk)
                    vals = Mult[ci][widx[sel]]
                    blk = result[i * N0:(i + 1) * N0, j * N0:(j + 1) * N0]
                    blk[sel] = mul[vals, dinv[sel]]
                    ops['div_mul'] += int(sel.sum())
                    nblk = nodiv[i * N0:(i + 1) * N0, j * N0:(j + 1) * N0]
                    nblk[sel] = vals
                    if classcheck:
                        # every output of this tile-run at intersection d (all duplicates)
                        nus, mus, oi, dv = class_data(kk)
                        v = mul[Mult[ci][oi], dv]
                        class_checked += len(nus)
                        class_wrong += int(np.sum(v != XYp[i * N0 + nus, j * N0 + mus]))
                del Mult
            del PsiB
    wrong = int(np.sum(result[:N, :N] != XYp[:N, :N]))
    wrong_pad = int(np.sum(result != XYp))
    nodiv_wrong = int(np.sum(nodiv[:N, :N] != XYp[:N, :N]))
    inF = bool(np.all(np.isin(result[:N, :N], emb)))
    p(f'entries wrong: {wrong} of {N*N} (padded {wrong_pad} of {Np*Np}); results in F: {inF}; '
      f'control without division: {nodiv_wrong} of {N*N} wrong')
    if classcheck:
        p(f'class outputs (every output at intersection d of every tile-run run): {class_wrong} wrong of {class_checked}')
    f = formula(L, B, K)
    same = {k_: (ops[k_] == f[k_]) for k_ in f}
    p('ops counted:', ops)
    p('equal to Table 2 with K = runs:', all(same.values()), '' if all(same.values()) else same)
    tot = sum(ops.values())
    p(f'total ops {tot} = {tot/(N*N):.2f} per entry of N^2 = {tot/(Np*Np):.2f} per padded entry; '
      f'8 L^4 N^2 = {8*L**4*N*N} (ratio {tot/(8*L**4*N*N):.4f}); 2 L^4 N\'^2 ratio {tot/(2*L**4*Np*Np):.4f}')
    p(f'leaf products (rank) {ops["leaf_mul"]}; 8 L^3 N^2 = {8*L**3*N*N} (ratio {ops["leaf_mul"]/(8*L**3*N*N):.4f}); '
      f'inversions {ops["inv"]} <= N: {ops["inv"] <= N}')
    p(f'time {time.time()-t0:.1f}s')
    return wrong, class_wrong

if __name__ == '__main__':
    outf = sys.argv[1]
    lines = []
    def p(*s):
        line = ' '.join(str(x) for x in s)
        lines.append(line); print(line, flush=True)
        open(outf, 'w').write('\n'.join(lines) + '\n')
    for case in sys.argv[2:]:
        N, m, Fn, seed, relab, schedule, cc = case.split(',')
        run(int(N), int(m), int(Fn), int(seed), relab, schedule, int(cc), p)
