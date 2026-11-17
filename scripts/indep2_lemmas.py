r"""Second implementation: brute-force checks of Lemma 2.1 (the decomposition gives T4 + E3, read mod 2), Lemma 2.3
(the recursion computes Mult), Lemma 3.1 (equal-size bands), Proposition 3.2 (the reading) and Lemma 3.3 (the
covering by L(L-1) relabelings, for two factorizations of the cyclic shift).
Usage: python3 indep2_lemmas.py OUTFILE
"""
import sys, itertools
import numpy as np
from math import comb
from indep2_gf import GF
from indep2_core import (gamma_T4_E3, p3_tensor_over_Z, axis_transform, PHI_ROWS, PSI_ROWS,
                          CHIT_ROWS, left_index, right_index, out_index, subsets, popcount)

out = []
def p(*s):
    line = ' '.join(str(x) for x in s)
    out.append(line); print(line, flush=True)

rng = np.random.default_rng(152)

# ---------------- Lemma 2.1 ----------------
p('## Lemma 2.1 (read mod 2): sum of the four terms vs T4 + E3')
P3Z, PHI, PSI, CHI = p3_tensor_over_Z()
g = gamma_T4_E3()
diff = P3Z - g
names_l, names_r, names_o = ['x0', 'x1', 'p0'], ['y0', 'y1', 'q0'], ['z0', 'z1', 'z2', 'w0']
nz = [(names_l[s] + names_r[t] + names_o[z], int(diff[s, t, z])) for s, t, z in zip(*np.nonzero(diff))]
p('over Z, (sum of terms) - (T4 + E3) =', nz)
p('mod 2 equal:', bool(np.all(diff % 2 == 0)))

# ---------------- Lemma 2.3 and Lemma 3.1 ----------------
E = GF(4)
def gamma_power(L):
    G = (g % 2).astype(np.uint8)
    for _ in range(L - 1):
        G = np.kron(g % 2, G).astype(np.uint8)  # new level most significant
    return G

def mult_recursion(a, b, L):
    Pa, _ = axis_transform(a[None, :], L, 3, PHI_ROWS)
    Pb, _ = axis_transform(b[None, :], L, 3, PSI_ROWS)
    prod = E.mul_t[Pa, Pb]
    M, _ = axis_transform(prod, L, 4, CHIT_ROWS)
    return M[0]

def mult_brute(a, b, GmT, L):
    prod = E.mul_t[a[:, None], b[None, :]].reshape(-1)  # (3^L*3^L,)
    bits = np.stack([(prod >> k) & 1 for k in range(E.n)], axis=1).astype(np.float32)
    res_bits = np.rint(GmT @ bits).astype(np.int64) % 2
    return sum((res_bits[:, k] << k) for k in range(E.n)).astype(np.uint8)

def strings_left(L):
    """for every left index: (r, P)"""
    rs, Ps = np.zeros(3 ** L, dtype=np.int64), np.zeros(3 ** L, dtype=np.int64)
    for idx in range(3 ** L):
        x, r, P = idx, 0, 0
        for l in range(L):
            dgt = x % 3; x //= 3
            if dgt >= 1: r |= 1 << l
            if dgt == 2: P |= 1 << l
        rs[idx], Ps[idx] = r, P
    return rs, Ps

p('\n## Lemma 2.3 (recursion = Mult by definition) and Lemma 3.1 (equal-size bands: Mult = subset convolution), GF(16)')
for L in (3, 4, 5):
    G = gamma_power(L)
    GmT = np.ascontiguousarray(G.reshape(-1, 4 ** L).T.astype(np.float32)); del G
    rs, Ps = strings_left(L)  # same parametrisation for right strings (c, Q)
    sizes = np.array([popcount(int(r)) for r in rs])
    tot_w2 = 0; bad2 = 0; tot3 = 0; bad3 = 0; ctrl_tot = 0; ctrl_bad = 0
    for a0 in range(L + 1):
        for b0 in range(L + 1):
            a = np.where(sizes == a0, rng.integers(0, 16, 3 ** L), 0).astype(np.uint8)
            b = np.where(sizes == b0, rng.integers(0, 16, 3 ** L), 0).astype(np.uint8)
            Mr = mult_recursion(a, b, L)
            Mb = mult_brute(a, b, GmT, L)
            tot_w2 += 4 ** L; bad2 += int(np.sum(Mr != Mb))
            for r in range(1 << L):
                if popcount(r) != a0: continue
                for c in range(1 << L):
                    if popcount(c) != b0: continue
                    S = r & c
                    val = 0
                    for P in subsets(S):
                        val ^= int(E.mul_t[a[left_index(r, P, L)], b[right_index(c, S ^ P, L)]])
                    tot3 += 1
                    bad3 += int(Mr[out_index(r, c, L)] != val)
            # control: left array also on rows of size a0+1
            if a0 + 1 <= L and 0 < b0:
                a2 = np.where((sizes == a0) | (sizes == a0 + 1), rng.integers(0, 16, 3 ** L), 0).astype(np.uint8)
                Mr2 = mult_recursion(a2, b, L)
                for r in range(1 << L):
                    if popcount(r) != a0: continue
                    for c in range(1 << L):
                        if popcount(c) != b0: continue
                        S = r & c
                        val = 0
                        for P in subsets(S):
                            val ^= int(E.mul_t[a2[left_index(r, P, L)], b[right_index(c, S ^ P, L)]])
                        ctrl_tot += 1
                        ctrl_bad += int(Mr2[out_index(r, c, L)] != val)
    p(f'L={L}: Lemma 2.3: {bad2} of {tot_w2} outputs differ (all a0, b0); Lemma 3.1: {bad3} of {tot3} '
      f'outputs with |r|=a0, |c|=b0 differ; control (left rows of sizes a0 and a0+1): {ctrl_bad} of {ctrl_tot} differ')

# ---------------- Proposition 3.2 ----------------
p('\n## Proposition 3.2 (reading), GF(16), L = 8 levels, every d-set S, Vandermonde forms, random distinct t')
E = GF(4)

def vmul(f, g, d):
    """product in V = E[x_1..x_d]/(x_i^2), elements as arrays indexed by bitmask"""
    h = np.zeros(1 << d, dtype=np.uint8)
    for T in range(1 << d):
        if f[T] == 0: continue
        for U in range(1 << d):
            if g[U] == 0 or (T & U): continue
            h[T | U] ^= E.mul_t[f[T], g[U]]
    return h

for d in (1, 2, 3, 4):
    L = 8
    t = rng.choice(16, L, replace=False)
    phis = []
    for l in range(L):
        v = np.zeros(1 << d, dtype=np.uint8)
        for i in range(d):
            v[1 << i] = E.pow(int(t[l]), i)
        phis.append(v)
    one = np.zeros(1 << d, dtype=np.uint8); one[0] = 1
    def aP(P):
        f = one.copy()
        for l in range(L):
            if (P >> l) & 1:
                f = vmul(f, phis[l], d)
        return f
    full = (1 << d) - 1
    bad = 0; tot = 0; detbad = 0; zero_det = 0; detform_bad = 0
    for trial in range(3):
        x = rng.integers(0, 16, 1 << d).astype(np.uint8)
        y = rng.integers(0, 16, 1 << d).astype(np.uint8)
        ip = 0
        for T in range(1 << d):
            ip ^= int(E.mul_t[x[T], y[T]])
        for S_t in itertools.combinations(range(L), d):
            S = sum(1 << l for l in S_t)
            lhs = 0
            for P in subsets(S):
                fa = aP(P); fb = aP(S ^ P)
                left = 0
                for T in range(1 << d):
                    left ^= int(E.mul_t[x[T], fa[T]])
                right = 0
                for T in range(1 << d):
                    right ^= int(E.mul_t[y[T], fb[full ^ T]])
                lhs ^= int(E.mul_t[left, right])
            Phi = [[E.pow(int(t[l]), i) for i in range(d)] for l in S_t]
            dt = E.det(Phi)
            dform = 1
            for i1 in range(d):
                for i2 in range(i1 + 1, d):
                    dform = int(E.mul_t[dform, int(t[S_t[i1]]) ^ int(t[S_t[i2]])])
            detform_bad += int(dform != dt)
            zero_det += int(dt == 0)
            tot += 1
            bad += int(lhs != int(E.mul_t[dt, ip]))
            detbad += int(aP(S)[full] != dt)  # top coefficient of a(S) = det
    p(f'd={d}: {bad} of {tot} (S, x, y) wrong; det(Phi_S) = 0 for {zero_det}; '
      f'product formula != Gaussian det for {detform_bad}; top coeff of a(S) != det for {detbad}')

# control: the same identity over Z (characteristic 0) fails; d = 2, small integers
p('control over Z (d = 2, t = 1, 2, 3, x, y random in 0..4): identity holds?')
def zmul(f, g, d):
    h = [0] * (1 << d)
    for T in range(1 << d):
        for U in range(1 << d):
            if T & U: continue
            h[T | U] += f[T] * g[U]
    return h
d = 2; full = 3
tz = [1, 2, 3]
phz = [[0, 1, tz[l], 0] for l in range(3)]
for trial in range(2):
    x = list(rng.integers(0, 5, 4)); y = list(rng.integers(0, 5, 4))
    for S_t in itertools.combinations(range(3), 2):
        S = sum(1 << l for l in S_t)
        lhs = 0
        for P in subsets(S):
            fa = [1, 0, 0, 0]; fb = [1, 0, 0, 0]
            for l in S_t:
                if (P >> l) & 1: fa = zmul(fa, phz[l], d)
                else: fb = zmul(fb, phz[l], d)
            left = sum(x[T] * fa[T] for T in range(4)); right = sum(y[T] * fb[full ^ T] for T in range(4))
            lhs += left * right
        det = tz[S_t[1]] - tz[S_t[0]]
        ip = sum(x[T] * y[T] for T in range(4))
        p(f'  S={S_t}: lhs={lhs}, det*ip={det*ip}, -det*ip={-det*ip}')

# ---------------- Lemma 3.3 ----------------
p('\n## Lemma 3.3 (covering by L(L-1) relabelings)')

def compose(f, g):  # (f o g)[l] = f[g[l]]
    return [f[g[l]] for l in range(len(g))]

def transposition(L, i, j):
    t = list(range(L)); t[i], t[j] = t[j], t[i]; return t

def factorizations(L):
    s = [(l + 1) % L for l in range(L)]
    facs = {'adjacent (j-1 j)': [transposition(L, j - 1, j) for j in range(1, L)],
            'star (0 L-j)': [transposition(L, 0, L - j) for j in range(1, L)]}
    for name, ts in facs.items():
        prod = list(range(L))
        for t in ts:
            prod = compose(prod, t)
        assert prod == s, name
    return s, facs

def path(L, ts, s):
    seq = []
    sk = list(range(L))
    for k in range(L):
        cur = sk
        seq.append(cur)
        for j in range(L - 2):
            cur = compose(cur, ts[j])
            seq.append(cur)
        sk = compose(sk, s)
    return seq

def apply_mask(pi, c):
    m = 0
    for l in range(len(pi)):
        if (c >> l) & 1:
            m |= 1 << pi[l]
    return m

for L in (4, 5, 6, 7, 8, 9, 10, 12):
    s, facs = factorizations(L)
    for name, ts in facs.items():
        seq = path(L, ts, s)
        assert len(seq) == L * (L - 1)
        # consecutive members differ by one transposition on the right
        okstep = True
        for i in range(len(seq) - 1):
            inv = [0] * L
            for l in range(L): inv[seq[i][l]] = l
            dlt = compose(inv, seq[i + 1])
            moved = [l for l in range(L) if dlt[l] != l]
            okstep &= (len(moved) == 2)
        # passes through every s^k
        sk = list(range(L)); through = True
        for k in range(L):
            through &= sk in seq
            sk = compose(sk, s)
        size_pairs = [(L // 2, L // 2)] if L == 12 else [(a0, b0) for a0 in range(1, L) for b0 in range(1, L)]
        masks = {k: [m for m in range(1 << L) if popcount(m) == k] for k in range(L + 1)}
        tot = 0; miss = 0; jump = 0; maxfirst = 0; used = set()
        for (a0, b0) in size_pairs:
            dtarget = (a0 * b0) // L
            R = np.array(masks[a0], dtype=np.int64)
            C = np.array(masks[b0], dtype=np.int64)
            first = np.full((len(R), len(C)), -1, dtype=np.int64)
            prev = None
            for idx, pi in enumerate(seq):
                lut = np.array([apply_mask(pi, 1 << l) for l in range(L)], dtype=np.int64)
                PC = np.zeros_like(C)
                for l in range(L):
                    PC |= np.where((C >> l) & 1, lut[l], 0)
                inter = np.bitwise_count(R[:, None] & PC[None, :]).astype(np.int64)
                if prev is not None:
                    jump += int(np.sum(np.abs(inter - prev) > 1))
                prev = inter
                newly = (first < 0) & (inter == dtarget)
                first[newly] = idx
            tot += first.size
            miss += int(np.sum(first < 0))
            if L % 4 == 0 and a0 == b0 == L // 2:
                maxfirst = int(first.max()); used = set(np.unique(first).tolist())
        extra = f'; a=b=L/2: max first index {maxfirst}, distinct first-covering relabelings {len(used)} of {L*(L-1)}' if L % 4 == 0 else ''
        p(f'L={L} {name}: steps by one transposition: {okstep}; passes all s^k: {through}; '
          f'pairs {tot} (sizes {"a=b=L/2" if L == 12 else "all 1..L-1"}), uncovered at floor(a0 b0/L): {miss}; '
          f'steps changing an intersection by >1: {jump}{extra}')

open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
