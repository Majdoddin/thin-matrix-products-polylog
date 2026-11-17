r"""Second implementation of the characteristic-3 algorithm of Section 5, end to end, written from the text only.
Levels 0..L-1; rows labelled by all a-subsets, columns by all b-subsets (one band each side), floor(ab/L) = 2d.
Forms phi_l = (1, z^l, ..., z^{(d-1)l}); table a_z(P) for |P| <= 2d in V = E[x]/(x_i^3); left values <X_r, a_z(P)>,
right values sum_T Y[T,c] sigma(T) a_z(Q)_{2-T}; signed decomposition (Lemma 2.1), encoding/decoding axis by axis;
Pi_L = s^k t_1..t_j (Lemma 3.3); Z by the greedy of Corollary 5.3(i) over Z0 = first 2 Delta_L nonzero elements
(integer order), or given.
Checks every class output (all pairs with |r cap pi(c)| = 2d, good or bad S) of every executed tile-run, and the
assembled product via the z-major covering table. Controls: unsigned right pairing, plus-sign decomposition, no
division.
usage: python3 indep3_impl.py L a b d k inputs(F3|E) schedule(all|first) control(none|unsigned|plus|nodiv) [z list]
       ->  ../outputs/indep3_impl_<tag>_out.txt
"""
import itertools, math, os, sys, time
import numpy as np
from indep3_gf import GF

L, a, b, d, k = map(int, sys.argv[1:6]); inputs, schedule, control = sys.argv[6:9]
zgiven = [int(x) for x in sys.argv[9].split(',')] if len(sys.argv) > 9 else None
assert (a * b) // L == 2 * d
tag = f'L{L}a{a}b{b}d{d}k{k}{inputs}{schedule}{control}' + ('z' + '_'.join(map(str, zgiven)) if zgiven else '')
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', f'indep3_impl_{tag}_out.txt'), 'w')
def say(*s_):
    s = ' '.join(str(x) for x in s_); print(s); out.write(s + '\n'); out.flush()
t0 = time.time()
F = GF(k)
for name in ('ADD', 'SUB', 'NEG', 'MUL'):
    setattr(F, name, getattr(F, name).astype(np.int16))
D = 3 ** d; DeltaL = (L - 1) * (d * d // 2)
rng = np.random.default_rng(12345 + L)
rows = list(itertools.combinations(range(L), a)); cols = list(itertools.combinations(range(L), b))
Nr, Nc = len(rows), len(cols)
hi = F.q if inputs == 'E' else 3
X = rng.integers(0, hi, size=(Nr, D)).astype(np.int16); Y = rng.integers(0, hi, size=(D, Nc)).astype(np.int16)
XY = np.zeros((Nr, Nc), dtype=np.int16)
for t in range(D):
    XY = F.ADD[XY, F.MUL[X[:, t][:, None], Y[t, :][None, :]]]
say(f'L={L} a={a} b={b} d={d} D={D} E=GF(3^{k}) Nr x Nc = {Nr} x {Nc} inputs={inputs} schedule={schedule} control={control} Delta_L={DeltaL}')

Tof = [tuple((t // 3 ** i) % 3 for i in range(d)) for t in range(D)]
sig = np.array([(-1) ** sum(1 for x in T if x == 1) for T in Tof])
comp = np.array([sum((2 - T[i]) * 3 ** i for i in range(d)) for T in Tof])
allP = [P for i in range(2 * d + 1) for P in itertools.combinations(range(L), i)]
Pidx = {P: n for n, P in enumerate(allP)}

def table(z):
    phis = [[F.pow(z, l * i) for i in range(d)] for l in range(L)]
    A = np.zeros((len(allP), D), dtype=np.int16)
    for n, P in enumerate(allP):
        if not P:
            A[n, 0] = 1; continue
        prev = A[Pidx[P[:-1]]]; phi = phis[P[-1]]; v = np.zeros(D, dtype=np.int16)
        for t in range(D):
            c = int(prev[t])
            if not c: continue
            for i in range(d):
                if Tof[t][i] < 2:
                    j = t + 3 ** i; v[j] = F.ADD[v[j], F.MUL[c, phi[i]]]
        A[n] = v
    cS = {P: int(A[Pidx[P], D - 1]) for P in allP if len(P) == 2 * d}
    return A, cS

def enc(Pset, base):
    return sum(base ** (L - 1 - l) for l in Pset)

# relabelings Pi_L: s(l) = l+1 mod L, s = t_1 o ... o t_{L-1}
s = [(l + 1) % L for l in range(L)]
def compose(p, q):  # (p o q)(x) = p(q(x))
    return [p[q[x]] for x in range(L)]
ts = []
for order in ([(j - 1, j) for j in range(1, L)], [(j - 1, j) for j in range(L - 1, 0, -1)]):
    prod = list(range(L))
    for (u, v) in order:
        t = list(range(L)); t[u], t[v] = v, u; prod = compose(prod, t)
    if prod == s:
        ts = order; break
assert ts, 'decomposition of s not found'
Pi = []
sk = list(range(L))
for kk in range(L):
    cur = sk
    Pi.append(cur)
    for (u, v) in ts[:-1]:
        t = list(range(L)); t[u], t[v] = v, u; cur = compose(cur, t); Pi.append(cur)
    sk = compose(s, sk)
assert len(Pi) == L * (L - 1) and len({tuple(p) for p in Pi}) == L * (L - 1)
assert all(tuple(compose(s if False else list(range(L)), p)) for p in Pi)
for kk in range(L):  # passes through every s^k
    assert [ (l + kk) % L for l in range(L)] in Pi

rmask = np.array([sum(1 << l for l in r) for r in rows]); 
def popcount(x):
    return np.array([bin(int(v)).count('1') for v in x.ravel()]).reshape(x.shape)
cperm = []  # for each pi: column label sets pi(c_mu) as tuples
for p in Pi:
    cperm.append([tuple(sorted(p[l] for l in c)) for c in cols])
inter = np.zeros((len(Pi), Nr, Nc), dtype=np.int8)
for n, cp in enumerate(cperm):
    cm = np.array([sum(1 << l for l in c) for c in cp])
    inter[n] = popcount(rmask[:, None] & cm[None, :])
covered = (inter == 2 * d).any(axis=0)
say(f'Covering check (Lemma 3.3): every (row, column) pair meets in exactly 2d = {2*d} levels under some pi in Pi_L: {bool(covered.all())}')

# choose Z
t1 = time.time()
tables = {}
if zgiven is not None:
    Z = zgiven
    for z in Z: tables[z] = table(z)
else:
    Z0 = list(range(1, F.q))[:max(1, 2 * DeltaL)]
    sets2d = [P for P in allP if len(P) == 2 * d]
    good = np.zeros((len(sets2d), len(Z0)), dtype=bool)
    for j, z in enumerate(Z0):
        A_, cS_ = table(z); good[:, j] = [cS_[S] != 0 for S in sets2d]
        tables[z] = (A_, cS_)
    unc = np.ones(len(sets2d), dtype=bool); Z = []
    while unc.any():
        j = int(np.argmax(good[unc].sum(axis=0))); Z.append(Z0[j]); unc &= ~good[:, j]
    tables = {z: tables[z] for z in Z}
say(f'Z = {Z} (|Z0| = {max(1, 2*DeltaL)}; greedy rounds bound {math.floor(math.log2(math.comb(L, 2*d))) + 1}); '
    f'bad sets per z: {[sum(1 for v in tables[z][1].values() if v == 0) for z in Z]} of {math.comb(L, 2*d)} ({time.time()-t1:.1f}s)')

sign = 1 if control == 'plus' else -1
def axis_apply(arr, ax, rowsfun):
    A_ = np.moveaxis(arr, ax, 0)
    return np.moveaxis(np.stack(rowsfun(A_)), 0, ax)
def phi_rows(A_):
    f = F.ADD if sign == 1 else F.SUB
    return [f[A_[0], A_[1]], f[A_[0], A_[2]], A_[1], A_[0]]
def psi_rows(B_):
    f = F.ADD if sign == 1 else F.SUB
    return [B_[0], B_[1], f[B_[0], B_[2]], F.ADD[B_[0], B_[1]]]
def chi_rows(C_):
    f = F.ADD if sign == 1 else F.SUB
    w = f[f[f[C_[3], C_[0]], C_[1]], C_[2]] if sign == -1 else F.ADD[F.ADD[F.ADD[C_[3], C_[0]], C_[1]], C_[2]]
    return [C_[0], C_[1], C_[2], w]
def encode(arr, rf):
    for ax in range(L):
        arr = axis_apply(arr, ax, rf)
    return arr

# covering table, z-major: for each pair the first (z, pi) with |S| = 2d and c_z(S) != 0
first = -np.ones((Nr, Nc), dtype=np.int64)
for zi, z in enumerate(Z):
    cS = tables[z][1]
    for n, cp in enumerate(cperm):
        m = (inter[n] == 2 * d) & (first < 0)
        if not m.any(): continue
        nu, mu = np.nonzero(m)
        okc = np.array([cS[tuple(sorted(set(rows[x]) & set(cp[y])))] != 0 for x, y in zip(nu, mu)])
        first[nu[okc], mu[okc]] = zi * len(Pi) + n
say(f'covering: every entry has some (z, pi) with |S| = 2d and c_z(S) != 0: {bool((first >= 0).all())}')
runs = sorted(set(first.ravel().tolist())) if schedule == 'first' else [zi * len(Pi) + n for zi in range(len(Z)) for n in range(len(Pi))]
say(f'tile-runs executed: {len(runs)} of {len(Z) * len(Pi)}')

enc3 = lambda P: enc(P, 3); enc4 = lambda P: enc(P, 4)
renc3 = np.array([enc3(r) for r in rows]); renc4 = np.array([enc4(r) for r in rows])
result = np.zeros((Nr, Nc), dtype=np.int16); have = np.zeros((Nr, Nc), dtype=bool)
n_checked = n_wrong = n_bad_checked = n_bad_nonzero = 0
cache_left = {}
for run in runs:
    zi, n = divmod(run, len(Pi)); z = Z[zi]; A_, cS = tables[z]
    if zi not in cache_left:
        # left values <X_nu, a_z(P)> for P subset r_nu, |P| <= 2d
        a3 = np.zeros(3 ** L, dtype=np.int16)
        for nu, r in enumerate(rows):
            Ps = [P for i in range(2 * d + 1) for P in itertools.combinations(r, i)]
            vals = F.dot(np.broadcast_to(X[nu], (len(Ps), D)), A_[[Pidx[P] for P in Ps]])
            a3[[renc3[nu] + enc3(P) for P in Ps]] = vals
        cache_left = {zi: encode(a3.reshape((3,) * L), phi_rows)}
    Phi = cache_left[zi]
    cp = cperm[n]
    b3 = np.zeros(3 ** L, dtype=np.int16)
    Arev = A_[:, comp]  # a(Q)_{2-T}
    sgn = sig if control != 'unsigned' else np.ones_like(sig)
    for mu, c in enumerate(cp):
        Qs = [Q for i in range(2 * d + 1) for Q in itertools.combinations(c, i)]
        M_ = Arev[[Pidx[Q] for Q in Qs]].copy()
        M_[:, sgn == -1] = F.NEG[M_[:, sgn == -1]]
        vals = F.dot(np.broadcast_to(Y[:, mu], (len(Qs), D)), M_)
        ce = enc3(c)
        b3[[ce + enc3(Q) for Q in Qs]] = vals
    Psi = encode(b3.reshape((3,) * L), psi_rows)
    Mult = encode(F.MUL[Phi, Psi], chi_rows).ravel()
    # class outputs: all pairs with |S| = 2d
    nu, mu = np.nonzero(inter[n] == 2 * d)
    cenc4 = np.array([enc4(c) for c in cp])
    vals = Mult[2 * renc4[nu] + cenc4[mu]]
    cs = np.array([cS[tuple(sorted(set(rows[x]) & set(cp[y])))] for x, y in zip(nu, mu)])
    expect = F.MUL[cs, XY[nu, mu]]
    n_checked += len(nu); n_wrong += int((vals != expect).sum())
    n_bad_checked += int((cs == 0).sum()); n_bad_nonzero += int(((cs == 0) & (vals != 0)).sum())
    sel = first[nu, mu] == run
    if control == 'nodiv':
        result[nu[sel], mu[sel]] = vals[sel]
    else:
        result[nu[sel], mu[sel]] = F.MUL[F.INV[cs[sel]].astype(np.int16), vals[sel]]
    have[nu[sel], mu[sel]] = True
say(f'class outputs checked {n_checked}, wrong {n_wrong}; of them at bad S (c_z(S) = 0): {n_bad_checked}, raw value != 0: {n_bad_nonzero}')
say(f'assembled product: entries {Nr*Nc}, computed {int(have.sum())}, wrong {int(((result != XY) | ~have).sum())}')
say(f'time {time.time()-t0:.1f}s')
out.close()
