r"""Second implementation: Lemma 5.2 (unique minimiser / maximiser), the degree bound Delta_L, Corollary 5.3.
c_S(Z) is computed as an INTEGER polynomial (counts of assignments per weight) by a DP over levels with state
T in {0,1,2}^d; cross-checked by direct enumeration of the 2-regular assignments on samples.
Output: ../outputs/indep3_isolation_out.txt.
"""
import itertools, math, os, sys, time
import numpy as np
from indep3_gf import GF

out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'indep3_isolation_out.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()

def cS_dp(S, d, L):
    maxdeg = sum(S) * (d - 1)
    nst = 3 ** d
    A = np.zeros((nst, maxdeg + 1), dtype=np.int64); A[0, 0] = 1
    src = []; 
    for i in range(d):
        s_ = [t for t in range(nst) if (t // 3 ** i) % 3 < 2]
        src.append((np.array(s_), np.array(s_) + 3 ** i))
    for l in S:
        B = np.zeros_like(A)
        for i in range(d):
            sh = l * i  # weight l*(i-1) with i 1-based = l*i here 0-based
            s_, t_ = src[i]
            if sh == 0:
                B[t_, :] += A[s_, :]
            else:
                B[t_, sh:] += A[s_, :maxdeg + 1 - sh]
        A = B
    return A[nst - 1]

def cS_enum(S, d):
    cnt = {}
    for f in itertools.product(range(d), repeat=len(S)):
        if all(f.count(i) == 2 for i in range(d)):
            w = sum(l * f[j] for j, l in enumerate(S))
            cnt[w] = cnt.get(w, 0) + 1
    return cnt

def Wmin_pred(S, d):  # f*(s_{2m-1}) = f*(s_{2m}) = d+1-m (1-based); weight l*(f-1)
    return sum(l * (d - math.ceil((j + 1) / 2)) for j, l in enumerate(S))
def Wmax_pred(S, d):
    return sum(l * (math.ceil((j + 1) / 2) - 1) for j, l in enumerate(S))

say('== degree bound: sum_{c_j > 0} c_j = floor(d^2/2), c_j = 2 ceil(j/2) - d - 1, sum c_j = 0')
ok = True
for d in range(1, 101):
    c = [2 * math.ceil(j / 2) - d - 1 for j in range(1, 2 * d + 1)]
    ok &= (sum(c) == 0) and (sum(x for x in c if x > 0) == d * d // 2)
    # sign pattern: non-positive then non-negative
    ok &= all(c[j] <= c[j + 1] for j in range(2 * d - 1))
say('d = 1..100:', ok)

say('== Lemma 5.2: on every 2d-set S of {0..L-1}: lowest exponent = predicted W_min with coefficient 1, highest =',
    'predicted W_max with coefficient 1, Delta(S) = sum c_j s_j <= Delta_L = (L-1) floor(d^2/2); DP cross-checked')
polys = {}
for (L, d) in [(8, 1), (6, 2), (10, 2), (12, 2), (16, 2), (9, 3), (12, 3), (14, 3), (10, 4), (12, 4), (24, 3)]:
    t0 = time.time()
    DeltaL = (L - 1) * (d * d // 2)
    bad = 0; maxD = 0; nsets = 0; nenum = 0; enum_bad = 0; zero_mod3 = 0
    store = (L, d) in [(16, 2), (24, 3)]
    P = {}
    rng = np.random.default_rng(L * 100 + d)
    for S in itertools.combinations(range(L), 2 * d):
        p = cS_dp(S, d, L)
        nz = np.nonzero(p)[0]
        lo, hi = nz[0], nz[-1]
        c = [2 * math.ceil(j / 2) - d - 1 for j in range(1, 2 * d + 1)]
        DS = sum(cj * s for cj, s in zip(c, S))
        okS = (lo == Wmin_pred(S, d) and p[lo] == 1 and hi == Wmax_pred(S, d) and p[hi] == 1
               and hi - lo == DS and DS <= DeltaL)
        bad += (not okS); maxD = max(maxD, DS); nsets += 1
        zero_mod3 += (not np.any(p % 3))
        if d <= 3 and (nsets <= 40 or rng.random() < 0.01):
            cnt = cS_enum(S, d); nenum += 1
            q = np.zeros_like(p)
            for w, v in cnt.items(): q[w] = v
            enum_bad += (not np.array_equal(p, q))
        if store:
            P[S] = (int(lo), (p[lo:hi + 1] % 3).astype(np.int64))
    if store: polys[(L, d)] = P
    say(f'(L,d)=({L},{d}): sets {nsets}, sets violating Lemma 5.2 or the bound {bad}, c_S = 0 mod 3 on {zero_mod3}, '
        f'max Delta(S) {maxD} vs Delta_L {DeltaL}, DP vs enumeration: {nenum} sets compared, {enum_bad} differ ({time.time()-t0:.1f}s)')

say('== Corollary 5.3 at the tile shapes L = 8d (theorem shape): root counts, greedy over Z0 (first 2 Delta_L nonzero',
    'elements, two orders), explicit list of Delta_L + 1 values')

def eval_all(F, P, zs):
    """for each set, a boolean vector over zs: g_S(z) != 0. Uses digit-vector linear algebra over F_3."""
    sets = list(P.keys())
    maxdeg = max(len(v[1]) for v in P.values())
    C = np.zeros((len(sets), maxdeg), dtype=np.float64)
    for n, S in enumerate(sets):
        g = P[S][1]; C[n, :len(g)] = g
    k = F.k
    good = np.zeros((len(sets), len(zs)), dtype=bool)
    for zi, z in enumerate(zs):
        pw = [1]
        for e in range(1, maxdeg): pw.append(int(F.MUL[pw[-1], z]))
        V = np.array([F.digits[x] for x in pw], dtype=np.float64)  # maxdeg x k
        R = np.rint(C @ V).astype(np.int64) % 3
        good[:, zi] = np.any(R != 0, axis=1)
    return sets, good

for (L, d, k) in [(16, 2, 4), (24, 3, 5)]:
    t0 = time.time()
    F = GF(k); P = polys[(L, d)]; DeltaL = (L - 1) * (d * d // 2)
    zs_all = list(range(1, F.q))
    sets, good = eval_all(F, P, zs_all)
    nbad = (~good).sum(axis=1)
    Ds = np.array([len(P[S][1]) - 1 for S in sets])
    say(f'(L,d)=({L},{d}) over GF(3^{k}) (q-1 = {F.q-1}, 2 Delta_L = {2*DeltaL}): sets {len(sets)}; '
        f'max #bad z per set {nbad.max()}, sets with #bad z > Delta(S): {(nbad > Ds).sum()}; '
        f'fraction of bad (S,z) {(~good).mean():.4f}; z good for all sets: {int(good.all(axis=0).sum())} of {F.q-1}')
    bound = math.floor(math.log2(math.comb(L, 2 * d))) + 1
    for order_name, order in [('integer order 1,2,..', list(range(1, F.q))), ('powers g^1,g^2,..', [F.pow(F.g, e) for e in range(1, F.q)])]:
        Z0 = order[:2 * DeltaL]
        cols = [zs_all.index(z) for z in Z0]
        Gz = good[:, cols]
        unc = np.ones(len(sets), dtype=bool); rounds = 0; halving_ok = True; chosen = []
        while unc.any():
            gains = Gz[unc].sum(axis=0)
            j = int(np.argmax(gains)); before = unc.sum()
            unc &= ~Gz[:, j]; rounds += 1; chosen.append(Z0[j])
            halving_ok &= (unc.sum() <= before // 2)
            if rounds > 100: break
        say(f'  greedy from Z0 ({order_name}): rounds {rounds} (bound floor(log2 C(L,2d)) + 1 = {bound}), '
            f'every round at least halves: {halving_ok}')
        Zx = order[:DeltaL + 1]
        cols = [zs_all.index(z) for z in Zx]
        say(f'  explicit list of the first Delta_L + 1 = {DeltaL+1} values ({order_name}) hits every set: {bool(good[:, cols].any(axis=1).all())}')
    # adversarial explicit list: for the set with most bad z, take all its bad z plus others up to Delta_L + 1
    worst = int(np.argmax(nbad)); badz = [zs_all[i] for i in np.nonzero(~good[worst])[0]]
    rest = [z for z in zs_all if z not in badz][:DeltaL + 1 - len(badz)]
    cols = [zs_all.index(z) for z in badz + rest]
    say(f'  adversarial explicit list (all {len(badz)} bad z of the worst set + others, {len(cols)} values) hits every set: '
        f'{bool(good[:, cols].any(axis=1).all())} ({time.time()-t0:.1f}s)')
out.close()
