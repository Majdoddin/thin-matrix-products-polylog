r"""Second implementation: the cost table of the characteristic-3 algorithm (Section 5), re-derived from the text,
exact integers: the cost per padded entry (maximum 1.61741 L^4 at L = 8 over L = 8d <= 256), the bounds of each part
used in the proof for L > 256, the set-up cost against the bound L 2^(1.01 L), an exhaustive candidate search (not
used in the paper), the size of the hitting set Z (Corollary 5.3(i)), Delta_L < L^3/128, the field size of the proof
of Corollary 1.3 in characteristic 3 for F = F_3, and the expected number of rounds of the Las Vegas variant
(Remark 5.4).
Usage: python3 indep3_count.py  ->  ../outputs/indep3_count_out.txt
"""
import itertools, math, functools, os
from math import comb, log2

out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'indep3_count_out.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()

@functools.lru_cache(maxsize=None)
def M3row(d):
    r = [1]
    for _ in range(d):
        n = [0] * (len(r) + 2)
        for i, c in enumerate(r):
            n[i] += c; n[i + 1] += c; n[i + 2] += c
        r = n
    return r
@functools.lru_cache(maxsize=None)
def altrow(d):  # (1 - x + x^2)^d
    r = [1]
    for _ in range(d):
        n = [0] * (len(r) + 2)
        for i, c in enumerate(r):
            n[i] += c; n[i + 1] -= c; n[i + 2] += c
        r = n
    return r
def M3(d, i):
    r = M3row(d); return r[i] if 0 <= i < len(r) else 0
def np_(d, i):
    return d * (M3(d, i) - M3(d - 1, i)) if d >= 1 else 0
def nminus(d, i):
    r = altrow(d); return (M3(d, i) - (r[i] if i < len(r) else 0)) // 2

say('== closed forms of M3, np, n_- against brute force (d <= 5)')
ok = True
for d in range(1, 6):
    Ts = list(itertools.product(range(3), repeat=d))
    for i in range(0, 2 * d + 1):
        m3 = sum(1 for T in Ts if sum(T) == i)
        npb = sum(1 for T in Ts if sum(T) == i - 1 for j in range(d) if T[j] < 2) if i >= 1 else 0
        nmb = sum(1 for T in Ts if sum(T) == i and sum(1 for x in T if x == 1) % 2 == 1)
        tgt_hit = all(any(T[j] >= 1 for j in range(d)) for T in Ts if sum(T) == i) if i >= 1 else True
        ok &= (m3 == M3(d, i)) and (i == 0 or npb == np_(d, i)) and nmb == nminus(d, i) and tgt_hit
say('all agree:', ok)

def per_z(L, B, with_setup=True):
    d = L // 8; a = L // 2; K = L * (L - 1); N0 = comb(L, a)
    forms = (L - 2) + L * (d - 2) if d >= 2 else 0
    tab_m = sum(comb(L, i) * np_(d, i) for i in range(1, 2 * d + 1))
    tab_a = sum(comb(L, i) * (np_(d, i) - M3(d, i)) for i in range(1, 2 * d + 1))
    neg = sum(comb(L, i) * nminus(d, i) for i in range(0, 2 * d + 1))
    inv = comb(L, 2 * d)
    R = sum(comb(a, i) * M3(d, i) for i in range(0, 2 * d + 1)); S0 = sum(comb(a, i) for i in range(0, 2 * d + 1))
    read = (B + B * K) * N0 * R + (B + B * K) * N0 * (R - S0)
    encs = (B + B * K) * 2 * (4 ** L - 3 ** L)
    leaf = B * B * K * 4 ** L
    dec = B * B * K * 3 * L * 4 ** (L - 1)
    setup = forms + tab_m + tab_a + neg + inv
    parts = dict(setup=setup, read=read, encs=encs, leaf=leaf, dec=dec)
    tot = (setup if with_setup else 0) + read + encs + leaf + dec
    return tot, parts, N0

say('== per padded entry and per form choice at B = 1 (division of 1 per entry added once), L = 8d')
worst = (0, None)
for d in range(1, 33):
    L = 8 * d
    tot, parts, N0 = per_z(L, 1)
    r = (tot + N0 * N0) / (N0 * N0) / L ** 4
    if r > worst[0]: worst = (r, L)
    if L in (8, 16, 24, 32, 64, 128, 256):
        say(f'L={L}: per entry per z (incl. division) = {r:.5f} L^4; leaf+decoding share {(parts["leaf"]+parts["dec"])/tot:.4f}; '
            f'setup/N0^2 = {parts["setup"]/N0**2:.3e}; reading/N0^2 = {parts["read"]/N0**2:.3e}')
say(f'max over L = 8..256: {worst[0]:.5f} L^4 at L = {worst[1]} (paper: 1.6175 at L = 8)')
tot2, _, N0_8 = per_z(8, 2)
say(f'L=8, B=2 (N = 140): per entry {(tot2 + (2*N0_8)**2)/(2*N0_8)**2:.2f} (char3_impl.py: 5934.4 at L = 8, schedule all)')

say('== L > 256: exact per-entry ratio, and the bounds of each part used in the proof')
okb = True; maxr = 0
for d in range(33, 129):
    L = 8 * d
    tot, p, N0 = per_z(L, 1)
    N2 = N0 * N0; K = L * (L - 1)
    b_leafdec = 2 * (p['leaf'] + p['dec']) <= (3 * L ** 4 + L ** 3) * N2
    b_enc = p['encs'] <= 4 * L * (L * L - L + 1) * N2
    b_read = p['read'] < N2
    b_set = p['setup'] < N2 and log2(p['setup']) <= log2(L) + 1.01 * L
    b_4L = 4 ** L <= 2 * L * N2
    r = ((tot + N2) * 10 ** 6 // (N2 * L ** 4)) / 10 ** 6; maxr = max(maxr, r)
    okb &= b_leafdec and b_enc and b_read and b_set and b_4L and (tot + N2) <= 2 * L ** 4 * N2
say(f'L = 264..1024: all part bounds hold and per entry <= 2 L^4: {okb}; max ratio {maxr:.5f}')

say('== set-up size: exact (table + signed table + forms + inversions) against C(L, <= 2d), and log2(setup)/L')
for d in (1, 2, 4, 8, 16, 32, 64, 128, 256):
    L = 8 * d
    _, p, N0 = per_z(L, 1)
    s = p['setup']; cle = sum(comb(L, i) for i in range(0, 2 * d + 1))
    say(f'L={L}: log2 setup = {log2(s):.2f}, setup / C(L,<=2d) = {2 ** (log2(s) - log2(cle)):.4e}, log2(setup)/L = {log2(s)/L:.4f}, '
        f'log2(C(L,2d))/L = {log2(comb(L, 2*d))/L:.4f}, bound of the proof log2(L 2^(1.01L))/L = {(log2(L)+1.01*L)/L:.4f}')

say('== exhaustive candidate search (not used in the paper): (C(L,L/4) Delta_L + 1) candidates x one set-up each, against L^8 2^(1.82 L)')
for d in (2, 8, 32, 128, 512):
    L = 8 * d
    _, p, _ = per_z(L, 1)
    DeltaL = (L - 1) * (d * d // 2)
    cand = comb(L, 2 * d) * DeltaL + 1
    exact = cand * (p['setup'] + comb(L, 2 * d))
    textb = cand * (3 * d * 3 ** d * sum(comb(L, i) for i in range(2 * d + 1)) + L * L)
    say(f'L={L}: log2(exact)/L = {log2(exact)/L:.4f}, log2(bound product)/L = {log2(textb)/L:.4f}, '
        f'log2(L^8 2^(1.82L))/L = {(8*log2(L) + 1.82*L)/L:.4f}; exact <= claim: {log2(exact) <= 8*log2(L) + 1.82*L}')
H = -(0.25 * log2(0.25) + 0.75 * log2(0.75)); b = log2(3) / 8
say(f'H(1/4) = {H:.6f}, beta = {b:.6f}, H + beta = {H+b:.6f}, 2H + beta = {2*H+b:.6f}')
# where the analytic chain (3/8) L 2^{(H+b)L} x (L^3/128) 2^{HL} exceeds L^8 2^{1.82 L}
Lx = None
for L in range(8, 10 ** 7, 8):
    if math.log2(3 / 1024) + 4 * math.log2(L) + (2 * H + b) * L > 8 * math.log2(L) + 1.82 * L:
        Lx = L; break
say(f'the analytic chain ((3/8) L 2^((H+beta)L) per set-up, (L^3/128) 2^(HL) candidates) exceeds '
    f'L^8 2^(1.82L) from L = {Lx}')

say('== Theorem 1.1(c): K_z bound, field sizes; Remark 5.4 (Las Vegas): E[J*] bound')
okk = True
for d in range(1, 300):
    L = 8 * d; C = comb(L, 2 * d); DeltaL = (L - 1) * (d * d // 2)
    Kz = math.floor(log2(C)) + 1
    okk &= (log2(C) <= H * L) and (d == 1 or Kz <= L) and DeltaL < L ** 3 / 128
    lc = log2(C); EJ = sum(1.0 if J <= lc else 2.0 ** (lc - J) for J in range(0, 4 * L + 50))
    okk &= EJ <= log2(C) + 2 + 1e-9
    # field of Corollary 1.3 for F = F_3: least k with 3^k > 2 Delta_L
    if d >= 2:
        kk = 1
        while 3 ** kk <= 2 * DeltaL: kk += 1
        okk &= 3 ** kk < 3 * (2 * DeltaL + 1)
say('d = 1..299: log2 C(L,L/4) <= H(1/4) L, K_z <= L (d >= 2), Delta_L < L^3/128, E[J*] <= log2 C + 2, |E| < 3(2 Delta_L + 1):', okk)
say('d = 1: K_z bound floor(log2 C(8,2)) + 1 =', math.floor(log2(comb(8, 2))) + 1, '(greedy pool Z0 of 2 Delta_L = 0 elements is empty)')
out.close()
