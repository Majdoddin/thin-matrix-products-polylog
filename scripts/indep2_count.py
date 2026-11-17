r"""Second implementation: cost per entry of the algorithm of Section 4, recomputed from the list of its steps
(Table 2 re-derived independently), exact integers via fractions.

Checks: (1) per padded entry at B = 1 <= 2 L^4 for L = 4, 8, ..., 256 (and all even L) (Lemma 4.2);
(2) an analytic tail bound for L > 256; (3) the rank per entry K 4^L / N0^2 < 2 L^3 (Theorem 1.1(a));
(4) N0 >= 2^L / sqrt(2L) (inequality (1)); (5) whether D = 2^floor(L/4) >= N0^(1/4) for even L;
(6) the parameter claims 2^q < 2m, N0 < N, L <= log2 N of Section 4 for m <= 4096.
Usage: python3 indep2_count.py OUTFILE
"""
from fractions import Fraction
from math import comb, isqrt, log2, lgamma, log, exp
import sys

def phase_counts(L, B):
    a = L // 2
    d = L // 4  # floor(a^2/L) = floor(L/4) for even L
    N0 = comb(L, a)
    K = L * (L - 1)
    R = sum(comb(a, i) * comb(d, i) for i in range(d + 1))
    S0 = sum(comb(a, i) for i in range(d + 1))
    assert R == comb(a + d, d)
    c = {}
    # 0(i) table a(P): P of size i built from P minus max (degree i-1, C(d,i-1) coefficients) times a
    # linear form: C(d,i-1)*(d-i+1) products; each of C(d,i) targets gets i contributions.
    c['table_mul'] = sum(comb(L, i) * comb(d, i - 1) * (d - i + 1) for i in range(1, d + 1))
    c['table_add'] = sum(comb(L, i) * comb(d, i) * (i - 1) for i in range(1, d + 1))
    # 0(ii) det(Phi_S) = prod_{l<l'} (t_l + t_l'): C(d,2) sums, C(d,2)-1 products, one inversion
    c['det_add'] = comb(L, d) * comb(d, 2)
    c['det_mul'] = comb(L, d) * max(comb(d, 2) - 1, 0)
    c['inv'] = comb(L, d)
    # 1 reading: per array N0 rows, a string with |P| = i: C(d,i) products, C(d,i)-1 additions
    narr = B + B * K
    c['read_mul'] = narr * N0 * sum(comb(a, i) * comb(d, i) for i in range(d + 1))
    c['read_add'] = narr * N0 * sum(comb(a, i) * (comb(d, i) - 1) for i in range(d + 1))
    # 2 encodings: axis l has 4^l 3^(L-1-l) fibres, 2 additions each
    c['enc_add'] = narr * sum(2 * 4 ** l * 3 ** (L - 1 - l) for l in range(L))
    # 3 leaf products
    c['leaf_mul'] = B * B * K * 4 ** L
    # 4 decoding: 4^(L-1) fibres per axis, 3 additions each
    c['dec_add'] = B * B * K * L * 3 * 4 ** (L - 1)
    # 5 division
    c['div_mul'] = (B * N0) ** 2
    return c, N0, K, d

def per_entry(L, B=1):
    c, N0, K, d = phase_counts(L, B)
    tot = sum(c.values())
    return Fraction(tot, (B * N0) ** 2), c, N0, K, d

out = []
def p(*s):
    out.append(' '.join(str(x) for x in s))

p('# per padded entry at B = 1, ratio to L^4; must be <= 2 for L divisible by 4')
worst = (0, None)
worst_even = (0, None)
fail = []
for L in range(4, 257, 2):
    pe, c, N0, K, d = per_entry(L, 1)
    r = pe / L ** 4
    if L % 4 == 0:
        if r > worst[0]:
            worst = (r, L)
        if r > 2:
            fail.append(L)
    if r > worst_even[0]:
        worst_even = (r, L)
    if L in (4, 8, 12, 16, 32, 64, 128, 256) or L % 4 != 0 and L <= 10:
        p(f'L={L} d={d} N0={N0} K={K} per-entry={float(pe):.4f} ratio/L^4={float(r):.5f}')
p(f'max ratio over L = 4,8,...,256: {float(worst[0]):.5f} at L = {worst[1]}; failures (>2): {fail}')
p(f'max ratio over all even L in [4,256]: {float(worst_even[0]):.5f} at L = {worst_even[1]}')

# B dependence: per entry decreases in B
p('\n# per entry vs B (L = 8, 12): should decrease in B')
for L in (8, 12):
    row = []
    for B in (1, 2, 3, 19, 1000):
        pe, *_ = per_entry(L, B)
        row.append(f'B={B}:{float(pe):.2f}')
    p(f'L={L}', ' '.join(row))

# rank per entry
p('\n# rank per padded entry K 4^L / N0^2 vs 2 L^3 (L divisible by 4)')
bad = []
for L in range(4, 257, 4):
    N0 = comb(L, L // 2); K = L * (L - 1)
    rk = Fraction(K * 4 ** L, N0 ** 2)
    if rk >= 2 * L ** 3:
        bad.append(L)
    if L in (4, 8, 12, 64, 256):
        p(f'L={L} rank/entry={float(rk):.2f} 2L^3={2*L**3} ratio={float(rk/(2*L**3)):.4f}')
p('L with rank/entry >= 2 L^3:', bad)

# N0 >= 2^L/sqrt(2L): check N0^2 * 2L >= 4^L
bad = [L for L in range(2, 513, 2) if comb(L, L // 2) ** 2 * 2 * L < 4 ** L]
p('\n# N0^2 * 2L >= 4^L fails at even L in [2,512]:', bad)

# D = 2^floor(L/4) >= N0^(1/4), i.e. 2^(4 floor(L/4)) >= N0 (it holds for L divisible by 4)
bad = [L for L in range(2, 201, 2) if 2 ** (4 * (L // 4)) < comb(L, L // 2)]
p('# D = 2^floor(L/4) >= N0^(1/4) fails at even L in [2,200]:', bad)
p('#   L=6: D=2, N0=20, N0^(1/4)=%.4f; L=10: D=4, N0^(1/4)=%.4f' % (20 ** 0.25, 252 ** 0.25))

# analytic tail for L > 256 (cf. the proof of Lemma 4.2): bound = 1.5 L^4 + 0.5 L^3 + 4 L (L^2-L+1) + 3 <= 2 L^4 for L >= 10
bad = [L for L in range(10, 100000, 2) if Fraction(3, 2) * L ** 4 + Fraction(1, 2) * L ** 3 + 4 * L * (L * L - L + 1) + 3 > 2 * L ** 4]
p('\n# analytic bound 1.5L^4+0.5L^3+4L(L^2-L+1)+3 <= 2L^4 fails at even L in [10,1e5):', bad[:10])
# the three small terms are each < 1 for L > 256: reading 2 L^2 sqrt(2L) 2^(-L/4); set-up/N0^2; division <= 1
for L in (258, 260, 1000):
    rd = 2 * L * L * (2 * L) ** 0.5 * 2.0 ** (-L / 4)
    c, N0, K, d = phase_counts(L, 1)
    setup = c['table_mul'] + c['table_add'] + c['det_add'] + c['det_mul'] + c['inv']
    p(f'L={L}: reading bound={rd:.3e}, set-up/N0^2={setup / N0**2:.3e} (exact ints)')
# exact check of the full per-entry bound at a few L > 256 (exact integers)
for L in (260, 300, 400, 512):
    pe, c, N0, K, d = per_entry(L, 1)
    p(f'L={L}: exact per-entry/L^4 = {float(pe / L**4):.6f}')

# the hypotheses: N0 < N, N' < 2N, L <= log2 N for N >= (2m)^4, q = ceil(log2 m)
p('\n# hypotheses: for m in 2..2^12 and N = (2m)^4 (smallest allowed N): 2^q < 2m, N0 < N, L <= log2 N')
bad = []
for m in range(2, 4097):
    q = (m - 1).bit_length()  # ceil(log2 m)
    assert 2 ** q >= m and 2 ** (q - 1) < m
    L = 4 * q; N0 = comb(L, L // 2); N = (2 * m) ** 4
    if not (2 ** q < 2 * m and N0 < N and L <= log2(N)):
        bad.append(m)
p('violations:', bad)

open(sys.argv[1], 'w').write('\n'.join(out) + '\n')
print('\n'.join(out))
