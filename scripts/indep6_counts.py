r"""Independent evaluation of Table 5 per padded entry at B = 1 for EVERY d = 1..DMAX (Lemma 6.6: total <= L^6 and
leaf products <= L^5/4; t6_counts.py prints d = 1..30, 40, 50, 60 only), numeric checks of the inequalities of the
proof of Lemma 6.6 for d >= 31, and of the constants of Theorem 1.2 and Remark 6.7(i).
M_k = [x^k] (1+x+x^2)^d, P_k = d [x^k] (1+x)(1+x+x^2)^(d-1), encoding summed axis by axis (not by the closed form),
everything in exact integers / Fractions.
Usage: python3 indep6_counts.py [DMAX] > ../outputs/indep6_counts_out.txt   (DMAX = 150 by default)
"""
from math import comb, factorial, log, log2, pi, exp, sqrt, ceil
from fractions import Fraction
import sys

def polypow(base, n):
    out = [1]
    for _ in range(n):
        new = [0] * (len(out) + len(base) - 1)
        for i, a in enumerate(out):
            for j, b in enumerate(base):
                new[i + j] += a * b
        out = new
    return out

def table(d):
    Lp = 3 * d; L = 6 * d; A = (3 * d + 1) // 2
    N0 = comb(Lp, A) * factorial(Lp) // factorial(d) ** 3
    K = L * (L - 2) // 2
    M = polypow([1, 1, 1], d)                                   # M[k], k = 0..2d
    PP = polypow([1, 1, 1], d - 1); PP = [sum(PP[k - t] for t in (0, 1) if 0 <= k - t < len(PP)) * d for k in range(2 * d + 1)]
    Mk = lambda k: M[k] if 0 <= k < len(M) else 0
    Pk = lambda k: PP[k] if 0 <= k < len(PP) else 0
    tm = sum(comb(L, j) * comb(j, i) * Pk(j + i - 1) for j in range(1, d + 1) for i in range(j + 1))
    ta = sum(comb(L, j) * comb(j, i) * (Pk(j + i - 1) - Mk(j + i)) for j in range(1, d + 1) for i in range(j + 1))
    powers = L * max(d - 2, 0)
    div = (comb(L, d) * (2 * comb(d, 2) + 1)) if d >= 2 else 0   # C(L,d) C(d,2) mults + C(L,d) C(d,2) adds + C(L,d) inversions
    R = sum(comb(A + d, j) * comb(j, i) * Mk(j + i) for j in range(d + 1) for i in range(j + 1))
    Rm = sum(comb(A + d, j) * comb(j, i) * (Mk(j + i) - 1) for j in range(d + 1) for i in range(j + 1))
    # encoding: five-letter axes first (11 additions per fibre), then four-letter axes (9 per fibre)
    enc = sum(11 * 6 ** (k - 1) * 5 ** (Lp - k) * 4 ** Lp for k in range(1, Lp + 1)) + \
          sum(9 * 6 ** (Lp + k - 1) * 4 ** (Lp - k) for k in range(1, Lp + 1))
    leaf = K * 6 ** L
    dec = K * 20 * L * 6 ** (L - 1)
    readout = N0 * N0
    parts = dict(setup=tm + ta + powers + div, reading=(1 + K) * N0 * (R + Rm), encoding=(1 + K) * enc, leaf=leaf,
                 decoding=dec, readout=readout)
    return dict(d=d, L=L, A=A, N0=N0, K=K, parts=parts, R=R)

DMAX = int(sys.argv[1]) if len(sys.argv) > 1 else 150
worst = (0, None); worst2 = (0, None); worst_leaf = (0, None); first97 = None
print('d, L, per-entry total / L^6, leaf products per entry / L^5, decoding share')
for d in range(1, DMAX + 1):
    t = table(d); L = t['L']; N2 = t['N0'] ** 2
    tot = Fraction(sum(t['parts'].values()), N2)
    r6 = tot / L ** 6; r5 = Fraction(t['parts']['leaf'], N2) / L ** 5
    share = Fraction(t['parts']['decoding'], sum(t['parts'].values()))
    if r6 > worst[0]: worst = (r6, d)
    if d >= 2 and r6 > worst2[0]: worst2 = (r6, d)
    if r5 > worst_leaf[0]: worst_leaf = (r5, d)
    if first97 is None and share > Fraction(97, 100): first97 = d
    if d <= 12 or d % 10 == 0 or 31 <= d <= 39:
        print('  d=%3d L=%4d  total/L^6=%.5f  leaf/L^5=%.5f  decoding share=%.5f' % (d, L, float(r6), float(r5), float(share)))
    assert r6 <= 1 and r5 <= Fraction(1, 4), d
print('max total/L^6 over d=1..%d: %.5f at d=%d; over d=2..%d: %.5f at d=%d; max leaf/L^5: %.5f at d=%d'
      % (DMAX, worst[0], worst[1], DMAX, worst2[0], worst2[1], worst_leaf[0], worst_leaf[1]))
print('first d with decoding share > 97%%: %d (share at d=9: %.5f)' % (first97, float(Fraction(table(9)['parts']['decoding'], sum(table(9)['parts'].values())))))
# proof of Lemma 6.6 (d >= 31): inequalities, checked numerically for d = 31..DMAX
bad = []
for d in range(31, DMAX + 1):
    t = table(d); L = t['L']; N0 = t['N0']; A = t['A']
    lnu = L * log(6) - 2 * log(N0)                                  # log of u = 6^L / N0^2 (math.log is exact enough on ints)
    lnb = log(pi ** 3 / 108) + 2 * log(L) + log(L + 2) + 13 / (3 * L)
    if not lnu <= lnb: bad.append(('u-bound', d))
    if not lnb <= log(0.2971) + 3 * log(L): bad.append(('0.2971', d))
    if not 196 * d ** 3 * N0 * N0 >= 6 ** (6 * d): bad.append(('N0>=6^3d/(14 d^1.5) and N0^2>=6^L/(196 d^3)', d))
    if not t['R'] <= 3 ** (A + 2 * d): bad.append(('R<=3^(A+2d)', d))
    n = 3 * d
    if not log(comb(n, A)) >= n * log(2) + 0.5 * log(2 / (pi * (n + 1))) - 1 / (3 * n): bad.append(('binom', d))
    if not log(factorial(n) // factorial(d) ** 3) >= n * log(3) + 0.5 * log(3) - 1 / (4 * d) - log(2 * pi * d): bad.append(('trinom', d))
print('Lemma 6.6 (d >= 31) inequalities for d = 31..%d: failures %s' % (DMAX, bad if bad else 'none'))
# where do the binomial / multinomial bounds already hold (to see whether (ii) could start lower)
lo = [d for d in range(1, 31) if all([
    comb(3 * d, (3 * d + 1) // 2) >= 2 ** (3 * d) * sqrt(2 / (pi * (3 * d + 1))) * exp(-1 / (9 * d)),
    factorial(3 * d) // factorial(d) ** 3 >= 3 ** (3 * d) * sqrt(3) * exp(-1 / (4 * d)) / (2 * pi * d)])]
print('binomial and trinomial lower bounds also hold for d in 1..30:', lo == list(range(1, 31)))
# constants of Theorem 1.2
l6 = log2(6)
print('2/log2(6) = %.5f;  4*64/log2(6)^6 = %.5f;  32/log2(6)^5 = %.5f;  5 pi^3/324 = %.5f;  3^(1/alpha6) = %.3f'
      % (2 / l6, 256 / l6 ** 6, 32 / l6 ** 5, 5 * pi ** 3 / 324, 3 ** (3 * log(6) / log(3))))
