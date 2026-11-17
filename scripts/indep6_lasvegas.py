r"""Remark 6.7(iii): the number of uniform random relabelings that cover all N0^2 label pairs with probability >= 1/2,
K_r = ceil(ln(2 N0^2)/f), with the EXACT hitting fraction f (hypergeometric convolution, the same for every label
pair since it depends only on |r1| = A, |c1| = d, |r2| = d, |c2| = A), against K = L(L-2)/2 of the diagonal path and
against K ln(2 N0^2) (the bound from f >= 1/K).
Usage: python3 indep6_lasvegas.py > ../outputs/indep6_lasvegas_out.txt
"""
from math import comb, factorial, log, ceil, sqrt
from fractions import Fraction
def f_hit(d):
    Lp = 3 * d; A = (3 * d + 1) // 2
    h = lambda N, K, n, k: Fraction(comb(K, k) * comb(N - K, n - k), comb(N, n)) if 0 <= k <= min(K, n) else Fraction(0)
    return sum(h(Lp, A, d, k) * h(Lp, d, A, d - k) for k in range(d + 1))
print(' d    L       K   f        f*sqrt(L)  K_r(exact f)  K_r/K    K ln(2N0^2)')
for d in (1, 2, 3, 4, 6, 8, 12, 16, 24, 31, 32, 48, 64):
    Lp = 3 * d; L = 6 * d; A = (3 * d + 1) // 2
    N0 = comb(Lp, A) * factorial(Lp) // factorial(d) ** 3
    K = L * (L - 2) // 2; f = float(f_hit(d)); lnN = log(2) + 2 * log(N0)
    Kr = ceil(lnN / f)
    print('%3d %4d %7d  %.5f  %.4f    %8d      %.4f   %d' % (d, L, K, f, f * sqrt(L), Kr, Kr / K, ceil(K * lnN)))
