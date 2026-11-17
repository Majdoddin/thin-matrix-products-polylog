r"""Exact operation count of the algorithm of Section 6 (Table 5), per padded entry, for the parameters of
Theorem 1.2: d >= 1, L' = 3d, L = 6d, A = ceil(3d/2), N0 = C(3d, A) (3d)!/(d!)^3, K = 2L'(L'-1). Exact integer
arithmetic (fractions only for the per-entry ratios), d = 1..30, 40, 50, 60 (Lemma 6.6; indep6_counts.py covers
every d <= 150). Also: the fraction f of relabelings that hit |S| = d for a fixed label pair (hypergeometric
convolution, Remark 6.7(iii)), and the relation between L and log2 N0.
Usage: python3 t6_counts.py > ../outputs/t6_counts_out.txt
"""
from math import comb, factorial, log2, log, ceil, pi
from fractions import Fraction


def M(d, k):
    """number of T in {0,1,2}^d with |T| = k (monomials of degree k of V)"""
    return sum((-1) ** j * comb(d, j) * comb(k - 3 * j + d - 1, d - 1) for j in range(0, k // 3 + 1)) if k >= 0 else 0


def Mtab(d):
    return [M(d, k) for k in range(2 * d + 1)]


def Pprod(d, k):
    """products when an element of degree k (all M(d,k) monomials) is multiplied by a linear form in V:
    sum over T with |T| = k of #{i : T_i <= 1}"""
    # count pairs (T, i) with |T| = k, T_i <= 1: = d * (#T' in {0,1,2}^{d-1} with |T'| = k) + d * (#|T'| = k-1)
    if d == 1:
        return 1 if k in (0, 1) else 0
    return d * (M(d - 1, k) + M(d - 1, k - 1))


def counts(d, B=1):
    Lp = 3 * d; L = 6 * d; A = -(-3 * d // 2)
    N0 = comb(Lp, A) * factorial(Lp) // factorial(d) ** 3
    K = 2 * Lp * (Lp - 1)
    Mk = Mtab(d)
    c = {}
    # step 0: powers t_l^{i-1}, i = 3..d
    c['0 powers'] = (L * max(d - 2, 0), 0, 0)
    # step 0: table a(e), 1 <= |supp e| <= d, as a(e - delta_max) * phi_max
    tm = ta = 0
    for j in range(1, d + 1):
        for i in range(0, j + 1):
            k = j + i                       # degree of a(e)
            cnt = comb(L, j) * comb(j, i)
            pp = Pprod(d, k - 1)
            tm += cnt * pp; ta += cnt * (pp - Mk[k])
    c['0 table a(e)'] = (tm, ta, 0)
    # step 0: c(S)^{-1} = det(Phi_S)^{-2} for all d-sets (none needed for d = 1)
    c['0 divisors'] = ((comb(L, d) * comb(d, 2), comb(L, d) * comb(d, 2), comb(L, d)) if d >= 2 else (0, 0, 0))
    # step 1: reading, per label: e on a set of size A + d with |supp| <= d
    Rm = sum(comb(A + d, j) * comb(j, i) * Mk[j + i] for j in range(d + 1) for i in range(j + 1))
    Ra = sum(comb(A + d, j) * comb(j, i) * (Mk[j + i] - 1) for j in range(d + 1) for i in range(j + 1))
    c['1 reading'] = ((1 + K) * B * N0 * Rm, (1 + K) * B * N0 * Ra, 0)
    enc = 11 * 4 ** Lp * (6 ** Lp - 5 ** Lp) + 9 * 6 ** Lp * (6 ** Lp - 4 ** Lp) // 2
    c['2 encoding'] = (0, (1 + K) * B * enc, 0)
    c['3 leaf products'] = (K * B * B * 6 ** L, 0, 0)
    c['4 decoding'] = (0, K * B * B * 20 * L * 6 ** (L - 1), 0)
    c['5 read-out'] = ((B * N0) ** 2, 0, 0)      # bound: N^2 <= N'^2 = (B N0)^2
    return dict(d=d, L=L, Lp=Lp, A=A, N0=N0, K=K, Rm=Rm, Ra=Ra, enc=enc, c=c)


def f_hit(d):
    """Pr over uniform (pi1, pi2) that |r1 cap pi1 c1| + |r2 cap pi2 c2| = d, |r1| = A, |c1| = d, |r2| = d, |c2| = A,
    on L' = 3d levels each."""
    Lp = 3 * d; A = -(-3 * d // 2)
    def hyp(N, K, n, k): return Fraction(comb(K, k) * comb(N - K, n - k), comb(N, n))
    p1 = [hyp(Lp, A, d, k) for k in range(d + 1)]
    return sum(p1[k] * hyp(Lp, d, A, d - k) for k in range(d + 1))


if __name__ == '__main__':
    print('Exact count of the T6 algorithm, per padded entry at B = 1 (the worst case; every B-dependent term per')
    print('entry decreases in B). Columns: d, L, N0, log2 N0, L/log2 N0, K, total/N0^2, total/(N0^2 L^6),')
    print('essential mults K 6^L/N0^2 and its ratio to L^5, share of decoding.')
    rows = []
    for d in list(range(1, 31)) + [40, 50, 60]:
        r = counts(d)
        tot = sum(sum(v) for v in r['c'].values())
        N0 = r['N0']; L = r['L']
        pe = Fraction(tot, N0 * N0)
        ess = Fraction(r['K'] * 6 ** L, N0 * N0)
        dec = Fraction(r['c']['4 decoding'][1], tot)
        print('d=%2d L=%3d N0=%.4g log2N0=%.2f L/log2N0=%.4f K=%d  per-entry=%.6g  /L^6=%.4f  ess=%.6g /L^5=%.4f  dec=%.3f'
              % (d, L, N0, log2(N0), L / log2(N0), r['K'], float(pe), float(pe) / L ** 6, float(ess), float(ess) / L ** 5, float(dec)))
        rows.append((d, float(pe) / L ** 6))
    print()
    print('Per-step breakdown (mults, adds, inversions) at d = 1, 2, 3, B = 1:')
    for d in (1, 2, 3):
        r = counts(d)
        print(' d=%d L=%d N0=%d K=%d Rm=%d Ra=%d enc=%d' % (d, r['L'], r['N0'], r['K'], r['Rm'], r['Ra'], r['enc']))
        for k, v in r['c'].items():
            print('   %-18s %s' % (k, v))
    print()
    print('Asymptotics: 6^L/N0^2 * 108/(pi^3 L^3) -> 1:')
    for d in (5, 10, 20, 40, 80, 160):
        r = counts(d); L = r['L']
        print('  d=%d  6^L/N0^2 = %.6g, ratio = %.5f' % (d, 6 ** L / r['N0'] ** 2, (6 ** L / r['N0'] ** 2) * 108 / (pi ** 3 * L ** 3)))
    print()
    print('Fraction f of relabelings in S_L\' x S_L\' that hit |S| = d (fixed label pair), and f * sqrt(L), 1/K:')
    for d in (1, 2, 3, 4, 6, 8, 12, 16, 24, 32):
        f = f_hit(d); L = 6 * d
        print('  d=%2d L=%3d f=%.5f f*sqrt(L)=%.4f 1/K=%.5f' % (d, L, float(f), float(f) * L ** 0.5, 1 / (2 * 3 * d * (3 * d - 1)) if d > 0 else 0))
    print()
    print('alpha6 = ln3/(3 ln6) = %.6f; check 3^d = (6^{3d})^alpha6: d=7: %.6g vs %.6g' % (log(3) / (3 * log(6)), 3 ** 7, (6 ** 21) ** (log(3) / (3 * log(6)))))
