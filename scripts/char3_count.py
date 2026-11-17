r"""The exact operation count of the characteristic-3 algorithm of Section 5 at the shape of Theorem 1.1(b),(c)
(L = 8d, a = b = L/2, B bands per side, K_z form choices, K = L(L-1) relabelings), per padded entry, and the
bound per entry <= K_z * 2 L^4 (+1 for the division), for every L = 8d up to 256 (exact integers). The maximum,
1.61741 L^4 at L = 8, is the constant 1.6175 of the proof of Theorem 1.1(b),(c).
Closed forms (checked against brute force for d <= 5):
  M3(d,i)   = #{T in {0,1,2}^d : |T| = i} = sum_j C(d,j) C(d-j, i-2j)
  npairs(d,i) = #{(T, j): |T| = i-1, T_j < 2} = d (M3(d,i) - M3(d-1,i))   (count by the target T' = T + e_j, T'_j >= 1)
  nminus(d,i) = #{T : |T| = i, #{j: T_j = 1} odd} = (M3(d,i) - M3m(d,i))/2, M3m = coefficient of x^i in (1 - x + x^2)^d
The count is that of char3_impl.py's function `formula` (counted = formula there at L = 8, 10, 12).
Usage: python3 char3_count.py  ->  ../outputs/char3_count_out.txt
"""
import itertools, math, os, sys
from fractions import Fraction

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'char3_count_out.txt')
lines = []


def log(s):
    lines.append(s); print(s, flush=True)


def M3(d, i):
    if i < 0 or d < 0:
        return 0
    return sum(math.comb(d, j) * math.comb(d - j, i - 2 * j) for j in range(0, i // 2 + 1) if i - 2 * j <= d - j)


def M3m(d, i):
    # coefficient of x^i in (1 - x + x^2)^d = sum_j C(d,j) C(d-j, i-2j) (-1)^{i-2j}
    return sum(math.comb(d, j) * math.comb(d - j, i - 2 * j) * (-1) ** (i - 2 * j)
               for j in range(0, i // 2 + 1) if i - 2 * j <= d - j)


def npairs(d, i):
    return d * (M3(d, i) - M3(d - 1, i))


def nminus(d, i):
    return (M3(d, i) - M3m(d, i)) // 2


def brute(d):
    ok = True
    for i in range(0, 2 * d + 1):
        Ts = [T for T in itertools.product(range(3), repeat=d) if sum(T) == i]
        m3 = len(Ts)
        nm = sum(1 for T in Ts if sum(1 for x in T if x == 1) % 2)
        if i >= 1:
            npr = sum(1 for T in itertools.product(range(3), repeat=d) if sum(T) == i - 1 for j in range(d) if T[j] < 2)
        else:
            npr = 0
        ok &= (m3 == M3(d, i)) and (nm == nminus(d, i)) and (i == 0 or npr == npairs(d, i))
    return ok


def count(L, d, B, Kz, Krel, ninv=None):
    """Exact count (char3_impl.formula) for square bands a = b = L/2 and the schedule 'all'."""
    a = b = L // 2
    N0 = math.comb(L, a)
    f = {}
    f['forms_mul'] = Kz * (((L - 2) + L * max(d - 2, 0)) if d >= 2 else 0)
    f['table_mul'] = Kz * sum(math.comb(L, i) * npairs(d, i) for i in range(1, 2 * d + 1))
    f['table_add'] = Kz * sum(math.comb(L, i) * (npairs(d, i) - M3(d, i)) for i in range(1, 2 * d + 1))
    f['sign_neg'] = Kz * sum(math.comb(L, i) * nminus(d, i) for i in range(0, 2 * d + 1))
    f['inv'] = Kz * math.comb(L, 2 * d) if ninv is None else ninv          # upper bound: every S good
    R = sum(math.comb(a, i) * M3(d, i) for i in range(0, min(2 * d, a) + 1))
    S0 = sum(math.comb(a, i) for i in range(0, min(2 * d, a) + 1))
    f['read_mul'] = Kz * (B + B * Krel) * N0 * R
    f['read_add'] = Kz * (B + B * Krel) * N0 * (R - S0)
    f['enc_add'] = Kz * (B + B * Krel) * 2 * (4 ** L - 3 ** L)
    f['leaf_mul'] = Kz * Krel * B * B * 4 ** L
    f['dec_add'] = Kz * Krel * B * B * 3 * L * 4 ** (L - 1)
    f['div_mul'] = (B * N0) ** 2
    f['total'] = sum(f.values())
    f['N'] = B * N0
    return f


def main():
    log('Characteristic 3: exact count at L = 8d, a = b = L/2 (char3_count.py)')
    log(f'closed forms M3, npairs, nminus = brute force for d = 1..5: {all(brute(d) for d in range(1, 6))}')
    # agreement with the pipeline's formula (square bands, small L)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from char3_impl import formula
    for L, d in [(8, 1), (16, 2), (24, 3)]:
        Krel = L * (L - 1)
        for B, Kz in [(1, 1), (2, 3)]:
            mine = count(L, d, B, Kz, Krel)
            theirs = formula(L, L // 2, L // 2, d, B, Kz, Kz * Krel, Kz * math.comb(L, 2 * d))
            same = mine['total'] == theirs['total'] - theirs['inv'] + mine['inv'] if False else \
                (mine['total'] == theirs['total'])
            log(f'L={L} d={d} B={B} K_z={Kz}: this script = pipeline formula: {same}')
    # per entry at B = 1 (largest, every B-dependent term decreases in B), K_z = 1 (the ratio
    # (K_z A + 1)/(K_z L^4) is largest at K_z = 1)
    worst = Fraction(0); worstL = None
    log('L, d, per entry (B=1, K_z=1), per entry / L^4, leading K(1+3L/4)4^L/N0^2 / L^4, parts per entry: '
        'leaf+dec, enc, read, set-up, div')
    for d in range(1, 33):
        L = 8 * d
        Krel = L * (L - 1)
        f = count(L, d, 1, 1, Krel)
        N2 = f['N'] ** 2
        per = Fraction(f['total'], N2)
        r = per / L ** 4
        if r > worst:
            worst, worstL = r, L
        lead = Fraction(Krel * (4 ** L + 3 * L * 4 ** (L - 1)), N2)
        enc = Fraction(f['enc_add'], N2)
        rd = Fraction(f['read_mul'] + f['read_add'], N2)
        su = Fraction(f['forms_mul'] + f['table_mul'] + f['table_add'] + f['sign_neg'] + f['inv'], N2)
        if L <= 64 or L % 64 == 0:
            log(f'L={L:3d} d={d:2d}: per entry {float(per):.6g}, /L^4 = {float(r):.5f}, leading/L^4 = '
                f'{float(lead / L ** 4):.5f}; leaf+dec {float(lead):.4g}, enc {float(enc):.4g}, read {float(rd):.3g}, '
                f'set-up {float(su):.3g}, div 1')
    log(f'max over L = 8d <= 256 of (per entry at K_z = 1, B = 1)/L^4: {float(worst):.5f} at L = {worstL}')
    with open(OUT, 'w') as fh:
        fh.write('\n'.join(lines) + '\n')


if __name__ == '__main__':
    main()
