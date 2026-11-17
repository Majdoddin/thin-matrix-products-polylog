r"""The exact operation count of the algorithm of Section 4 (Table 2) as a function of L and B, its cost per entry
and leading term (Lemma 4.2 and the paragraph after it), and the crossover against the textbook algorithm and the
Four Russians method over F_2 (Section 10). Exact integers (Python), no floating point except for the reported
ratios. Output: ../outputs/char2_count_out.txt.

Model for the crossover (Section 10): one operation = one GF(2^k) addition, multiplication or inversion for the T4
algorithm; one F_2 multiplication or addition for the textbook algorithm (2D - 1 per entry) and 2D/log2(N) per entry
for Four Russians (blocks of log2 N inner indices; per block a table of 2^k combinations of k rows of Y and one
lookup-and-add per output entry). The word-packed variant divides the F_2 algorithms by 64 (64 entries per word)
and the T4 additions by floor(64/k) (k-bit field elements), multiplications unpacked. The rows "K = ideal 1/f" use
1/f runs per band pair, f the fraction of label pairs at intersection d (the least number of runs of any schedule).
"""
import math, os, sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from char2_impl import formula

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'char2_count_out.txt')
lines = []


def log(s):
    lines.append(s); print(s, flush=True)


log('Exact operation count (Table 2; identical to the instrumented counts of char2_impl.py at L = 6, 8, 10, 12)')
log('Per entry, a = L/2, d = floor(L/4), K = L(L-1) (all of Pi_L), B -> infinity (the B-independent part) and B = 1:')
log(f'{"L":>4} {"d":>3} {"N0":>12} {"D":>8} {"perentry_Binf":>14} {"/L^4":>8} {"perentry_B1":>14} '
    f'{"4^L/N0^2":>9} {"/(piL/2)":>8}')


def per_entry_Binf(L, a, d, K):
    """Limit B -> infinity of total/N^2: leaf products + decoding + division (the B^2 and N^2 terms)."""
    N0 = math.comb(L, a)
    return (K * 4 ** L + K * 3 * L * 4 ** (L - 1)) / N0 ** 2 + 1


rows = []
for L in list(range(4, 41, 2)) + [48, 64, 80, 96, 112, 128, 144, 160, 192, 256]:
    a = L // 2; d = (a * a) // L
    K = L * (L - 1)
    N0 = math.comb(L, a)
    f1 = formula(L, a, d, 1, K)
    pinf = per_entry_Binf(L, a, d, K)
    r = 4 ** L / N0 ** 2
    rows.append((L, d, N0, pinf, f1['per_entry']))
    log(f'{L:>4} {d:>3} {N0:>12.4g} {2 ** d:>8.3g} {pinf:>14.6g} {pinf / L ** 4:>8.4f} {f1["per_entry"]:>14.6g} '
        f'{r:>9.4f} {r / (math.pi * L / 2):>8.4f}')
log(f'leading constant: per entry ~ K (1 + 3L/4) (pi L/2) ~ (3 pi/8) L^4 = {3 * math.pi / 8:.4f} L^4')
log('Bound (1) of Section 2: C(L, L/2) >= 2^L/sqrt(2L) (L even), so 4^L/N0^2 <= 2L.')
ok = all(4 ** L <= 2 * L * math.comb(L, L // 2) ** 2 for L in range(2, 400, 2))
log(f'  checked 4^L <= 2L C(L,L/2)^2 for all even L in [2, 398]: {ok}')

worst = max((formula(L, L // 2, L // 4, 1, L * (L - 1))['per_entry'] / L ** 4, L) for L in range(4, 257, 2))
log(f'max over every even L in [4, 256] of (per entry at B = 1, K = L(L-1))/L^4: {worst[0]:.4f} at L = {worst[1]}')
for B in (1, 2, 4, 16):
    vals = [formula(L, L // 2, L // 4, B, L * (L - 1))['per_entry'] for L in (12, 32, 128)]
    log(f'  B = {B}: per entry at L = 12, 32, 128: ' + ', '.join(f'{v:.6g}' for v in vals))

# ---------- crossover ----------
log('')
log('Crossover over F_2 (L multiple of 4 so that D = 2^{L/4} >= N0^{1/4}; N = N0 = C(L, L/2), B = 1, and B -> inf):')


def t4_cost(L, B, K, packed=False):
    a = L // 2; d = (a * a) // L
    f = formula(L, a, d, B, K)
    if not packed:
        return f['per_entry']
    k = max(1, math.ceil(math.log2(L)))
    per_word = max(1, 64 // k)
    return (f['mul'] + f['inv'] + f['add'] / per_word) / (B * math.comb(L, a)) ** 2


def trivial(D, N, packed=False):
    c = 2 * D - 1
    return c / 64 if packed else c


def four_russians(D, N, packed=False):
    c = 2 * D / math.log2(N)
    return c / 64 if packed else c


for packed in (False, True):
    for Kname in ('L(L-1)', 'ideal 1/f'):
        for comp_name, comp in (('trivial', trivial), ('Four Russians', four_russians)):
            found = None
            for L in range(8, 1200, 4):
                a = L // 2; d = L // 4
                N0 = math.comb(L, a)
                if Kname == 'L(L-1)':
                    K = L * (L - 1)
                else:
                    # ideal covering: K = M-weighted 1/f, f = M/N0^2 the fraction of pairs at intersection d
                    M = math.factorial(L) // (math.factorial(L - 2 * a + d) * math.factorial(a - d) ** 2
                                              * math.factorial(d))
                    K = N0 ** 2 / M
                cost = t4_cost(L, 1, K, packed) if Kname == 'L(L-1)' else None
                if cost is None:
                    f = formula(L, a, d, 1, 1)
                    # scale the K-proportional parts by K (formula is affine in K)
                    f0 = formula(L, a, d, 1, 0)
                    if packed:
                        k = max(1, math.ceil(math.log2(L))); pw = max(1, 64 // k)
                        tot = lambda g: g['mul'] + g['inv'] + g['add'] / pw
                    else:
                        tot = lambda g: g['total']
                    cost = (tot(f0) + K * (tot(f) - tot(f0))) / N0 ** 2
                if cost < comp(2 ** d, N0, packed):
                    found = (L, d, N0, cost, comp(2 ** d, N0, packed), K)
                    break
            if found:
                L, d, N0, cost, cc, K = found
                log(f'  {"word-packed" if packed else "unit-cost"}, K = {Kname}, vs {comp_name}: first L = {L} '
                    f'(D = 2^{d} = {2 ** d:.3g}, N = C(L,L/2) = 2^{math.log2(N0):.1f}, K = {K:.4g}): '
                    f'T4 {cost:.4g} vs {cc:.4g} per entry')
            else:
                log(f'  {"word-packed" if packed else "unit-cost"}, K = {Kname}, vs {comp_name}: no crossover below '
                    f'L = 1200')

with open(OUT, 'w') as fh:
    fh.write('\n'.join(lines) + '\n')
