r"""Second implementation: Lemma 3.3 (covering) at the shapes of Theorem 1.1(b),(c), L = 8d, a = L/2, target 2d;
exhaustive at L = 8, sampled at L = 16, 24. Pi_L = s^k o t_1 o ... o t_j built from adjacent transpositions.
Output: ../outputs/indep3_covering_out.txt.
"""
import itertools, os, random
out = open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'outputs', 'indep3_covering_out.txt'), 'w')
def say(*a):
    s = ' '.join(str(x) for x in a); print(s); out.write(s + '\n'); out.flush()
def compose(p, q): return [p[q[x]] for x in range(len(p))]
for L, samples in [(8, None), (16, 20000), (24, 5000)]:
    d = L // 8; a = L // 2
    s = [(l + 1) % L for l in range(L)]
    ts = None
    for order in ([(j - 1, j) for j in range(1, L)], [(j - 1, j) for j in range(L - 1, 0, -1)]):
        prod = list(range(L))
        for (u, v) in order:
            t = list(range(L)); t[u], t[v] = v, u; prod = compose(prod, t)
        if prod == s: ts = order
    Pi = []; sk = list(range(L))
    for k in range(L):
        cur = sk; Pi.append(cur)
        for (u, v) in ts[:-1]:
            t = list(range(L)); t[u], t[v] = v, u; cur = compose(cur, t); Pi.append(cur)
        sk = compose(s, sk)
    # consecutive members differ by one transposition applied on the right
    okpath = all(sum(1 for x in range(L) if compose([Pi[i].index(y) for y in range(L)], Pi[i + 1])[x] != x) == 2
                 for i in range(len(Pi) - 1))
    shifts_in = all([(l + k) % L for l in range(L)] in Pi for k in range(L))
    masks = [[1 << p[l] for l in range(L)] for p in Pi]
    rng = random.Random(L)
    if samples is None:
        pairs = [(r, c) for r in itertools.combinations(range(L), a) for c in itertools.combinations(range(L), a)]
    else:
        pairs = [(tuple(rng.sample(range(L), a)), tuple(rng.sample(range(L), a))) for _ in range(samples)]
    miss = 0
    for r, c in pairs:
        rm = sum(1 << l for l in r)
        if not any(bin(rm & sum(m[l] for l in c)).count('1') == 2 * d for m in masks): miss += 1
    say(f'L={L} a={a} 2d={2*d}: |Pi_L| = {len(Pi)} (distinct {len({tuple(p) for p in Pi})}), path steps are single '
        f'right transpositions: {okpath}, contains every s^k: {shifts_in}; pairs tested {len(pairs)}'
        f'{" (all)" if samples is None else " (random)"}, pairs with no pi giving 2d: {miss}')
out.close()
