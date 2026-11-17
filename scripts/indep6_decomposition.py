r"""Independent check (the files indep6_*.py share no code with t6_*.py) of the properties of T6 and T6'.
Reads the six rank-one terms and the grading from ../data/T6_decomposition.json, defines T6 from Section 6,
expands over F_3 and checks:
 (a) product = T6 + E, the eight monomials of T6 with coefficient exactly 1, E disjoint from T6 (Lemma 6.1);
 (b) weight 0 on T6, >= 1 on every monomial of E (so >= 0 on every monomial of the product);
 (c) grading constant on row blocks {x0,p0,o0},{x1} and column blocks {y0,q0,r0},{y1},{y2};
 (d) h(z_ij) = -h_A(i) - h_B(j) for every cell;
 (e) the same for T6' (A and B exchanged);
 (f) the terms hard-coded in t6_impl.py equal the data file's terms (text comparison of the literal).
Usage: python3 indep6_decomposition.py > ../outputs/indep6_decomposition_out.txt
"""
import json, itertools, os, re
p = 3
HERE = os.path.dirname(os.path.abspath(__file__))
rec = json.load(open(os.path.join(HERE, '..', 'data', 'T6_decomposition.json')))
terms, g = rec['terms'], rec['grading']
print('record p =', rec['p'], ' number of terms =', len(terms))
T6 = {('o0', 'y0', 'w00'): 1, ('p0', 'q0', 'w00'): 1, ('x0', 'r0', 'w00'): 1, ('x0', 'y1', 'z01'): 1,
       ('x0', 'y2', 'z02'): 1, ('x1', 'y0', 'z10'): 1, ('x1', 'y1', 'z11'): 1, ('x1', 'y2', 'z12'): 1}
rowblk = {'x0': 0, 'p0': 0, 'o0': 0, 'x1': 1}
colblk = {'y0': 0, 'q0': 0, 'r0': 0, 'y1': 1, 'y2': 2}
cell = {'w00': (0, 0), 'z01': (0, 1), 'z02': (0, 2), 'z10': (1, 0), 'z11': (1, 1), 'z12': (1, 2)}

def expand(tr, swap):
    P = {}
    for a, b, c in tr:
        if swap: a, b = b, a
        for (x, cx), (y, cy), (z, cz) in itertools.product(a.items(), b.items(), c.items()):
            P[(x, y, z)] = (P.get((x, y, z), 0) + cx * cy * cz) % p
    return {k: v for k, v in P.items() if v}

for swap in (False, True):
    name = "T6'" if swap else 'T6'
    T = {((b, a, z) if swap else (a, b, z)): c for (a, b, z), c in T6.items()}
    hA = {('A:' if not swap else 'B:') + s: None for s in rowblk}
    hval = lambda letter, side: g[('A:' if side == 'L' else 'B:') + letter] if not swap else g[('B:' if side == 'L' else 'A:') + letter]
    P = expand(terms, swap)
    wt = lambda m: hval(m[0], 'L') + hval(m[1], 'R') + g['C:' + m[2]]
    okT = all(P.get(m) == c for m, c in T.items())
    E = {m: c for m, c in P.items() if m not in T}
    print('%s: monomials of product %d, of T %d, of E %d' % (name, len(P), len(T), len(E)))
    print('  (a) T monomials present with coefficient 1:', okT)
    print('  (b) weight 0 on T:', all(wt(m) == 0 for m in T), '| min weight on E:', min(wt(m) for m in E),
          '| weights on E:', sorted(set(wt(m) for m in E)))
    # (c) block constancy (left letters of this kind are rowblk keys for T6, colblk keys for T6')
    L_blk, R_blk = (colblk, rowblk) if swap else (rowblk, colblk)
    bc = all(len({hval(s, 'L') for s in L_blk if L_blk[s] == i}) == 1 for i in set(L_blk.values())) and \
         all(len({hval(s, 'R') for s in R_blk if R_blk[s] == j}) == 1 for j in set(R_blk.values()))
    print('  (c) block-constant:', bc)
    # (d) output weights
    hb_L = {L_blk[s]: hval(s, 'L') for s in L_blk}; hb_R = {R_blk[s]: hval(s, 'R') for s in R_blk}
    okd = all(g['C:' + z] == -(hb_L[(ij[1] if swap else ij[0])] + hb_R[(ij[0] if swap else ij[1])]) for z, ij in cell.items())
    print('  (d) h(z_ij) = -h_A(i) - h_B(j):', okd)
    # monomials of the product whose output letter's cell does not match the blocks of its letters
    off = [m for m in P if ((L_blk[m[0]], R_blk[m[1]]) != ((cell[m[2]][1], cell[m[2]][0]) if swap else cell[m[2]]))]
    print('  E monomials with letters outside the blocks of their output cell: %d of %d' % (len(off), len(E)))
# (f) hard-coded terms in t6_impl.py
src = open(os.path.join(HERE, 't6_impl.py')).read()
mt = re.search(r'terms = (\[\[\{"o0".*?\}\]\])\n', src, re.S)
lit = eval(mt.group(1).replace('\n', ' '))
print('(f) t6_impl.py terms == data-file terms:', lit == terms)
gl = re.search(r"h = (\{'o0'.*?\})\n", src, re.S)
hcode = eval(gl.group(1).replace('\n', ' '))
hjs = {k.split(':')[1]: v for k, v in g.items()}
print('(f) t6_impl.py grading == data-file grading:', hcode == hjs)
