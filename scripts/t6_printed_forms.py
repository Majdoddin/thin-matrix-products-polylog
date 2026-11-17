r"""Check of the decomposition of T6 as printed in Lemma 6.1 of the paper (Section 6).
The six rank-one terms are typed here exactly as in the paper (signs over F_3), expanded over F_3, and compared
with T6 and with the data file ../data/T6_decomposition.json.
Prints the error monomials E6 grouped by class (cell; row block of s, column block of t) with their weights under
the block grading h_A = (-3, -5), h_B = (4, -2, -1), h(z_ij) = -h_A(i) - h_B(j) (Table 4).
Usage: python3 t6_printed_forms.py > ../outputs/t6_printed_forms_out.txt
"""
import itertools, json, os
p = 3
# paper's forms: phi (left), psi (right), chi (output)
phi = [dict(o0=1, p0=-1, x0=-1, x1=1), dict(p0=1, x0=-1, x1=-1), dict(x0=1, x1=-1),
       dict(x0=1, x1=1), dict(x0=1, p0=1, x1=-1), dict(x0=1)]
psi = [dict(y0=1), dict(y0=1, y1=1, y2=-1, q0=1), dict(y0=-1, y1=1, r0=1),
       dict(y0=-1, y1=1, q0=1, r0=1), dict(y1=1, y2=-1), dict(y0=-1, y2=-1, r0=1)]
chi = [dict(w00=1, z01=1, z02=1, z10=-1), dict(w00=1, z01=1, z02=-1, z10=1, z11=1),
       dict(w00=1, z02=-1, z10=1, z11=-1, z12=-1), dict(w00=1, z02=-1, z10=1),
       dict(w00=-1, z01=-1, z02=1, z10=-1, z11=-1, z12=1), dict(w00=-1, z10=-1, z12=1)]
T6 = {('o0', 'y0', 'w00'): 1, ('p0', 'q0', 'w00'): 1, ('x0', 'r0', 'w00'): 1, ('x0', 'y1', 'z01'): 1,
      ('x0', 'y2', 'z02'): 1, ('x1', 'y0', 'z10'): 1, ('x1', 'y1', 'z11'): 1, ('x1', 'y2', 'z12'): 1}
rb = dict(x0=0, p0=0, o0=0, x1=1)
cb = dict(y0=0, q0=0, r0=0, y1=1, y2=2)
cell = dict(w00=(0, 0), z01=(0, 1), z02=(0, 2), z10=(1, 0), z11=(1, 1), z12=(1, 2))
hA, hB = (-3, -5), (4, -2, -1)
def expand(F):
    P = {}
    for a, b, c in F:
        for (x, cx), (y, cy), (z, cz) in itertools.product(a.items(), b.items(), c.items()):
            P[(x, y, z)] = (P.get((x, y, z), 0) + cx * cy * cz) % p
    return {k: v for k, v in P.items() if v}
P = expand(zip(phi, psi, chi))
print('T6 monomials present with coefficient 1:', all(P.get(m) == 1 for m in T6))
E = {m: c for m, c in P.items() if m not in T6}
print('product monomials', len(P), ' of E6', len(E))
w = lambda m: hA[rb[m[0]]] - hA[cell[m[2]][0]] + hB[cb[m[1]]] - hB[cell[m[2]][1]]
print('weight 0 on T6:', all(w(m) == 0 for m in T6), ' min weight on E6:', min(w(m) for m in E))
# compare with the data file
rec = json.load(open(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data', 'T6_decomposition.json')))
terms = rec['terms']
Prec = expand([tuple({k: v for k, v in f.items()} for f in t) for t in terms])
print('expansion equals that of the data file:', Prec == P)
g = rec['grading']
hl = {**{s: hA[rb[s]] for s in rb}, **{t: hB[cb[t]] for t in cb}}
hz = {z: -hA[i] - hB[j] for z, (i, j) in cell.items()}
print('grading equals the data file:', all(g['A:' + s] == hl[s] for s in rb) and all(g['B:' + t] == hl[t] for t in cb)
      and all(g['C:' + z] == hz[z] for z in cell), hz)
classes = {}
for m, c in sorted(E.items(), key=lambda kv: (cell[kv[0][2]], rb[kv[0][0]], cb[kv[0][1]], kv[0])):
    classes.setdefault((m[2], rb[m[0]], cb[m[1]]), []).append((m, c if c == 1 else -1))
for (z, i, j), ms in classes.items():
    print('%s  class (%d,%d) weight %d :' % (z, i, j, w(ms[0][0])), ' '.join(('+' if c == 1 else '-') + a + b for (a, b, _), c in ms))
print('number of classes', len(classes))
