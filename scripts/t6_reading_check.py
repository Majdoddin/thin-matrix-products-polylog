r"""Direct check of Proposition 6.3 (the reading of T6), independent of t6_impl.py.
In V = E[xi_1..xi_d]/(xi_i^3), E = GF(81): for random forms phi_l (rows of Phi_S, random or Vandermonde),
  sum_{e : S -> {0,1,2}} <x, a(e)> <y, a(2 - e)>^  ==  det(Phi_S)^2 sum_T x_T y_T,
with a(e) = prod_l phi_l^{e_l}, <x, v> = sum_T x_T v_T, <y, v>^ = sum_T y_T v_{2-T}. Also singular Phi_S (both
sides 0) and a control with the sign convention xi + eta (slice coefficients (1, 2, 1) would be needed; with all
coefficients 1 the identity fails).
Usage: python3 t6_reading_check.py > ../outputs/t6_reading_check_out.txt
"""
import itertools, random, sys

class GF:
    def __init__(self, p, k, f):
        self.p, self.k, self.f = p, k, f
        self.q = p ** k
    def add(self, a, b): return tuple((x + y) % self.p for x, y in zip(a, b))
    def sub(self, a, b): return tuple((x - y) % self.p for x, y in zip(a, b))
    def mul(self, a, b):
        k, p = self.k, self.p
        c = [0] * (2 * k - 1)
        for i in range(k):
            for j in range(k):
                c[i + j] += a[i] * b[j]
        for t in range(2 * k - 2, k - 1, -1):
            for i in range(k):
                c[t - k + i] -= self.f[i] * c[t]
        return tuple(x % p for x in c[:k])
    def zero(self): return (0,) * self.k
    def one(self): return (1,) + (0,) * (self.k - 1)
    def rand(self): return tuple(random.randrange(self.p) for _ in range(self.k))
    def elem(self, i): return tuple((i // self.p ** j) % self.p for j in range(self.k))

def det(F, M):
    n = len(M); M = [row[:] for row in M]; d = F.one()
    for c in range(n):
        piv = next((r for r in range(c, n) if M[r][c] != F.zero()), None)
        if piv is None: return F.zero()
        if piv != c:
            M[c], M[piv] = M[piv], M[c]; d = F.sub(F.zero(), d)
        inv = next(e for e in (F.elem(i) for i in range(1, F.q)) if F.mul(e, M[c][c]) == F.one())
        d = F.mul(d, M[c][c])
        for r in range(c + 1, n):
            fac = F.mul(M[r][c], inv)
            M[r] = [F.sub(M[r][j], F.mul(fac, M[c][j])) for j in range(n)]
    return d

def vmul(F, d, u, v):
    """product in V of dicts T -> coeff"""
    out = {}
    for T, a in u.items():
        for U, b in v.items():
            W = tuple(x + y for x, y in zip(T, U))
            if max(W) <= 2:
                out[W] = F.add(out.get(W, F.zero()), F.mul(a, b))
    return {T: c for T, c in out.items() if c != F.zero()}

def check(F, d, Phi, sign_plus=False):
    phis = [{tuple(int(i == j) for j in range(d)): Phi[l][i] for i in range(d) if Phi[l][i] != F.zero()} for l in range(d)]
    one = {(0,) * d: F.one()}
    pw = [[one, phis[l], vmul(F, d, phis[l], phis[l])] for l in range(d)]
    mons = list(itertools.product(range(3), repeat=d))
    x = {T: F.rand() for T in mons}; y = {T: F.rand() for T in mons}
    lhs = F.zero()
    for e in itertools.product(range(3), repeat=d):
        a = one; b = one
        for l in range(d):
            a = vmul(F, d, a, pw[l][e[l]]); b = vmul(F, d, b, pw[l][2 - e[l]])
        la = F.zero()
        for T, c in a.items(): la = F.add(la, F.mul(x[T], c))
        rb = F.zero()
        for T, c in b.items(): rb = F.add(rb, F.mul(y[tuple(2 - t for t in T)], c))
        term = F.mul(la, rb)
        if sign_plus and sum(1 for t in e if t == 1) % 2:     # control: coefficients (1, -1, 1) on the slice
            term = F.sub(F.zero(), term)
        lhs = F.add(lhs, term)
    dt = det(F, Phi); c = F.mul(dt, dt)
    ip = F.zero()
    for T in mons: ip = F.add(ip, F.mul(x[T], y[T]))
    return lhs == F.mul(c, ip), dt == F.zero()

if __name__ == '__main__':
    random.seed(5)
    F = GF(3, 4, [2, 0, 0, 1])     # x^4 + x^3 + 2 over F_3 (irreducible; checked below by brute force: no factor)
    # irreducibility check of x^4 + x^3 + 2: no root and no quadratic factor
    def polymod_irred():
        import itertools as it
        f = [2, 0, 0, 1, 1]        # coefficients low to high
        def pmod(a, b):
            a = a[:]
            while len(a) >= len(b) and any(a):
                if a[-1] == 0: a.pop(); continue
                c = a[-1] * pow(b[-1], -1, 3) % 3; s = len(a) - len(b)
                for i in range(len(b)): a[s + i] = (a[s + i] - c * b[i]) % 3
                a.pop()
            return a
        for deg in (1, 2):
            for co in it.product(range(3), repeat=deg):
                g = list(co) + [1]
                if not any(pmod(f, g)): return False
        return True
    print('GF(81) = F_3[x]/(x^4 + x^3 + 2), irreducible:', polymod_irred())
    tot = bad = sing = 0
    for d in (2, 3, 4, 5):
        reps = {2: 30, 3: 20, 4: 6, 5: 2}[d]
        for r in range(reps):
            if r % 2 == 0:   # Vandermonde with distinct random nodes
                nodes = random.sample(range(81), d)
                Phi = [[F.one()] + [None] * (d - 1) for _ in range(d)]
                for l in range(d):
                    t = F.elem(nodes[l]); pw = F.one()
                    for i in range(1, d):
                        pw = F.mul(pw, t); Phi[l][i] = pw
            else:            # random matrix (sometimes singular: force one singular case per d)
                Phi = [[F.rand() for _ in range(d)] for _ in range(d)]
                if r == 1: Phi[1] = Phi[0][:]
            ok, sg = check(F, d, Phi)
            tot += 1; bad += (not ok); sing += sg
        okc, _ = check(F, d, [[F.rand() for _ in range(d)] for _ in range(d)], sign_plus=True)
        print('d=%d: %d cases, identity fails in %d so far; control (slice signs 1,-1,1) holds: %s' % (d, tot, bad, okc), flush=True)
    print('TOTAL: %d cases (%d with singular Phi_S), failures %d' % (tot, sing, bad))
