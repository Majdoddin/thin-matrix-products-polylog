r"""Second implementation for Section 5 (files indep3_*.py, written from the text only, sharing no code with
char3_impl.py and gf3k.py): GF(3^k). Elements are ints 0..q-1; base-3 digit i is the coefficient of x^i. Tables ADD,
SUB, NEG, MUL, INV (numpy int32). The modulus is the first monic irreducible polynomial of degree k in lexicographic
order of its low coefficients, found by trial division; g is the smallest element of multiplicative order q-1.
"""
import itertools
import numpy as np

P = 3


def poly_mod(a, m):
    a = list(a)
    while len(a) >= len(m):
        c = a[-1] % P
        if c:
            sh = len(a) - len(m)
            for i, mi in enumerate(m):
                a[sh + i] = (a[sh + i] - c * mi) % P
        a.pop()
    return [x % P for x in a]


def poly_mul(a, b):
    r = [0] * (len(a) + len(b) - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                r[i + j] = (r[i + j] + x * y) % P
    return r


def is_irreducible(m):
    k = len(m) - 1
    for deg in range(1, k // 2 + 1):
        for low in itertools.product(range(P), repeat=deg):
            f = list(low) + [1]
            rem = poly_mod(m, f)
            if not any(rem):
                return False
    return True


class GF:
    def __init__(self, k):
        self.k, self.q = k, P ** k
        if k == 1:
            self.mod = [0, 1]
        else:
            for low in itertools.product(range(P), repeat=k):
                m = list(low) + [1]
                if m[0] != 0 and is_irreducible(m):
                    self.mod = m
                    break
        q, kk = self.q, k
        digits = [[(e // P ** i) % P for i in range(kk)] for e in range(q)]
        self.digits = digits
        toint = lambda v: sum((v[i] % P) * P ** i for i in range(kk))
        D = np.array(digits, dtype=np.int64)
        pw = P ** np.arange(kk)
        self.ADD = (((D[:, None, :] + D[None, :, :]) % P) @ pw).astype(np.int32)
        self.SUB = (((D[:, None, :] - D[None, :, :]) % P) @ pw).astype(np.int32)
        self.NEG = (((-D) % P) @ pw).astype(np.int32)
        MUL = np.zeros((q, q), dtype=np.int32)
        for a in range(q):
            for b in range(a, q):
                pr = poly_mul(digits[a], digits[b]) if kk > 1 else [digits[a][0] * digits[b][0] % P]
                r = poly_mod(pr, self.mod) if kk > 1 else pr
                r = (r + [0] * kk)[:kk]
                MUL[a, b] = MUL[b, a] = toint(r)
        self.MUL = MUL
        INV = np.zeros(q, dtype=np.int32)
        for a in range(1, q):
            INV[a] = int(np.nonzero(MUL[a] == 1)[0][0])
        self.INV = INV
        # schoolbook self-check of the multiplication: associativity and distributivity on a sample
        rng = np.random.default_rng(1)
        s = rng.integers(0, q, size=(2000, 3))
        a, b, c = s[:, 0], s[:, 1], s[:, 2]
        assert np.all(MUL[MUL[a, b], c] == MUL[a, MUL[b, c]])
        assert np.all(MUL[a, self.ADD[b, c]] == self.ADD[MUL[a, b], MUL[a, c]])
        # generator
        self.g = None
        for e in range(2, q) if q > 2 else []:
            x, o = e, 1
            while x != 1:
                x = int(MUL[x, e]); o += 1
            if o == q - 1:
                self.g = e
                break
        if q == 3:
            self.g = 2

    def pow(self, a, n):
        r = 1
        a = int(a)
        while n:
            if n & 1:
                r = int(self.MUL[r, a])
            a = int(self.MUL[a, a]); n >>= 1
        return r

    def from_int(self, n):
        """image of the integer n in the prime field"""
        return n % P  # element n mod 3 is the constant digit

    def dot(self, A, B, axis=-1):
        """sum over axis of A*B elementwise, by tables"""
        M = self.MUL[A, B]
        M = np.moveaxis(M, axis, 0)
        acc = M[0].copy()
        for t in range(1, M.shape[0]):
            acc = self.ADD[acc, M[t]]
        return acc
