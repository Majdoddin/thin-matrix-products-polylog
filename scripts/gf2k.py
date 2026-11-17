r"""Arithmetic in GF(2^k), k = 4 or 8, by log/antilog tables; elements are the ints 0..2^k-1 and addition is
XOR. Also the ring V = F[x_1..x_d]/(x_i^2) with elements as numpy int arrays of length 2^d (index = bitmask T of the
monomial x_T). Used by char2_impl.py, which computes in GF(16).
"""
import numpy as np

_POLY = {4: 0b10011, 8: 0b100011011}   # x^4+x+1, x^8+x^4+x^3+x+1 (both primitive for these tables? 0x11b is not
                                       # primitive with generator 2; we use generator 3 for k = 8)
_GEN = {4: 2, 8: 3}


class GF:
    def __init__(self, k=4):
        self.k = k
        self.q = 1 << k
        q = self.q
        poly = _POLY[k]
        g = _GEN[k]
        exp = np.zeros(2 * q, dtype=np.int64)
        log = np.full(q, -1, dtype=np.int64)

        def mulslow(a, b):
            r = 0
            while b:
                if b & 1:
                    r ^= a
                b >>= 1
                a <<= 1
                if a & q:
                    a ^= poly
            return r
        x = 1
        for i in range(q - 1):
            exp[i] = x
            if log[x] != -1:
                raise ValueError('generator not primitive')
            log[x] = i
            x = mulslow(x, g)
        for i in range(q - 1, 2 * q):
            exp[i] = exp[i - (q - 1)]
        self.exp, self.log = exp, log
        # full multiplication table (q <= 256)
        T = np.zeros((q, q), dtype=np.int64)
        for a in range(1, q):
            for b in range(1, q):
                T[a, b] = exp[log[a] + log[b]]
        self.T = T
        self.mulslow = mulslow
        # self-check
        for a in range(q):
            for b in range(q):
                assert T[a, b] == mulslow(a, b)

    def mul(self, a, b):
        return self.T[a, b]

    def inv(self, a):
        assert a != 0
        return int(self.exp[(self.q - 1 - self.log[a]) % (self.q - 1)])

    def pow(self, a, e):
        if e == 0:
            return 1
        if a == 0:
            return 0
        return int(self.exp[(self.log[a] * e) % (self.q - 1)])


def popcount(x):
    return bin(x).count('1')


class Vring:
    """V = F[x_1..x_d]/(x_i^2) over GF F; element = int array of length 2^d."""
    def __init__(self, F, d):
        self.F, self.d, self.D = F, d, 1 << d

    def lin(self, coeffs):
        v = np.zeros(self.D, dtype=np.int64)
        for i, c in enumerate(coeffs):
            v[1 << i] = c
        return v

    def mul(self, u, v):
        F = self.F
        w = np.zeros(self.D, dtype=np.int64)
        nzu = np.nonzero(u)[0]
        nzv = np.nonzero(v)[0]
        for s in nzu:
            for t in nzv:
                if s & t == 0:
                    w[s | t] ^= F.T[u[s], v[t]]
        return w

    def mul_lin(self, u, coeffs):
        """u * (sum_i coeffs[i] x_i), O(d 2^d)."""
        F = self.F
        w = np.zeros(self.D, dtype=np.int64)
        for i, c in enumerate(coeffs):
            if c == 0:
                continue
            bit = 1 << i
            idx = np.arange(self.D)
            sel = (idx & bit) == 0
            src = idx[sel]
            w[src | bit] ^= F.T[u[src], c]
        return w

    def top(self, u):
        return int(u[self.D - 1])


def det_gf(F, A):
    """Determinant over GF by Gaussian elimination (char 2: no signs)."""
    A = [list(map(int, row)) for row in A]
    n = len(A)
    det = 1
    for col in range(n):
        piv = None
        for r in range(col, n):
            if A[r][col] != 0:
                piv = r
                break
        if piv is None:
            return 0
        if piv != col:
            A[col], A[piv] = A[piv], A[col]
        det = F.T[det, A[col][col]]
        ic = F.inv(A[col][col])
        for r in range(col + 1, n):
            if A[r][col]:
                f = F.T[A[r][col], ic]
                for cc in range(col, n):
                    A[r][cc] ^= F.T[f, A[col][cc]]
    return int(det)
