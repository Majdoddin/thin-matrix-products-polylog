r"""Second, independently written implementation of Section 4 (files indep2_*.py, written from the text only,
sharing no code with char2_impl.py): GF(2^n) arithmetic by full multiplication tables (n <= 8), and embeddings
GF(2^j) -> GF(2^n) for j | n.
"""
import numpy as np

PRIM = {1: 0b11, 2: 0b111, 3: 0b1011, 4: 0b10011, 5: 0b100101, 6: 0b1000011, 8: 0b100011101}


class GF:
    def __init__(self, n):
        self.n = n
        self.q = 1 << n
        poly = PRIM[n]
        q = self.q
        if n == 1:
            self.mul_t = np.array([[0, 0], [0, 1]], dtype=np.uint8)
            self.inv_t = np.array([0, 1], dtype=np.uint8)
            return
        exp = np.zeros(2 * q, dtype=np.int64)
        log = np.full(q, -1, dtype=np.int64)
        x = 1
        for i in range(q - 1):
            exp[i] = x
            assert log[x] == -1, 'polynomial not primitive'
            log[x] = i
            x <<= 1
            if x & q:
                x ^= poly
        assert x == 1
        for i in range(q - 1, 2 * q):
            exp[i] = exp[i - (q - 1)]
        self.exp, self.log = exp, log
        mt = np.zeros((q, q), dtype=np.uint8)
        for a in range(1, q):
            for b in range(1, q):
                mt[a, b] = exp[log[a] + log[b]]
        self.mul_t = mt
        inv = np.zeros(q, dtype=np.uint8)
        for a in range(1, q):
            inv[a] = exp[(q - 1 - log[a]) % (q - 1)]
        self.inv_t = inv
        # sanity: distributivity on a sample, a * inv(a) = 1
        for a in range(1, q):
            assert mt[a, inv[a]] == 1

    def mul(self, a, b):
        return self.mul_t[a, b]

    def inv(self, a):
        assert a != 0
        return int(self.inv_t[a])

    def pow(self, a, e):
        r = 1
        for _ in range(e):
            r = int(self.mul_t[r, a])
        return r

    def det(self, M):
        """Determinant of a square matrix over GF(2^n) by Gaussian elimination (char 2: no signs)."""
        A = [list(map(int, row)) for row in M]
        k = len(A)
        det = 1
        for col in range(k):
            piv = next((r for r in range(col, k) if A[r][col] != 0), None)
            if piv is None:
                return 0
            A[col], A[piv] = A[piv], A[col]
            det = int(self.mul_t[det, A[col][col]])
            iv = self.inv(A[col][col])
            for r in range(col + 1, k):
                if A[r][col]:
                    f = int(self.mul_t[A[r][col], iv])
                    for cc in range(col, k):
                        A[r][cc] ^= int(self.mul_t[f, A[col][cc]])
        return det


def embedding(F, E):
    """Ring embedding F = GF(2^j) -> E = GF(2^n), j | n, as a lookup array of length |F|."""
    j, n = F.n, E.n
    assert n % j == 0
    if j == n:
        return np.arange(F.q, dtype=np.uint8)
    if j == 1:
        return np.array([0, 1], dtype=np.uint8)
    poly = PRIM[j]
    # find a root beta in E of F's defining polynomial
    beta = None
    for b in range(2, E.q):
        val, pw = 0, 1
        for i in range(j + 1):
            if (poly >> i) & 1:
                val ^= pw
            pw = int(E.mul_t[pw, b])
        if val == 0:
            beta = b
            break
    assert beta is not None
    powers = [1]
    for i in range(1, j):
        powers.append(int(E.mul_t[powers[-1], beta]))
    emb = np.zeros(F.q, dtype=np.uint8)
    for x in range(F.q):
        v = 0
        for i in range(j):
            if (x >> i) & 1:
                v ^= powers[i]
        emb[x] = v
    # check homomorphism on all pairs
    for x in range(F.q):
        for y in range(F.q):
            assert emb[x ^ y] == emb[x] ^ emb[y]
            assert emb[F.mul_t[x, y]] == E.mul_t[emb[x], emb[y]]
    return emb
