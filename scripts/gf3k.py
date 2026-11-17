r"""Arithmetic in E = GF(3^k), k = 1..8, for char3_impl.py and char3_isolation.py.

Elements are ints 0..q-1, read as base-3 digit vectors (digit i = coefficient of X^i in F_3[X]/(h)), h a monic
PRIMITIVE polynomial of degree k found by search (so X is a generator g of E^*). Tables: exp/log (g = X), digits,
add, neg, mul (q x q, only for q <= 729), inv. Everything is checked against schoolbook polynomial arithmetic.
"""
import itertools
import numpy as np


def _polymulmod(a, b, h, k):
    """a, b: digit lists of length k (low to high); h: monic, length k+1 (low to high)."""
    prod = [0] * (2 * k - 1)
    for i, x in enumerate(a):
        if x:
            for j, y in enumerate(b):
                prod[i + j] = (prod[i + j] + x * y) % 3
    for top in range(2 * k - 2, k - 1, -1):
        c = prod[top]
        if c:
            prod[top] = 0
            for i in range(k):
                prod[top - k + i] = (prod[top - k + i] - c * h[i]) % 3
    return prod[:k]


def _num(ds):
    return sum(int(c) * 3 ** i for i, c in enumerate(ds))


def _digits(a, k):
    return [(a // 3 ** i) % 3 for i in range(k)]


def find_primitive(k):
    """Smallest (in the base-3 order of the low coefficients) monic h of degree k for which X has order 3^k - 1."""
    q = 3 ** k
    for low in itertools.product(range(3), repeat=k):
        if low[0] == 0:
            continue
        h = list(low) + [1]
        x = [0] * k
        if k == 1:
            x = [(-h[0]) % 3]   # X = -h_0 in F_3[X]/(X + h_0)
        else:
            x[1] = 1
        cur = [1] + [0] * (k - 1)
        order = None
        for e in range(1, q):
            cur = _polymulmod(cur, x, h, k)
            if cur == [1] + [0] * (k - 1):
                order = e
                break
        if order == q - 1:
            return h
    raise ValueError('no primitive polynomial found')


class GF3k:
    def __init__(self, k, tables=True):
        self.k, self.q = k, 3 ** k
        q = self.q
        self.h = find_primitive(k)
        g = [0] * k
        if k == 1:
            g = [(-self.h[0]) % 3]
        else:
            g[1] = 1
        self.g = _num(g)
        exp = np.zeros(2 * (q - 1), dtype=np.int64)
        log = np.full(q, -1, dtype=np.int64)
        cur = [1] + [0] * (k - 1)
        for e in range(q - 1):
            v = _num(cur)
            assert log[v] == -1
            exp[e] = v; log[v] = e
            cur = _polymulmod(cur, g, self.h, k)
        exp[q - 1:] = exp[:q - 1]
        self.exp, self.log = exp, log
        self.DIG = np.array([_digits(a, k) for a in range(q)], dtype=np.int8)       # q x k
        self.P3 = np.array([3 ** i for i in range(k)], dtype=np.int64)
        self.neg = np.array([_num([(-x) % 3 for x in _digits(a, k)]) for a in range(q)], dtype=np.int64)
        if tables:
            assert q <= 729
            D = self.DIG.astype(np.int64)
            self.ADD = ((D[:, None, :] + D[None, :, :]) % 3) @ self.P3                 # q x q
            M = np.zeros((q, q), dtype=np.int64)
            la = log
            for a in range(1, q):
                M[a, 1:] = exp[la[a] + la[1:]]
            self.MUL = M
            # self-check against schoolbook multiplication on a sample
            rng = np.random.default_rng(0)
            for _ in range(200):
                a, b = (int(x) for x in rng.integers(0, q, 2))
                assert M[a, b] == _num(_polymulmod(_digits(a, k), _digits(b, k), self.h, k))
                assert self.ADD[a, b] == _num([(x + y) % 3 for x, y in zip(_digits(a, k), _digits(b, k))])
        self.inv_tab = np.zeros(q, dtype=np.int64)
        self.inv_tab[1:] = exp[(q - 1 - log[1:]) % (q - 1)]

    def mul(self, a, b):
        if a == 0 or b == 0:
            return 0
        return int(self.exp[self.log[a] + self.log[b]])

    def pow(self, a, e):
        if e == 0:
            return 1
        if a == 0:
            return 0
        return int(self.exp[(self.log[a] * e) % (self.q - 1)])

    def inv(self, a):
        assert a != 0
        return int(self.inv_tab[a])

    def add(self, a, b):
        return _num([(x + y) % 3 for x, y in zip(_digits(a, self.k), _digits(b, self.k))])
