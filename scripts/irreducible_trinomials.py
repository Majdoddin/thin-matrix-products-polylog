# Rabin irreducibility test over F_3 for f = x^n + x^(n/2) + 2, n = 2^(j+1) = 2, 4, ..., 1024: the example field
# of Corollary 1.3(ii), irreducible for every such n by [LN97, Theorem 3.35] (proof of Corollary 1.3 in
# characteristic 3); this is an independent computer check. Output: one line "n True" per n if f is irreducible.
# Usage: python3 irreducible_trinomials.py > ../outputs/irreducible_trinomials_out.txt
# f is irreducible iff x^(3^n) = x mod f and gcd(x^(3^(n/2)) - x, f) = 1 (2 is the only prime factor of n).
# Cubing is linear over F3 (sum a_i x^i)^3 = sum a_i x^(3i), so x^(3^k) mod f costs O(k n).
def reduce(c, n):  # x^n = 2 x^(n/2) + 1 mod f
    h = n // 2
    for k in range(len(c) - 1, n - 1, -1):
        a = c[k]
        if a:
            c[k] = 0
            c[k - h] = (c[k - h] + 2 * a) % 3
            c[k - n] = (c[k - n] + a) % 3
    return c[:n]
def frob(c, n):
    out = [0] * (3 * n)
    for i, a in enumerate(c):
        out[3 * i] = a
    return reduce(out, n)
def trim(p):
    while p and p[-1] == 0: p.pop()
    return p
def pmod(a, b):
    a = a[:]; inv = b[-1] % 3
    while len(trim(a)) >= len(b):
        s = len(a) - len(b); q = a[-1] * inv % 3
        for i, bc in enumerate(b): a[s + i] = (a[s + i] - q * bc) % 3
    return a
def gcd(a, b):
    a, b = trim(a[:]), trim(b[:])
    while b: a, b = b, trim(pmod(a, b))
    return a
for j in range(0, 10):
    n = 2 ** (j + 1)
    f = [0] * (n + 1); f[0] = 2; f[n // 2] = (f[n // 2] + 1) % 3; f[n] = 1
    c = [0] * n; c[1] = 1
    half = None
    for k in range(1, n + 1):
        c = frob(c, n)
        if k == n // 2: half = c[:]
    xpoly = [0] * n; xpoly[1] = 1
    h = half[:]; h[1] = (h[1] - 1) % 3
    print(n, c == xpoly and len(gcd(h, f)) == 1, flush=True)
