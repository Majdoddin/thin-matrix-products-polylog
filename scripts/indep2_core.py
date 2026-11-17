r"""Second implementation of Section 4: core pieces (the matrices of the decomposition, axis-by-axis transforms,
string indexing).

Letters: left x0, x1, p0 = 0, 1, 2; right y0, y1, q0 = 0, 1, 2; outputs z0, z1, z2, w0 = 0, 1, 2, 3.
Strings are flat indices sum_l digit_l * base^l (level l <-> digit of base^l).
The decomposition of Lemma 2.1 read in characteristic 2 (signs dropped): 1: (x0+x1)(y0)(z0+w0); 2: (x0+p0)(y1)(z1+w0);
3: (x1)(y0+q0)(z2+w0); 4: (x0)(y0+y1)(w0).
"""
import numpy as np

# rows = terms 1..4, entry = list of letters summed
PHI_ROWS = [[0, 1], [0, 2], [1], [0]]
PSI_ROWS = [[0], [1], [0, 2], [0, 1]]
# decoding: output letter <- terms (CHI^T rows): z0 <- 1, z1 <- 2, z2 <- 3, w0 <- 1+2+3+4
CHIT_ROWS = [[0], [1], [2], [0, 1, 2, 3]]

# tensors as 0/1 integer arrays
def gamma_T4_E3():
    g = np.zeros((3, 3, 4), dtype=np.int64)
    for s, t, z in [(0, 0, 0), (0, 1, 1), (1, 0, 2), (1, 2, 3), (2, 1, 3)]:  # T4
        g[s, t, z] += 1
    for s, t, z in [(1, 0, 0), (2, 1, 1), (1, 2, 2)]:  # E3
        g[s, t, z] += 1
    return g

def p3_tensor_over_Z():
    PHI = np.zeros((4, 3), dtype=np.int64); PSI = np.zeros((4, 3), dtype=np.int64)
    CHI = np.zeros((4, 4), dtype=np.int64)
    for lam in range(4):
        for s in PHI_ROWS[lam]:
            PHI[lam, s] = 1
        for t in PSI_ROWS[lam]:
            PSI[lam, t] = 1
    for z, terms in enumerate(CHIT_ROWS):
        for lam in terms:
            CHI[lam, z] = 1
    return np.einsum('ls,lt,lz->stz', PHI, PSI, CHI), PHI, PSI, CHI


def axis_transform(A, L, nin, rows):
    """Apply M^{(x)L} to each row of A (shape (batch, nin^L)), M given by `rows` (output letter ->
    list of input letters XORed). Returns (result (batch, nout^L), additions counted per fibre)."""
    batch = A.shape[0]
    nout = len(rows)
    cur = A.reshape(batch, 1, nin ** L)
    pre = 1
    adds = 0
    for k in range(L):
        post = nin ** (L - 1 - k)
        cur = cur.reshape(batch, pre, nin, post)
        new = np.empty((batch, pre, nout, post), dtype=A.dtype)
        for o, letters in enumerate(rows):
            acc = cur[:, :, letters[0], :].copy()
            for s in letters[1:]:
                acc ^= cur[:, :, s, :]
            new[:, :, o, :] = acc
            adds += (len(letters) - 1) * pre * post * batch
        cur = new
        pre *= nout
    return cur.reshape(batch, nout ** L), adds


def left_index(r, P, L):
    """string (r, P): p0 on P, x1 on r minus P, x0 elsewhere."""
    idx = 0
    for l in range(L):
        dgt = 2 if (P >> l) & 1 else (1 if (r >> l) & 1 else 0)
        idx += dgt * 3 ** l
    return idx

def right_index(c, Q, L):
    """string (c, Q): q0 on Q, y1 on c minus Q, y0 elsewhere."""
    idx = 0
    for l in range(L):
        dgt = 2 if (Q >> l) & 1 else (1 if (c >> l) & 1 else 0)
        idx += dgt * 3 ** l
    return idx

def out_index(r, c, L):
    """output (r, c): w0 on r & c, z2 on r - c, z1 on c - r, z0 elsewhere."""
    idx = 0
    for l in range(L):
        inr, inc = (r >> l) & 1, (c >> l) & 1
        dgt = 3 if (inr and inc) else (2 if inr else (1 if inc else 0))
        idx += dgt * 4 ** l
    return idx

def subsets(mask):
    """all submasks of mask"""
    s = mask
    out = []
    while True:
        out.append(s)
        if s == 0:
            break
        s = (s - 1) & mask
    return out

def popcount(x):
    return bin(x).count('1')
