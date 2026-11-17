# Thicker rectangular matrix multiplication with polylogarithmic overhead over fields of characteristic 2 and 3

This repository holds the paper (in `paper/`), the scripts behind every computer check in it, and the outputs of a
fresh run of each script. The section, theorem, lemma and table numbers below are those of the paper.

Most of the checks are those of Section 10 ("Implementation and crossover"): an instrumented implementation of each
algorithm, whose counted field operations are compared with the exact counts of Tables 2, 3 and 5 and whose results are
compared with direct multiplication, a second implementation written independently from the text of the paper, and
the crossover computation. The others support single statements elsewhere: the decomposition of T6 (Lemma 6.1 and
Table 4), the evaluations in the proofs of Lemma 4.2, of Theorem 1.1(b),(c) and of Lemma 6.6, the constants of
Theorem 1.2, Remark 6.7, the example field of Corollary 1.3(ii), and the case list in the proof of Corollary A.2.

## Requirements

Python 3 with numpy 2.0 or later (the scripts use `numpy.bitwise_count`); nothing else. The outputs in `outputs/`
were produced with Python 3.13.5 and numpy 2.4.4 on Linux. Every run stays below 3 GB of virtual memory (all were
run under `ulimit -v 3000000`).

## Layout and how to run

- `paper/`: the paper, `main.tex` and `main.pdf`; `pdflatex main.tex` run three times builds it, with no other files.
- `scripts/`: the scripts. Run each from this directory, for example `cd scripts && python3 char2_impl.py 8 2 F2`.
  A script either writes its output file into `../outputs/` itself or prints to standard output; the commands
  below redirect the latter into `../outputs/` as well.
- `data/T6_decomposition.json`: the six rank-one terms of the decomposition of T6 over F_3, its 30 error monomials
  and its grading (Lemma 6.1). The scripts `t6_printed_forms.py`, `indep6_decomposition.py`,
  `indep6_separation_dp.py` and `indep6_entry.py` read it.
- `outputs/`: the output of every command below, from the run made for this repository.
- `run_all.sh`: runs every command below. `JOBS=3 ./run_all.sh` runs three at a time; the whole set takes about
  3.0 CPU hours, and the longest single command about 46 minutes.

The random inputs use fixed seeds, so a rerun reproduces the outputs exactly, except for the timings they print and,
in the second line of the `char3_impl.py` outputs ("plus-sign form minus (T4 + E3) = {...}"), the order of the
entries of a dictionary, which depends on Python's string hashing.

## Provenance of the outputs

Every file in `outputs/` was produced by running the staged copy of its script with the command listed below
(all 65 commands, on 17 November 2026). Each was compared with the output of the original run of the same computation, made while the paper
was written: after the renaming of printed labels described in the docstrings (for example the name of the tensor) and
with the timings blanked, every output is identical to the original one, except that the original outputs of
`char3_impl.py` list the entries of one dictionary in a different order (see above) and that there is no original
output of `irreducible_trinomials.py` to compare with.

## Index: statement of the paper and the scripts that check it

| statement | scripts |
|---|---|
| inequality (1), Section 2 | `char2_count.py`, `indep2_count.py` |
| Lemma 2.1 (decomposition of T4) | `indep2_lemmas.py` (mod 2), `indep3_lemmas.py` (signed, over Z), start-up line of `char3_impl.py` |
| Lemma 2.3, Lemma 3.1, Proposition 3.2, Lemma 3.3 | `indep2_lemmas.py`; Proposition 3.2 also `indep6_identity.py`; Lemma 3.3 also `indep3_covering.py` |
| Table 2 (exact count, characteristic 2) | `char2_impl.py`, `indep2_impl.py`, `indep2_count.py`; `t6_impl.py` on T4 |
| Lemma 4.2 (exact totals at L = 4, 8; at most 2L^4) and the paragraph after it | `char2_count.py`, `indep2_count.py` |
| Remark 4.3(iv) (16, 21, 46, 55 relabelings) | `char2_impl.py` |
| Proposition 5.1 | `indep3_lemmas.py`, `char3_impl.py` |
| Lemma 5.2, Corollary 5.3 | `char3_isolation.py`, `indep3_isolation.py` |
| Table 3, proof of Theorem 1.1(b),(c) (1.6175 L^4; set-up bound), Remark 5.4, Δ < L^3/128 | `char3_count.py`, `indep3_count.py` |
| Corollary 1.3(ii) (the example field) | `irreducible_trinomials.py` |
| Lemma 6.1 and Table 4 (decomposition of T6) | `t6_printed_forms.py`, `indep6_decomposition.py` |
| Lemma 6.2 (separation) | `indep6_separation_dp.py`, `indep6_entry.py`, `t6_impl.py sep:...` |
| Proposition 6.3 (reading of T6) | `indep6_identity.py`, `t6_reading_check.py`, `indep6_entry.py` |
| Lemma 6.4 (covering) | `indep6_cover.py`, `t6_cover_check.py` |
| Table 5, Lemma 6.6 and the paragraph after it | `t6_impl.py`, `t6_formula_xcheck.py`, `t6_counts.py`, `indep6_counts.py` |
| constants of Theorem 1.2, Remark 6.7(i) | `indep6_counts.py` |
| Remark 6.7(iii) (random covering) | `indep6_lasvegas.py`, `t6_counts.py` |
| Section 10, characteristic two | `char2_impl.py`, `indep2_impl.py`, `indep2_count.py`, `indep2_lemmas.py` |
| Section 10, characteristic three | `char3_impl.py`, `indep3_impl.py`, `indep3_count.py` |
| Section 10, the tensor T6 | `t6_impl.py`, `indep6_entry.py`, `indep6_separation_dp.py`, `indep6_identity.py`, `indep6_cover.py`, `indep6_counts.py` |
| Section 10, crossover | `char2_count.py` |
| proof of Corollary A.2 (the four cases) | `one_high_cell_cases.py` |

## Two implementations

The scripts `char2_*.py`, `char3_*.py` and `t6_*.py`, with the field modules `gf2k.py` and `gf3k.py`, form the first
implementation. The scripts `indep2_*.py`, `indep3_*.py` and `indep6_*.py` form a second one, written from the text
of the paper alone; they share no code with the first (they have their own fields, algebras, relabelings and strings).
Section 10 refers to them as the second, independent implementation and the independent check.

## Terms used in the scripts and outputs

- *tile-run*: one run of the recursion of Lemma 2.3 for one pair of bands and one relabeling (steps 2 to 4 of the
  algorithm of Section 4).
- *class outputs*: all outputs of a tile-run at a label pair whose intersection has the size d of the reading (2d in
  characteristic 3), not only those that the read-out uses. The implementations check every one of them.
- *schedule all* / *first* (or *used*): all relabelings of the family Pi_L, or only those that are the first covering
  relabeling of some label pair (Remark 4.3(iv)).
- *P3*: the four-term decomposition of T4 in Lemma 2.1. The control *char2P3* (or *plus*) uses it with all signs +,
  which is correct only in characteristic 2.
- *(P1), (P2), (P3)* in the output of `t6_impl.py`: the properties of a base tensor listed in the docstring of that
  script (those of Section 7, "What a better tensor needs").
- *Pi^d*, the *diagonal path*: the family of relabelings of Section 6 (before Lemma 6.4).
- *trivial*: the textbook algorithm.
- *Z*, *K_z*: the list and the number of form parameters z in characteristic 3 (Corollary 5.3); *bad S*: an inner
  set S with c_z(S) = 0.

## The checks

Run times are those of the run made for this repository (three runs at a time, at low priority, on a desktop
computer); they are wall-clock times of single runs. The run of `indep6_cover.py` was paused for about 80 minutes by the operator, so its wall-clock time is not a run time.

### Characteristic two: Sections 2 to 4, Theorem 1.1(a), Section 10

**`char2_impl.py`**: the algorithm of Section 4, computing in GF(16) with nodes t_l = l + 1, with every field
operation counted as executed. It supports the first paragraph of Section 10: the counts equal Table 2 term by term,
all entries are correct at L = 6, 8, 10 with two bands per side (1600, 19600 and 254016 entries; for L not divisible
by 4 it takes d = floor(L/4)) and at L = 12 with one and with two bands per side (853776 and 3415104 entries, all 132
relabelings), and every class output of every tile-run is correct. The control without the division by det(Phi_S)
makes 39% to 84% of the entries wrong when d ≥ 2 (7643 and 16395 of 19600 at L = 8 with inputs over F_2 and over
GF(16), 104291 and 211633 of 254016 at L = 10, 411306 of 853776 and 1654617 of 3415104 at L = 12). The parts of the
formula printed for L = 12 give the shares of Section 10: decoding 75%, encodings 16%, leaf products 8%. The line
"first-covering relabelings: k distinct" gives the numbers 16, 21, 46, 55 of Remark 4.3(iv) at L = 6, 8, 10, 12; they
are for the factorization of the cyclic shift used here, into the transpositions (1 L), (1 L-1), ..., (1 2) (levels
numbered from 1); with the adjacent transpositions, L = 8 gives 19 (see `indep2_lemmas.py`).

| command | output | time | expected |
|---|---|---|---|
| `python3 char2_impl.py 6 2 F2` | `char2_impl_L6B2F2_out.txt` | < 1 s | L = 6, 1600 entries; d = 1, so the control is 0 wrong (det = 1) |
| `python3 char2_impl.py 6 2 GF16` | `char2_impl_L6B2GF16_out.txt` | < 1 s | the same, inputs over GF(16) |
| `python3 char2_impl.py 8 2 F2` | `char2_impl_L8B2F2_out.txt` | < 1 s | L = 8, 19600 entries; control 7643 wrong |
| `python3 char2_impl.py 8 2 GF16` | `char2_impl_L8B2GF16_out.txt` | < 1 s | inputs over GF(16); control 16395 wrong |
| `python3 char2_impl.py 10 2 F2` | `char2_impl_L10B2F2_out.txt` | 7 s | L = 10, 254016 entries; control 104291 wrong |
| `python3 char2_impl.py 10 2 GF16` | `char2_impl_L10B2GF16_out.txt` | 7 s | inputs over GF(16); control 211633 wrong |
| `python3 char2_impl.py 12 1 F2` | `char2_impl_L12B1F2_out.txt` | 88 s | L = 12, one band per side, 853776 entries; control 411306 wrong |
| `python3 char2_impl.py 12 2 F2 2` | `char2_impl_L12B2F2_out.txt` | 4 min | L = 12, two bands per side (seed 2), 3415104 entries; control 1654617 wrong |

In each output: "wrong entries of the assembled product: 0", "every class output of every tile-run checked: ... 0
wrong", "equal: True" for both schedules, and the control line.

**`char2_count.py`**: the exact count of Table 2 as a function of L and B. It gives the exact totals per entry in the
proof of Lemma 4.2 (475.444 = 1.857 L^4 at L = 4 and 6631.45 = 1.619 L^4 at L = 8, column `perentry_B1`), the
maximum 1.8572 of the cost per entry divided by L^4 over all even L ≤ 256, inequality (1) of Section 2 for all even
L ≤ 398, the asymptotic constant 3π/8 = 1.178 of the paragraph after Lemma 4.2, and the crossover of Section 10: the
algorithm is cheaper than the textbook algorithm from L = 108 on (D = 2^27, N = C(108, 54) = 2^104.3), cheaper than
the Four Russians method from L = 140 on (N = 2^136.1), and with an ideal covering schedule (1/f runs per band pair)
the crossover with the textbook algorithm is at L = 52 (N = 2^48.8). The word-packed rows are an additional model
not used in the paper.

| command | output | time |
|---|---|---|
| `python3 char2_count.py` | `char2_count_out.txt` | < 1 s |

**`indep2_impl.py`** (with `indep2_gf.py`, `indep2_core.py`): the second implementation of Section 4, with random
distinct nodes in the smallest field E with at least L elements. It supports the sentence of Section 10 that a second
implementation reproduced the counts and the results up to L = 12. Each case prints the number of wrong entries
(0), the wrong class outputs (0), the control without division, and whether the counted operations equal Table 2
("equal to Table 2 with K = runs: True"). The case string is N,m,Fn,seed,relab,schedule,classcheck (see the script).

| command | output | time |
|---|---|---|
| `python3 indep2_impl.py ../outputs/indep2_impl_L4_out.txt 256,2,1,1,adjacent,all,1 256,2,2,2,star,all,1 300,2,4,3,adjacent,all,1` | `indep2_impl_L4_out.txt` | 2 s |
| `python3 indep2_impl.py ../outputs/indep2_impl_L8_out.txt 1296,3,1,11,adjacent,all,1 1296,3,2,12,star,all,1 1296,4,4,13,adjacent,all,1` | `indep2_impl_L8_out.txt` | 96 s |
| `python3 indep2_impl.py ../outputs/indep2_impl_L8m4_out.txt 4096,4,4,21,star,all,1` | `indep2_impl_L8m4_out.txt` | 5 min |
| `python3 indep2_impl.py ../outputs/indep2_impl_L12_out.txt 1000,7,1,31,adjacent,used,1` | `indep2_impl_L12_out.txt` | 2 min |
| `python3 indep2_impl.py ../outputs/indep2_impl_L12all_out.txt 1000,5,4,41,star,all,1` | `indep2_impl_L12all_out.txt` | 5 min |

**`indep2_count.py`**: Table 2 re-derived from the steps. It checks that the cost per padded entry is at most 2L^4 for
L = 4, 8, ..., 256 (maximum 1.85720 L^4 at L = 4; Lemma 4.2), that it decreases in the number of bands B (so the
count of Lemma 4.2 holds for every number of bands, as Section 10 uses), that the essential multiplications per entry
stay below 2L^3 (Theorem 1.1(a)), inequality (1) for even L ≤ 512, and the parameter claims of Section 4
(2^q < 2m, N0 < N, L ≤ log2 N) for m ≤ 4096 at the smallest admissible N. It also reports that D = 2^floor(L/4) is
below N0^{1/4} at L = 2 and 6; the theorem uses only L divisible by 4.

| command | output | time |
|---|---|---|
| `python3 indep2_count.py ../outputs/indep2_count_out.txt` | `indep2_count_out.txt` | < 1 s |

**`indep2_lemmas.py`**: brute-force checks of Lemma 2.1 read mod 2, Lemma 2.3 and Lemma 3.1 at L = 3, 4, 5 (with a
control that breaks the equal-size condition), Proposition 3.2 at L = 8 for d = 1, ..., 4 over GF(16) (with a control
over Z, where the identity fails), and Lemma 3.3 for L = 4, ..., 12 and two factorizations of the cyclic shift.

| command | output | time |
|---|---|---|
| `python3 indep2_lemmas.py ../outputs/indep2_lemmas_out.txt` | `indep2_lemmas_out.txt` | 8 s |

### Characteristic three with T4: Section 5, Theorem 1.1(b),(c), Section 10

**`char3_impl.py`** (with `gf3k.py`): the algorithm of Section 5, with the signed decomposition of Lemma 2.1, the
signed pairing, the forms of Lemma 5.2 and the list Z of Corollary 5.3. Rows and columns may be labelled by sets of
different sizes a and b; the inner sets then have size floor(ab/L) = 2d. It supports the sentence of Section 10 that
the characteristic-three algorithm is exact at L = 8 and, with a ≠ b, at L = 10 and 12, with inner sets where
c_z(S) = 0 present: at L = 10 the single value z = g^8 is bad on 30 of the 210 inner sets and still all entries are
correct, because every label pair meets in 2d levels under several relabelings; with z = g^0 = 1 every inner set is
bad (c(S) = (2d)!/2^d ≡ 0 mod 3, Section 5), so no entry can be computed. Each output prints the wrong entries (0
except in the controls), the class outputs at good and at bad inner sets (at bad S the raw value must be 0,
Proposition 5.1), "equal: True" for the counted operations against the formula, and the control without division.

| command | output | time | expected |
|---|---|---|---|
| `python3 char3_impl.py 8 4 4 2 F3 2 greedy all` | `char3_impl_L8a4b4B2F3k2greedyall_out.txt` | 2 s | L = 8, a = b = 4, d = 1, inputs over F_3: 19600 entries, 0 wrong |
| `python3 char3_impl.py 8 4 4 2 E 2 greedy all` | `char3_impl_L8a4b4B2Ek2greedyall_out.txt` | 2 s | the same, inputs over GF(9) |
| `python3 char3_impl.py 8 4 4 2 F3 2 greedy first unsigned` | `char3_impl_L8a4b4B2F3k2greedyfirstunsigned_out.txt` | < 1 s | control (right pairing without signs): 8930 of 19600 wrong |
| `python3 char3_impl.py 8 4 4 2 F3 2 greedy first char2P3` | `char3_impl_L8a4b4B2F3k2greedyfirstchar2P3_out.txt` | < 1 s | control (plus signs): 13032 wrong |
| `python3 char3_impl.py 10 5 8 2 F3 4 greedy all` | `char3_impl_L10a5b8B2F3k4greedyall_out.txt` | 113 s | L = 10, a = 5, b = 8, d = 2: 45360 entries, 0 wrong; no division: 29930 wrong |
| `python3 char3_impl.py 10 5 8 2 E 4 explicit first` | `char3_impl_L10a5b8B2Ek4explicitfirst_out.txt` | 46 s | explicit list Z = (g^0, ..., g^16), inputs over GF(81): 0 wrong |
| `python3 char3_impl.py 10 5 8 2 F3 4 single:8 first` | `char3_impl_L10a5b8B2F3k4single8first_out.txt` | 47 s | z = g^8 alone, bad on 30 of 210 inner sets: 0 wrong |
| `python3 char3_impl.py 10 5 8 2 F3 4 single:0 first` | `char3_impl_L10a5b8B2F3k4single0first_out.txt` | < 1 s | z = 1 alone, bad on every inner set: no entry computed (control) |
| `python3 char3_impl.py 10 5 8 2 F3 4 greedy first unsigned` | `char3_impl_L10a5b8B2F3k4greedyfirstunsigned_out.txt` | 43 s | control (right pairing without signs): 29278 wrong |
| `python3 char3_impl.py 10 5 8 2 F3 4 greedy first char2P3` | `char3_impl_L10a5b8B2F3k4greedyfirstchar2P3_out.txt` | 41 s | control (plus signs): 44807 wrong |
| `python3 char3_impl.py 12 6 8 2 F3 4 greedy all` | `char3_impl_L12a6b8B2F3k4greedyall_out.txt` | 46 min | L = 12, a = 6, b = 8, d = 2, two bands per side: 1829520 entries, 0 wrong |
| `python3 char3_impl.py 12 8 9 1 E 5 greedy all` | `char3_impl_L12a8b9B1Ek5greedyall_out.txt` | 18 min | L = 12, a = 8, b = 9, d = 3, inputs over GF(243): 108900 entries, 0 wrong |
| `python3 char3_impl.py 12 8 9 2 F3 5 greedy first` | `char3_impl_L12a8b9B2F3k5greedyfirst_out.txt` | 28 min | the same shape, two bands per side, inputs over F_3: 435600 entries, 0 wrong |

**`char3_count.py`**: the exact count at the shape of Theorem 1.1(b),(c) (L = 8d, a = b = L/2). The maximum of the
cost per padded entry divided by L^4 over L = 8d ≤ 256 is 1.61741, at L = 8: the constant 1.6175 of the proof of
Theorem 1.1(b),(c), from Tables 2 and 3. It also checks its closed forms against brute force and against the formula of `char3_impl.py`.

| command | output | time |
|---|---|---|
| `python3 char3_count.py` | `char3_count_out.txt` | < 1 s |

**`char3_isolation.py`**: Lemma 5.2 on every inner set (2d-set of levels) for (L, d) = (8, 1), (12, 2), (16, 2),
(12, 3), (16, 3), (24, 3), (20, 4) (part A: the lowest coefficient of c_S(z) is 1, at the predicted W_min, and
deg g_S ≤ (L-1) floor(d^2/2)), and Corollary 5.3 in GF(3^k) (part B: bad values per set at most deg g_S, greedy
hitting sets within the bound log2 C(L, 2d) + 1).

| command | output | time |
|---|---|---|
| `python3 char3_isolation.py` | `char3_isolation_out.txt` | 35 s |

**`indep3_impl.py`** (with `indep3_gf.py`): the second implementation of Section 5. It supports the sentence of
Section 10 that a second implementation from the text was exact at L = 8, at L = 9 with label sizes 6 and 6
(a^2/L = 4 = 2d) and at L = 11 with label sizes 8 and 9. Several runs use lists Z that contain values bad on many
inner sets; the controls (unsigned right pairing, plus-sign decomposition, no division) must give wrong entries.

| command | output | time | expected |
|---|---|---|---|
| `python3 indep3_impl.py 8 4 4 1 1 F3 all none` | `indep3_impl_L8a4b4d1k1F3allnone_out.txt` | < 1 s | L = 8, d = 1: 4900 entries, 0 wrong |
| `python3 indep3_impl.py 8 4 4 1 1 F3 first unsigned` | `indep3_impl_L8a4b4d1k1F3firstunsigned_out.txt` | < 1 s | control (unsigned pairing): 2640 wrong |
| `python3 indep3_impl.py 8 4 4 1 1 F3 first plus` | `indep3_impl_L8a4b4d1k1F3firstplus_out.txt` | < 1 s | control (plus signs): 3315 wrong |
| `python3 indep3_impl.py 9 6 6 2 4 F3 all none` | `indep3_impl_L9a6b6d2k4F3allnone_out.txt` | 3 s | L = 9, a = b = 6, d = 2, greedy Z: 7056 entries, 0 wrong |
| `python3 indep3_impl.py 9 6 6 2 4 E all none 1,2,15,4` | `indep3_impl_L9a6b6d2k4Eallnonez1_2_15_4_out.txt` | 10 s | Z = (1, 2, 15, 4), bad on 126, 66, 30, 0 of the 126 inner sets: 0 wrong |
| `python3 indep3_impl.py 9 6 6 2 4 E first none 2,15` | `indep3_impl_L9a6b6d2k4Efirstnonez2_15_out.txt` | 3 s | Z = (2, 15), neither value good on all inner sets: 0 wrong |
| `python3 indep3_impl.py 9 6 6 2 4 E first unsigned 2,15,4` | `indep3_impl_L9a6b6d2k4Efirstunsignedz2_15_4_out.txt` | 3 s | control (unsigned pairing): 6962 wrong |
| `python3 indep3_impl.py 9 6 6 2 4 E first plus 2,15,4` | `indep3_impl_L9a6b6d2k4Efirstplusz2_15_4_out.txt` | 3 s | control (plus signs): 6970 wrong |
| `python3 indep3_impl.py 9 6 6 2 4 E first nodiv 2,15,4` | `indep3_impl_L9a6b6d2k4Efirstnodivz2_15_4_out.txt` | 3 s | control (no division): 248 wrong |
| `python3 indep3_impl.py 11 8 9 3 4 E first none` | `indep3_impl_L11a8b9d3k4Efirstnone_out.txt` | 22 s | L = 11, a = 8, b = 9, d = 3: 9075 entries, 0 wrong |
| `python3 indep3_impl.py 11 8 9 3 4 F3 all none 2,4` | `indep3_impl_L11a8b9d3k4F3allnonez2_4_out.txt` | 2 min | the same shape, Z = (2, 4) with z = 2 bad on all 462 inner sets: 0 wrong |

**`indep3_count.py`**: the cost of Section 5 re-derived ("reproduced the count", Section 10): 1.61741 L^4 at L = 8 as
the maximum over L = 8d ≤ 256, the bounds of each part used in the proof for L = 264, ..., 1024, the set-up cost
against the bound L·2^{1.01L} of the proof of Theorem 1.1(c), and for d ≤ 299: log2 C(L, L/4) ≤ 0.8113 L and
|Z| ≤ L (Corollary 5.3(i)), Δ < L^3/128 (Theorem 1.1), the field size |E| < 3(2Δ + 1) for F = F_3 (proof of
Corollary 1.3 in characteristic 3), and the expected number of rounds of Remark 5.4. The block "exhaustive candidate
search" evaluates a bound that the paper does not use.

| command | output | time |
|---|---|---|
| `python3 indep3_count.py` | `indep3_count_out.txt` | 2 s |

**`indep3_lemmas.py`**, **`indep3_isolation.py`**, **`indep3_covering.py`**: independent checks of the signed form of
Lemma 2.1 over Z, of Lemmas 2.3 and 3.1 with the signed decomposition, and of Proposition 5.1 over GF(3^k); of
Lemma 5.2 and Corollary 5.3 (c_S(z) computed as an integer polynomial by a dynamic programme); and of Lemma 3.3 at the
shapes L = 8d, a = L/2 with target 2d (exhaustive at L = 8, sampled at L = 16, 24).

| command | output | time |
|---|---|---|
| `python3 indep3_lemmas.py` | `indep3_lemmas_out.txt` | 3 s |
| `python3 indep3_isolation.py` | `indep3_isolation_out.txt` | 41 s |
| `python3 indep3_covering.py` | `indep3_covering_out.txt` | < 1 s |

### The six-term tensor T6: Section 6, Theorem 1.2, Section 10

**`t6_printed_forms.py`**: the forms of Lemma 6.1, typed as printed, expanded over F_3: the product has 38 monomials,
the 8 of T6 with coefficient 1 and the 30 of E6, all of weight at least 1; the script prints E6 by class with signs
and weights, which is Table 4, and compares the expansion and the grading with the data file.

| command | output | time |
|---|---|---|
| `python3 t6_printed_forms.py > ../outputs/t6_printed_forms_out.txt` | `t6_printed_forms_out.txt` | < 1 s |

**`indep6_decomposition.py`**: an independent expansion of the terms in the data file: T6 + E with the eight
monomials of T6 at coefficient 1, weight 0 on T6 and at least 1 on E, the grading constant on the blocks,
h(z_ij) = -h_A(i) - h_B(j), the same for the transpose T6' (Lemma 6.1), and the terms hard-coded in `t6_impl.py`
equal to those of the data file.

| command | output | time |
|---|---|---|
| `python3 indep6_decomposition.py > ../outputs/indep6_decomposition_out.txt` | `indep6_decomposition_out.txt` | < 1 s |

**`t6_impl.py`**: the algorithm of Section 6 for a base tensor given as a decomposition with a grading, over GF(9)
and GF(27). It supports the second paragraph of Section 10. Configurations: `thm_d1`, the parameters of Theorem 1.2
with d = 1 (L = 6, two bands per side); `sym_d2` and `sym_d3`, four and five levels of each kind (d = 2, 3); `rect_d2`
and `rect_d3`, T6 alone on seven and on nine levels with rectangular bands (d = 2, 3; 63504 entries at d = 3);
`t4_L8`, `t4_L10`, `t4_sym`, the same code on T4 in characteristic 2, which reproduces Table 2. Each configuration
prints the properties (P1) to (P3) of its tensors, the covering, the wrong entries (0), the other outputs at |S| = d
of every run (0 wrong), the counted operations against the formula step by step (EQUAL), and the control without
division, which makes 63% to 77% of the entries wrong when d ≥ 2 (2304 of 3675 for rect_d2, 15043 of 20736 for
sym_d2, 41229 of 63504 for rect_d3, 30965 of 40000 for sym_d3). `ctrl:sym_d2` is the wrong-pairing control, and
`sep:...` the separation check of Lemma 6.2 on random arrays: no difference at the outputs of the band types, many
outside them.

| command | output | time |
|---|---|---|
| `python3 t6_impl.py thm_d1 rect_d2 t4_L8 t4_sym t4_L10 > ../outputs/t6_impl_small_out.txt` | `t6_impl_small_out.txt` | 2 min |
| `python3 t6_impl.py sym_d2 ctrl:sym_d2 > ../outputs/t6_impl_sym_d2_out.txt` | `t6_impl_sym_d2_out.txt` | 3 min |
| `python3 t6_impl.py rect_d3 > ../outputs/t6_impl_rect_d3_out.txt` | `t6_impl_rect_d3_out.txt` | 10 min |
| `python3 t6_impl.py sym_d3 > ../outputs/t6_impl_sym_d3_out.txt` | `t6_impl_sym_d3_out.txt` | 40 min |
| `python3 t6_impl.py sep:thm_d1 sep:rect_d2 sep:t4_L8 sep:sym_d2 > ../outputs/t6_impl_sep_out.txt` | `t6_impl_sep_out.txt` | 6 s |

**`t6_counts.py`**, **`t6_formula_xcheck.py`**: Table 5 evaluated at the parameters of Theorem 1.2 for d ≤ 30 and
d = 40, 50, 60 (per entry 0.9601 L^6 at d = 1 and at most 0.63 L^6 from d = 2 on, essential multiplications 2/9 L^5 at
d = 1, the share of the decoding, Lemma 6.6 and the paragraph after it; and the hitting fraction f of Remark 6.7(iii));
and the agreement of Table 5 as printed with the formula that `t6_impl.py` compares with its executed counts, for
d = 1, 2, 3 and B = 1, 2.

| command | output | time |
|---|---|---|
| `python3 t6_counts.py > ../outputs/t6_counts_out.txt` | `t6_counts_out.txt` | 7 s |
| `python3 t6_formula_xcheck.py > ../outputs/t6_formula_xcheck_out.txt` | `t6_formula_xcheck_out.txt` | < 1 s |

**`t6_reading_check.py`**, **`t6_cover_check.py`**: Proposition 6.3 checked directly in V ⊗ V over GF(81) for
d = 2, ..., 5 (58 cases, 4 of them with singular Phi_S; the control with the slice signs (1, -1, 1) fails), and
Lemma 6.4 at the parameters of Theorem 1.2 (exhaustive at d = 2, 200000 random quadruples at d = 3, 4).

| command | output | time |
|---|---|---|
| `python3 t6_reading_check.py > ../outputs/t6_reading_check_out.txt` | `t6_reading_check_out.txt` | 4 s |
| `python3 t6_cover_check.py > ../outputs/t6_cover_check_out.txt` | `t6_cover_check_out.txt` | 25 s |

**The independent check of Section 10** (`indep6_*.py`): `indep6_entry.py` evaluates Mult from its definition, with
the error terms, at 77 outputs read out at the exact parameters of Theorem 1.2 for d = 2, 3, 4 (L = 12, 18, 24): the
nonzero terms are exactly the pairs of Lemma 6.2 and the values are det(Phi_S)^2 Σ_T x_T y_T (Proposition 6.3), and
with the wrong pairing about 1/27 of the values come out right by chance. `indep6_separation_dp.py` confirms Lemma
6.2 exactly for d ≤ 12 by a dynamic programme over the numbers of levels in each class (its two controls must find
violations). `indep6_identity.py` proves the identity behind Proposition 6.3 as a polynomial identity in a generic
d × d matrix for d ≤ 4 (also Proposition 3.2 and the form with q a power of p mentioned in Section 7).
`indep6_cover.py` checks Lemma 6.4 exhaustively for d ≤ 3 (and on random quadruples for d = 4, 5). `indep6_counts.py`
evaluates Table 5 for every d ≤ 150 (Lemma 6.6: maximum 0.96006 L^6 at d = 1, 0.62994 L^6 for d ≥ 2, essential
multiplications at most 0.22222 L^5; decoding above 97% from d = 9 on), checks the inequalities of the proof of Lemma
6.6 for 31 ≤ d ≤ 150, and prints the constants of Theorem 1.2 (2/log2 6 = 0.774 < 0.78, 256/log2^6 6 = 0.857 < 0.86,
32/log2^5 6 = 0.277 < 0.28), 5π^3/324 = 0.478 (the paragraph after Lemma 6.6) and 3^{1/α6} = 216 (Remark 6.7(i)).
`indep6_lasvegas.py` gives Remark 6.7(iii): f√L between 1.49 and 1.69 at the twelve values of d from 2 to 64 in its
table, and K_r = 36 against K = 60 at d = 2, 2608 against 17112 at d = 31.

| command | output | time |
|---|---|---|
| `python3 indep6_entry.py > ../outputs/indep6_entry_out.txt` | `indep6_entry_out.txt` | 3 min |
| `python3 indep6_separation_dp.py 12 > ../outputs/indep6_separation_dp_out.txt` | `indep6_separation_dp_out.txt` | 1 s |
| `python3 indep6_identity.py > ../outputs/indep6_identity_out.txt` | `indep6_identity_out.txt` | 1 s |
| `python3 indep6_counts.py 150 > ../outputs/indep6_counts_out.txt` | `indep6_counts_out.txt` | 17 s |
| `python3 indep6_cover.py > ../outputs/indep6_cover_out.txt` | `indep6_cover_out.txt` | not measured (run paused) |
| `python3 indep6_lasvegas.py > ../outputs/indep6_lasvegas_out.txt` | `indep6_lasvegas_out.txt` | < 1 s |

### Other statements

**`irreducible_trinomials.py`**: the example field of Corollary 1.3(ii). The paper proves that x^n + x^{n/2} + 2 is
irreducible over F_3 for every power of two n ≥ 2 by [LN97, Theorem 3.35]; the script confirms it by Rabin's test
for n = 2, 4, ..., 1024 (every line ends in True).

**`one_high_cell_cases.py`**: the case list in the proof of Corollary A.2. Among grid tensors with one high cell of
rank q and n outputs, the letter counts and tightness leave only (n, q) = (3, 2), (4, 3), (5, 3), (6, 4) with
α(T) > 1/4, and in every case one side is saturated with at most three cells per row or column, so Theorem A.1
applies ("ALL covered by Theorem A.1" on each of the four lines).

| command | output | time |
|---|---|---|
| `python3 irreducible_trinomials.py > ../outputs/irreducible_trinomials_out.txt` | `irreducible_trinomials_out.txt` | < 1 s |
| `python3 one_high_cell_cases.py > ../outputs/one_high_cell_cases_out.txt` | `one_high_cell_cases_out.txt` | < 1 s |

## License

The scripts, `run_all.sh`, the data file and the outputs are under the MIT License (`LICENSE`). The paper in
`paper/`, its LaTeX source and its PDF, is under the Creative Commons Attribution 4.0 International License
(CC BY 4.0, `paper/LICENSE`).
