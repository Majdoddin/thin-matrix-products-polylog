#!/bin/bash
# Runs every check of the package. Each command is run from scripts/ and writes its output to outputs/.
# JOBS=3 ./run_all.sh runs three commands at a time (default: one). The full set takes about 3 CPU hours; the
# five longest commands come first. Each run stays below 3 GB of virtual memory.
cd "$(dirname "$0")/scripts" || exit 1
JOBS=${JOBS:-1}
export PYTHONDONTWRITEBYTECODE=1
grep -v '^#' <<'COMMANDS' | grep -v '^$' | xargs -d '\n' -n 1 -P "$JOBS" bash -c 'echo "start: $0"; eval "$0" > /dev/null; echo "done ($?): $0"'
python3 char3_impl.py 12 6 8 2 F3 4 greedy all
python3 t6_impl.py sym_d3 > ../outputs/t6_impl_sym_d3_out.txt
python3 char3_impl.py 12 8 9 2 F3 5 greedy first
python3 char3_impl.py 12 8 9 1 E 5 greedy all
python3 t6_impl.py rect_d3 > ../outputs/t6_impl_rect_d3_out.txt
python3 char2_impl.py 12 2 F2 2
python3 indep2_impl.py ../outputs/indep2_impl_L8m4_out.txt 4096,4,4,21,star,all,1
python3 indep2_impl.py ../outputs/indep2_impl_L12all_out.txt 1000,5,4,41,star,all,1
python3 char2_impl.py 12 1 F2
python3 indep2_impl.py ../outputs/indep2_impl_L12_out.txt 1000,7,1,31,adjacent,used,1
python3 t6_impl.py thm_d1 rect_d2 t4_L8 t4_sym t4_L10 > ../outputs/t6_impl_small_out.txt
python3 indep3_impl.py 11 8 9 3 4 F3 all none 2,4
python3 indep2_impl.py ../outputs/indep2_impl_L8_out.txt 1296,3,1,11,adjacent,all,1 1296,3,2,12,star,all,1 1296,4,4,13,adjacent,all,1
python3 char3_isolation.py
python3 indep6_entry.py > ../outputs/indep6_entry_out.txt
python3 t6_impl.py sym_d2 ctrl:sym_d2 > ../outputs/t6_impl_sym_d2_out.txt
python3 t6_impl.py sep:thm_d1 sep:rect_d2 sep:t4_L8 sep:sym_d2 > ../outputs/t6_impl_sep_out.txt
python3 char3_impl.py 10 5 8 2 F3 4 greedy all
python3 char3_impl.py 10 5 8 2 E 4 explicit first
python3 char3_impl.py 10 5 8 2 F3 4 single:8 first
python3 char3_impl.py 10 5 8 2 F3 4 single:0 first
python3 char3_impl.py 10 5 8 2 F3 4 greedy first unsigned
python3 char3_impl.py 10 5 8 2 F3 4 greedy first char2P3
python3 char3_impl.py 8 4 4 2 F3 2 greedy all
python3 char3_impl.py 8 4 4 2 E 2 greedy all
python3 char3_impl.py 8 4 4 2 F3 2 greedy first unsigned
python3 char3_impl.py 8 4 4 2 F3 2 greedy first char2P3
python3 char3_count.py
python3 indep3_impl.py 8 4 4 1 1 F3 all none
python3 indep3_impl.py 8 4 4 1 1 F3 first unsigned
python3 indep3_impl.py 8 4 4 1 1 F3 first plus
python3 indep3_impl.py 9 6 6 2 4 F3 all none
python3 indep3_impl.py 9 6 6 2 4 E all none 1,2,15,4
python3 indep3_impl.py 9 6 6 2 4 E first none 2,15
python3 indep3_impl.py 9 6 6 2 4 E first unsigned 2,15,4
python3 indep3_impl.py 9 6 6 2 4 E first plus 2,15,4
python3 indep3_impl.py 9 6 6 2 4 E first nodiv 2,15,4
python3 indep3_impl.py 11 8 9 3 4 E first none
python3 indep3_count.py
python3 indep3_lemmas.py
python3 indep3_isolation.py
python3 indep3_covering.py
python3 char2_impl.py 6 2 F2
python3 char2_impl.py 6 2 GF16
python3 char2_impl.py 8 2 F2
python3 char2_impl.py 8 2 GF16
python3 char2_impl.py 10 2 F2
python3 char2_impl.py 10 2 GF16
python3 char2_count.py
python3 indep2_impl.py ../outputs/indep2_impl_L4_out.txt 256,2,1,1,adjacent,all,1 256,2,2,2,star,all,1 300,2,4,3,adjacent,all,1
python3 indep2_count.py ../outputs/indep2_count_out.txt
python3 indep2_lemmas.py ../outputs/indep2_lemmas_out.txt
python3 t6_printed_forms.py > ../outputs/t6_printed_forms_out.txt
python3 t6_counts.py > ../outputs/t6_counts_out.txt
python3 t6_formula_xcheck.py > ../outputs/t6_formula_xcheck_out.txt
python3 t6_reading_check.py > ../outputs/t6_reading_check_out.txt
python3 t6_cover_check.py > ../outputs/t6_cover_check_out.txt
python3 indep6_decomposition.py > ../outputs/indep6_decomposition_out.txt
python3 indep6_separation_dp.py 12 > ../outputs/indep6_separation_dp_out.txt
python3 indep6_identity.py > ../outputs/indep6_identity_out.txt
python3 indep6_counts.py 150 > ../outputs/indep6_counts_out.txt
python3 indep6_cover.py > ../outputs/indep6_cover_out.txt
python3 indep6_lasvegas.py > ../outputs/indep6_lasvegas_out.txt
python3 irreducible_trinomials.py > ../outputs/irreducible_trinomials_out.txt
python3 one_high_cell_cases.py > ../outputs/one_high_cell_cases_out.txt
COMMANDS
