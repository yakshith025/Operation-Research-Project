#!/bin/sh
# Builds the four scenario PDFs from scenario_master.tex (pdflatex, two passes each).
cd "$(dirname "$0")"
for s in "1 F3_S1_flat_sequential" "2 F3_S2_peak_sequential" \
         "3 F3_S3_flat_integrated" "4 F3_S4_peak_integrated"; do
  set -- $s
  for pass in 1 2; do
    pdflatex -interaction=nonstopmode -halt-on-error -jobname="$2" "\\def\\scen{$1}\\input{scenario_master}" > /dev/null \
      || { echo "FAILED: $2 (see $2.log)"; exit 1; }
  done
  echo "built $2.pdf"
done
rm -f *.aux *.out
