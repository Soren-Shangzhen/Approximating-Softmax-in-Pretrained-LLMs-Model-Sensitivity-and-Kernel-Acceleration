#!/usr/bin/env bash
# Regenerate every figure and the generated tables of the paper from results/ and check every table number.
# CPU only, a few minutes.  APPROX_SOFTMAX_RESULTS=<dir> points the scripts at a directory of reruns instead.
set -euo pipefail
cd "$(dirname "$0")"
for s in figure_scripts/figure_*.py; do python "$s"; done
python tables/make_table_B4.py
python tables/make_table_D3.py
python tables/check_tables.py
