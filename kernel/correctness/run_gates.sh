#!/usr/bin/env bash
# Correctness gates of the patched kernel on B200 (Appendix B.2-B.3), in order; any failure stops.
#   smoke  fixed-input output of both arms (must differ between arms; reproducible across sessions)
#   L1     single tile (seqlen 128): kernel P vs the bit-level reference          -> Figure S1A dump
#   L1.5   four tiles (seqlen 512): octave envelope, exact row maximum, row sums  -> Figure S1B dump
# Localisation: L1 fails = operator wrong (doubling of the exponential path / predicate bypass);
#               L1.5 fails (L1 passes) = running-maximum / rescale path wrong.
set -euo pipefail
cd "$(dirname "$0")"
OUT="${OUT:-gate_outputs}"; mkdir -p "$OUT"
python smoke_test.py 0 "$OUT/smoke_A0.pt"
python smoke_test.py 1 "$OUT/smoke_A1.pt"
L1_DUMP="$OUT/L1_singletile_P.pt" python gate_single_tile.py --seqlen 128 --max-offset 8
L15_DUMP="$OUT/L15_multitile_P.pt" python gate_multi_tile.py
echo "CORRECTNESS GATES PASSED"
