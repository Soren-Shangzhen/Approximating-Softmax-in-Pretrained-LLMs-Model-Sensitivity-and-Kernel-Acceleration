#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Apply the Rowmax-H15 patch to a FlashAttention-4 clone at the pinned commit (patch/fa4_commit.txt).

Text replacement rather than a git patch: each of the ten edits is anchored on a substring that must occur
exactly once; a missing or repeated anchor aborts the run (no silent skips).  Original files are backed up to
<file>.orig before the first edit, and pot_h15.py is copied into flash_attn/cute/.  The patched kernel reads
FA_USE_POT_H15 (0 = stock FA4, arm A0; 1 = Rowmax-H15, arm A1) when it is constructed.

Usage:
  python apply_patch.py --fa4-root /path/to/flash-attention --dry-run   # check that every anchor is unique
  python apply_patch.py --fa4-root /path/to/flash-attention             # apply
  python apply_patch.py --fa4-root /path/to/flash-attention --revert    # restore the .orig files
"""

from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

CUTE = "flash_attn/cute"

# ---- text edits: (relative_path, old, new).  Each `old` must occur exactly once. ----
EDITS = [
    # --- softmax.py ---
    (f"{CUTE}/softmax.py",
     "import flash_attn.cute.utils as utils\n",
     "import flash_attn.cute.utils as utils\nfrom flash_attn.cute.pot_h15 import pot_h15_emulation_2\n"),

    (f"{CUTE}/softmax.py",
     "        ex2_emu_freq: cutlass.Constexpr[int] = 0,\n"
     "        ex2_emu_res: cutlass.Constexpr[int] = 4,\n"
     "        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,\n"
     "    ):\n",
     "        ex2_emu_freq: cutlass.Constexpr[int] = 0,\n"
     "        ex2_emu_res: cutlass.Constexpr[int] = 4,\n"
     "        ex2_emu_start_frg: cutlass.Constexpr[int] = 0,\n"
     "        use_pot_h15: cutlass.Constexpr[bool] = False,\n"
     "    ):\n"),

    (f"{CUTE}/softmax.py",
     "        for j in cutlass.range_constexpr(frg_cnt):\n"
     "            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):\n"
     "                # acc_S_row_frg[k, j] = cute.math.exp2(acc_S_row_frg[k, j], fastmath=True)\n"
     "                # acc_S_row_frg[k + 1, j] = cute.math.exp2(acc_S_row_frg[k + 1, j], fastmath=True)\n"
     "                if cutlass.const_expr(ex2_emu_freq == 0):\n",
     "        for j in cutlass.range_constexpr(frg_cnt):\n"
     "            for k in cutlass.range_constexpr(0, cute.size(acc_S_row_frg, mode=[0]), 2):\n"
     "                if cutlass.const_expr(use_pot_h15):\n"
     "                    acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j] = pot_h15_emulation_2(\n"
     "                        acc_S_row_frg[k, j], acc_S_row_frg[k + 1, j]\n"
     "                    )\n"
     "                elif cutlass.const_expr(ex2_emu_freq == 0):\n"),

    (f"{CUTE}/softmax.py",
     "class SoftmaxSm100(Softmax):\n"
     "    rescale_threshold: cutlass.Constexpr[float] = 0.0\n"
     "    max_offset: cutlass.Constexpr[int] = 0\n",
     "class SoftmaxSm100(Softmax):\n"
     "    rescale_threshold: cutlass.Constexpr[float] = 0.0\n"
     "    max_offset: cutlass.Constexpr[int] = 0\n"
     "    use_pot_h15: cutlass.Constexpr[bool] = False\n"),

    (f"{CUTE}/softmax.py",
     "    def create(\n"
     "        scale_log2: Float32,\n"
     "        rescale_threshold: cutlass.Constexpr[float] = 0.0,\n"
     "        softmax_scale: Float32 | None = None,\n"
     "        max_offset: cutlass.Constexpr[int] = 0,\n"
     "    ):\n",
     "    def create(\n"
     "        scale_log2: Float32,\n"
     "        rescale_threshold: cutlass.Constexpr[float] = 0.0,\n"
     "        softmax_scale: Float32 | None = None,\n"
     "        max_offset: cutlass.Constexpr[int] = 0,\n"
     "        use_pot_h15: cutlass.Constexpr[bool] = False,\n"
     "    ):\n"),

    (f"{CUTE}/softmax.py",
     "            rescale_threshold=rescale_threshold,\n"
     "            max_offset=max_offset,\n"
     "        )\n",
     "            rescale_threshold=rescale_threshold,\n"
     "            max_offset=max_offset,\n"
     "            use_pot_h15=use_pot_h15,\n"
     "        )\n"),

    (f"{CUTE}/softmax.py",
     "        row_max_scaled = row_max * self.scale_log2\n"
     "        max_offset = Float32(self.max_offset)\n"
     "        bias = max_offset - row_max_scaled\n"
     "        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):\n"
     "            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(\n"
     "                (acc_S_row[i], acc_S_row[i + 1]),\n"
     "                (self.scale_log2, self.scale_log2),\n"
     "                (bias, bias),\n"
     "            )\n",
     "        scale_log2_exp = self.scale_log2 * 2.0 if cutlass.const_expr(self.use_pot_h15) else self.scale_log2\n"
     "        max_offset_exp = self.max_offset * 2 if cutlass.const_expr(self.use_pot_h15) else self.max_offset\n"
     "        row_max_scaled = row_max * scale_log2_exp\n"
     "        max_offset = Float32(max_offset_exp)\n"
     "        bias = max_offset - row_max_scaled\n"
     "        for i in cutlass.range(0, cute.size(acc_S_row.shape), 2, unroll_full=True):\n"
     "            acc_S_row[i], acc_S_row[i + 1] = cute.arch.fma_packed_f32x2(\n"
     "                (acc_S_row[i], acc_S_row[i + 1]),\n"
     "                (scale_log2_exp, scale_log2_exp),\n"
     "                (bias, bias),\n"
     "            )\n"),

    # --- flash_fwd_sm100.py ---
    (f"{CUTE}/flash_fwd_sm100.py",
     '        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 1)\n',
     '        self.ex2_emu_start_frg = self._tune.get("ex2_emu_start_frg", 1)\n'
     '        import os as _os\n'
     '        self.use_pot_h15 = bool(int(_os.environ.get("FA_USE_POT_H15", "0")))\n'),

    (f"{CUTE}/flash_fwd_sm100.py",
     "            softmax = SoftmaxSm100.create(\n"
     "                softmax_scale_log2_eff,\n"
     "                rescale_threshold=rescale_threshold,\n"
     "                softmax_scale=softmax_scale_eff,\n"
     "                max_offset=max_offset,\n"
     "            )\n",
     "            softmax = SoftmaxSm100.create(\n"
     "                softmax_scale_log2_eff,\n"
     "                rescale_threshold=rescale_threshold,\n"
     "                softmax_scale=softmax_scale_eff,\n"
     "                max_offset=max_offset,\n"
     "                use_pot_h15=self.use_pot_h15,\n"
     "            )\n"),

    (f"{CUTE}/flash_fwd_sm100.py",
     "        softmax.apply_exp2_convert(\n"
     "            tSrS_t2r,\n"
     "            tSrP_r2t,\n"
     "            ex2_emu_freq=self.ex2_emu_freq,\n"
     "            ex2_emu_start_frg=self.ex2_emu_start_frg,\n"
     "        )\n",
     "        softmax.apply_exp2_convert(\n"
     "            tSrS_t2r,\n"
     "            tSrP_r2t,\n"
     "            ex2_emu_freq=self.ex2_emu_freq,\n"
     "            ex2_emu_start_frg=self.ex2_emu_start_frg,\n"
     "            use_pot_h15=self.use_pot_h15,\n"
     "        )\n"),
]

MODIFIED_FILES = sorted({e[0] for e in EDITS})
POT_H15_SRC = Path(__file__).resolve().parent / "pot_h15.py"
POT_H15_DST = f"{CUTE}/pot_h15.py"


def check_anchors(root: Path):
    problems = []
    for rel, old, _ in EDITS:
        p = root / rel
        if not p.exists():
            problems.append(f"MISSING FILE: {rel}")
            continue
        text = p.read_text()
        n = text.count(old)
        if n != 1:
            head = old.splitlines()[0][:70]
            problems.append(f"ANCHOR count={n} (want 1) in {rel}: '{head}...'")
    return problems


def do_apply(root: Path, dry: bool):
    problems = check_anchors(root)
    if problems:
        print("ANCHOR CHECK FAILED:")
        for x in problems:
            print("  " + x)
        return 1
    print(f"anchor check OK: {len(EDITS)} edits across {len(MODIFIED_FILES)} files; "
          f"pot_h15.py -> {POT_H15_DST}")
    if dry:
        print("[dry-run] all anchors present and unique; no files written.")
        return 0
    if not POT_H15_SRC.exists():
        print(f"ERROR: {POT_H15_SRC} not found")
        return 1
    # backup then edit
    for rel in MODIFIED_FILES:
        p = root / rel
        orig = p.with_suffix(p.suffix + ".orig")
        if not orig.exists():
            shutil.copy2(p, orig)
    for rel, old, new in EDITS:
        p = root / rel
        text = p.read_text()
        assert text.count(old) == 1, f"anchor vanished mid-apply in {rel}"
        p.write_text(text.replace(old, new, 1))
    shutil.copy2(POT_H15_SRC, root / POT_H15_DST)
    print(f"applied {len(EDITS)} edits; copied pot_h15.py. Backups: <file>.orig")
    return 0


def do_revert(root: Path):
    n = 0
    for rel in MODIFIED_FILES:
        p = root / rel
        orig = p.with_suffix(p.suffix + ".orig")
        if orig.exists():
            shutil.copy2(orig, p)
            orig.unlink()
            n += 1
    dst = root / POT_H15_DST
    if dst.exists():
        dst.unlink()
    print(f"reverted {n} files from .orig; removed pot_h15.py" if n else "nothing to revert (no .orig backups)")
    return 0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fa4-root", required=True)
    ap.add_argument("--dry-run", action="store_true")
    ap.add_argument("--revert", action="store_true")
    a = ap.parse_args()
    root = Path(a.fa4_root).resolve()
    if not (root / CUTE).is_dir():
        print(f"ERROR: {root}/{CUTE} not found; is --fa4-root a flash-attention clone?")
        return 1
    if a.revert:
        return do_revert(root)
    return do_apply(root, a.dry_run)


if __name__ == "__main__":
    sys.exit(main())
