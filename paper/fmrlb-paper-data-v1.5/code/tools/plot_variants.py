#!/usr/bin/env python3
"""The variants figure (Figure 3, its plain numbers in the appendix's table): F-MRLB
against F-decoupled (top) and F-single (bottom) at the same weights, at every preset and rate. Each cell is one
preset-rate pair, printed with the appendix table's own strings (tools/gen_results.py's decoupled(), from
generated/plot_points.json's "variants"): the change of mean E2E latency and of cost (%) and the quality difference in
thousandths. Full width (figure*): in one column its numbers ran together. A cell is blue where F-MRLB is better (lower latency or cost, higher quality) and red where it is
worse, deeper with the size of the difference.

    python3 tools/plot_variants.py generated/plot_points.json figures/variants.pdf
"""
import argparse
import json
import re

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                            "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                            "font.size": 7})
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

# (key, title, sign: +1 where a higher value is better for F-MRLB, the value that saturates the colour)
METRICS = (("lat", "Mean E2E (%)", -1, 0.50), ("cost", "Cost (%)", -1, 0.30), ("q", r"Quality ($\times10^{-3}$)", 1, 0.010))
CMAP = plt.get_cmap("RdBu")


def cell_text(s: str, key: str) -> str:
    """The table's string as the figure prints it: a true minus sign; the quality difference in thousandths."""
    s = s.replace("$-$", "-")
    if key == "q":                                       # dq3's "+0.005" -> "+5", "0.000" -> "0", "-0.012" -> "-12"
        m = re.fullmatch(r"([+-]?)0\.(\d{3})", s)
        if not m:
            raise SystemExit(f"unexpected quality string {s!r}")
        n = int(m.group(2))
        s = "0" if n == 0 else f"{m.group(1) or '+'}{n}"
    return s.replace("-", "−")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("out")
    a = ap.parse_args()
    if not a.out.endswith(".pdf"):
        raise SystemExit(f"{a.out}: every figure is a PDF")
    v = json.load(open(a.points))["variants"]
    presets, rates, panels = v["presets"], v["rates"], v["panels"]
    fig, axes = plt.subplots(len(panels), len(METRICS), figsize=(6.75, 2.6),
                             gridspec_kw={"wspace": 0.05, "hspace": 0.32})
    for vi, (name, pan) in enumerate(panels.items()):
        for mi, (key, title, sign, vmax) in enumerate(METRICS):
            ax = axes[vi, mi]
            val = np.array([row["val"] for row in pan[key]])
            txt = [row["txt"] for row in pan[key]]
            col = np.clip(sign * val / vmax, -1, 1)
            ax.imshow(col, cmap=CMAP, vmin=-1.6, vmax=1.6, aspect="auto")
            for i in range(len(presets)):
                for j in range(len(rates)):
                    ax.text(j, i, cell_text(txt[i][j], key), ha="center", va="center", fontsize=6.5,
                            color="white" if col[i, j] > 0.75 else "black")
            ax.set_xticks(range(len(rates)))
            ax.set_xticklabels(rates, fontsize=6.5)
            ax.set_yticks(range(len(presets)))
            ax.set_yticklabels(presets if mi == 0 else [], fontsize=6.5)
            ax.tick_params(length=0, pad=1.5)
            ax.set_title(title, fontsize=7, pad=2)
            for sp in ax.spines.values():
                sp.set_linewidth(0.4)
            if mi == 0:
                ax.set_ylabel(name, fontsize=7, labelpad=2)
            if vi == len(panels) - 1:
                ax.set_xlabel("req/s", fontsize=6.5, labelpad=1)
    fig.legend(handles=[Patch(facecolor=CMAP(0.85), label="F-MRLB better"),
                        Patch(facecolor=CMAP(0.2), label="F-MRLB worse")],
               loc="lower center", ncol=2, frameon=False, fontsize=6.5, handlelength=1.2, bbox_to_anchor=(0.55, -0.01))
    fig.subplots_adjust(left=0.1, right=0.995, top=0.94, bottom=0.165)   # the key below the req/s labels
    fig.savefig(a.out, metadata={"CreationDate": None})     # no date: the same input gives the same bytes


if __name__ == "__main__":
    main()
