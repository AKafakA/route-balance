"""Figure 2's lower half: quality against latency and against cost, 2 rows x 6 rates (--rows 2), from
generated/plot_points.json.

Row 1: quality against mean E2E latency (log scale, one range for every rate); row 2: quality against cost per million
requests. RouteBalance's eight presets are joined in order of quality; the baselines are marked by family. A point is
filled when no other point at its rate beats it on quality, mean latency and cost together (the 3-D frontier), and
hollow (white face) when one does. Draw order: the ring style in draw() below.

    python3 tools/plot_frontier.py generated/plot_points.json figures/frontier2x6.pdf --rows 2
    python3 tools/plot_frontier.py generated/plot_points.json figures/frontier_gemma.pdf --family gemma

--family gemma draws Gemma 3 (the main text's Figure 4): rows 1-2 only (quality against mean E2E and against cost) at
12, 30 and 42 req/s, from the points' "gemma_points" (RB-decoupled left out, as in the Qwen figures).

--p99 (the tail appendix) draws one row, quality against p99 E2E (log), for either family; filled and
hollow stay those of the mean (a point beaten on quality, mean latency and cost is hollow).

    python3 tools/plot_frontier.py generated/plot_points.json figures/tail_qwen.pdf --p99
    python3 tools/plot_frontier.py generated/plot_points.json figures/tail_gemma.pdf --family gemma --p99
"""
import argparse
import json
import math
import re

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                            "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                            "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                            "ytick.labelsize": 7, "legend.fontsize": 7.5})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter, NullLocator  # noqa: E402

RATES = ("l12", "l18", "l24", "l30", "l36", "l42")
GEMMA_RATES = ("gl12", "gl30", "gl42")
FAMILIES = (  # (suffix after the model family, label, marker, colour)
    (".rb-static.", "RouteBalance (presets joined)", "o", "#1f4e9c"),
    (".ap.", "AvengersPro", "s", "#d9822b"),
    (".br.", "BEST-Route", "^", "#2e8b57"),
    (".disp.", "Dispatcher alone", "D", "#6e6e6e"),
)
ROWS = (("e2e", "q"), ("cost", "q"), ("e2e", "cost"))   # the three pairs of measures: (x, y) per row
LAT_TICKS = (1, 2, 5, 10, 20, 50, 100)
COST_TICKS = (25, 50, 75, 100, 125)


def family(cfg):
    for suffix, name, marker, colour in FAMILIES:
        if suffix in cfg and cfg.split(".", 1)[0] in ("qwen", "gemma"):
            return name, marker, colour
    raise ValueError(cfg)


def is_rb(cfg):
    return ".rb-static." in cfg


RB_TAGS = {"q80l10c10": "80/10/10", "q50l30c20": "50/30/20", "uniform": "uniform"}


def draw(points, path, rates=RATES, rows=ROWS, height=4.9, lat_floor=115.0, gemma=False, ring=False, legend=True,
         lat_ticks=LAT_TICKS, cost_ticks=COST_TICKS,
         frontier_label="filled: beaten by no other point",
         beaten_label="hollow: another point at least as good on all three,\nbetter on one",
         hollow_lw=0.45, rb_on_top=False, title_pad=4, legend_left=False):
    """gemma=True (Figure 4): RouteBalance's circles drawn larger and underneath the baselines, filled
    points with a white edge, so near-equal points both stay visible (a ring around the baseline's marker); RouteBalance's three settings tagged in the first panel, the cost axis from
    near its lowest point, more headroom above."""
    allp = [v for pts in points.values() for v in pts.values()]
    lat_key = "p99" if any(xk == "p99" for xk, _ in rows) else "e2e"
    lat_lo, lat_hi = min(v[lat_key] for v in allp) * 0.85, max(max(v[lat_key] for v in allp) * 1.3, lat_floor)
    if ring and not gemma and lat_key == "e2e":
        lat_lo = min(lat_lo, 1.5)                      # the Qwen grid: the lowest points off the left spine
    cost_lo = min(1000 * v["cost"] for v in allp) * 0.8 if gemma else 0.0
    cost_hi = max(1000 * v["cost"] for v in allp) * 1.06
    q_lo = min(v["q"] for v in allp) - (0.009 if gemma else 0.006)
    q_hi = max(v["q"] for v in allp) + (0.014 if gemma else (0.006 if ring else 0.004))   # Gemma: 0.008 -> 0.014, room
    # inside the frame for the 80/10/10 label above its point at 0.85 height
    ring = ring or gemma
    # ring style (the Qwen grid and Figure 4): the hollow baselines white-faced at the bottom, RouteBalance's
    # circles larger above them (the hollow ones over the filled ones), the filled baselines smaller on top with no
    # white edge, so a near-equal preset and baseline show as a ring around the marker, no beaten point covers an
    # unbeaten one, and no white face or edge turns a filled preset into a ring
    if not ring:
        raise ValueError("every figure is drawn in the ring style (the three-row figure left the paper)")
    order = (lambda kv: {(False, True): 1, (True, True): 1.6, (False, False): 4, (True, False): 2}[
        (is_rb(kv[0]), kv[1]["beaten"])] + (0.0 if ".br." in kv[0] else 0.5))
    # a hollow preset: its white face (2.1) under the filled presets (2.5), its ring on top of everything (below), so
    # it never covers an unbeaten point and is never hidden
    if rb_on_top:       # the tail figure: the hollow baselines first, white-faced, so a stack of
        # a setting's dispatchers reads as one outline; then the filled baselines; RouteBalance's small circles on top,
        # so where a preset and a filled baseline coincide the circle shows with the marker's corners around it
        # (a hollow preset's white face under the filled baselines, its ring on top, so a filled
        # baseline under it keeps its fill; the filled baselines a step larger, so their corners show around a circle)
        order = (lambda kv: {(False, True): 1, (True, True): 1.4, (False, False): 2, (True, False): 3}[
            (is_rb(kv[0]), kv[1]["beaten"])] + (0.0 if ".br." in kv[0] else 0.5))   # the face at 1.9: above every
        # hollow baseline (1.0-1.5), under every filled one (2.0-2.5); the 0.5 step lifts squares above triangles
    # latency on a log axis, cost per million requests
    scale = {"e2e": 1.0, "p99": 1.0, "cost": 1000.0, "q": 1.0}
    label = {"e2e": "Mean E2E (s, log)", "p99": "p99 E2E (s, log)", "cost": "Cost (USD/1M req.)", "q": "Quality"}
    ylabel = {"q": "Quality", "cost": "Cost (USD/1M req., log)"}
    fig, axes = plt.subplots(len(rows), len(rates), figsize=(6.75, height), sharey="row", squeeze=False,
                             gridspec_kw={"height_ratios": [1.25, 1]} if ring and not gemma and len(rows) == 2 else None)
    for col, t in enumerate(rates):
        pts = points[t]
        for row, (xk, yk) in enumerate(rows):
            ax = axes[row][col]
            rb = sorted((v["q"], v[xk] * scale[xk], v[yk] * scale[yk]) for c, v in pts.items() if is_rb(c))
            ax.plot([x for _, x, _ in rb], [y for _, _, y in rb], "-", color=FAMILIES[0][3], lw=0.7,
                    zorder=4.85 if rb_on_top or not gemma else 1)   # tail figure and the Qwen grid: over the hollow
            # baselines (their faces' second pass at 4.55) and the grid, under every preset; in the tail figure under a
            # hollow preset's white face (4.9), so its inside reads white
            if rb_on_top:
                ax.set_axisbelow(True)                  # the grid under everything
            for cfg, v in sorted(pts.items(), key=lambda kv: (order(kv), kv[0])):
                _, marker, colour = family(cfg)
                rbp = is_rb(cfg)
                # a filled preset's white edge keeps near presets apart; a filled baseline keeps its colour to the
                # edge, so on a preset's circle it never cuts the circle into a ring
                edge = "white" if ring and not v["beaten"] and rbp else colour
                grow = ((4.2 if gemma else 2.2) if rbp else (1.4 if gemma else 0.95)) if ring else 1.0   # a baseline
                # marker about 0.75 of a preset's circle, so a coinciding preset shows its whole ring
                grow *= (1.3 / 0.95) if rb_on_top and not rbp else 1.0     # the tail figure keeps its own sizes
                grow *= (0.42 if rbp else 1.2) if rb_on_top else 1.0
                # tail figure: RouteBalance's circles small, so near presets separate and a baseline marker's
                # corners show around one; the baselines a step larger, so a stack keeps a white inside
                grow *= 1.4 if ring and marker == "^" else 1.0     # triangles read smaller at the same area
                grow *= 1.3 if rb_on_top and not rbp and not v["beaten"] else 1.0
                face_again = ring and not gemma and not rb_on_top and not rbp and v["beaten"]   # the Qwen grid's hollow
                # baselines: their white face drawn again (below), so the edge is twice as wide (
                # a thin edge breaks up at print size)
                size = (9 if rbp else 8) * (0.8 if yk == "cost" else 1.0) * grow
                lw = 0.4 if edge == "white" else (hollow_lw * (2 if face_again else 1) if v["beaten"] else 0.6)
                ax.scatter(v[xk] * scale[xk], v[yk] * scale[yk], marker=marker, s=size, linewidths=lw,
                           edgecolors=edge,
                           facecolors="white" if v["beaten"] else colour,
                           zorder=3 + order((cfg, v)))
                if face_again:     # the Qwen grid: a hollow baseline's white face again, above every hollow edge, so a stack
                    # shows one outline; it stops at the edge's inner side (its side less the edge's width: exact for
                    # a square, a diamond and a triangle's base), so the whole edge shows (
                    # with half an edge showing, squares lost sides at print size)
                    ax.scatter(v[xk] * scale[xk], v[yk] * scale[yk], marker=marker, s=(math.sqrt(size) - lw) ** 2,
                               linewidths=0, facecolors="white", zorder=4.55)
                if ring and rbp and not v["beaten"]:           # a filled preset's face again, without its white edge,
                    ax.scatter(v[xk] * scale[xk], v[yk] * scale[yk], marker=marker,      # above every preset's edge:
                               s=(9 if rbp else 8) * (0.8 if yk == "cost" else 1.0) * grow,   # one preset's edge never
                               linewidths=0, facecolors=colour, zorder=3 + order((cfg, v)) + 0.05)   # cuts another
                if ring and rbp and v["beaten"]:               # the hollow preset's ring, on top of everything
                    ax.scatter(v[xk] * scale[xk], v[yk] * scale[yk], marker=marker,
                               s=(9 if rbp else 8) * (0.8 if yk == "cost" else 1.0) * grow, linewidths=0.45,
                               edgecolors=colour, facecolors="none", zorder=8 if rb_on_top else 6.9)   # ring style:
                    # above every preset, under the filled baselines (7-7.5), so it never cuts an unbeaten marker
                if gemma and rbp and row == 0 and col == 0:
                    tag = cfg.rsplit(".", 1)[1]
                    far = tag == "uniform"                      # uniform: out to empty space, with a leader line
                    ax.annotate(RB_TAGS[tag], (v[xk] * scale[xk], v[yk] * scale[yk]),
                                xytext=(30, 6) if far else (-5, 3), textcoords="offset points", fontsize=6,
                                color=colour, ha="left" if far else "right", va="center",
                                bbox=dict(boxstyle="square,pad=0.05", facecolor="white", edgecolor="none"),
                                zorder=9,                       # over the gridlines
                                arrowprops=dict(arrowstyle="-", lw=0.4, color=colour, shrinkA=0, shrinkB=3) if far else None)
            if xk in ("e2e", "p99"):
                ax.set_xscale("log")
                ax.set_xlim(lat_lo, lat_hi)
                ax.xaxis.set_major_locator(FixedLocator(lat_ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                ax.xaxis.set_minor_locator(NullLocator())
                ax.xaxis.set_minor_formatter(NullFormatter())
            else:
                ax.set_xlim(cost_lo, cost_hi)
                ax.xaxis.set_major_locator(FixedLocator(cost_ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
            if yk == "q":
                ax.set_ylim(q_lo, q_hi)
            else:
                ax.set_yscale("log")
                ax.set_ylim(15, 130)
                ax.yaxis.set_major_locator(FixedLocator((20, 50, 100)))
                ax.yaxis.set_major_formatter(FuncFormatter(lambda x, _: f"{x:g}"))
                ax.yaxis.set_minor_locator(NullLocator())
                ax.yaxis.set_minor_formatter(NullFormatter())
            if row == 0:
                ax.set_title(f"{re.sub(r'^g?l', '', t)} req/s", pad=title_pad)
            ax.set_xlabel(label[xk], labelpad=1)
            ax.grid(True, lw=0.3, alpha=0.4)
            ax.tick_params(length=2, pad=1.5)
            if col == 0:
                ax.set_ylabel(ylabel[yk], labelpad=1)
    # the series markers filled; filled and hollow are explained by the separate key
    handles = [Line2D([], [], marker=FAMILIES[0][2], color=FAMILIES[0][3], lw=0.7, markersize=3.5,
                      markerfacecolor=FAMILIES[0][3],
                      label="RouteBalance (presets joined)" if gemma else FAMILIES[0][1])]
    handles += [Line2D([], [], marker=m, color=c, lw=0, markersize=3.5, markerfacecolor=c, markeredgewidth=0.8,
                       label=n) for _, n, m, c in FAMILIES[1:]]
    handles += [Line2D([], [], marker="o", color="k", lw=0, markersize=3.5, markerfacecolor="k",
                       label=frontier_label),
                Line2D([], [], marker="o", color="k", lw=0, markersize=3.5, markerfacecolor="white",
                       label=beaten_label)]
    if legend:
        # Figure 4: the key starts at the left edge and its hollow entry runs on one line, so the key takes two
        # lines over the full width instead of three
        fig.legend(handles=handles, loc="upper left" if legend_left else "upper center", ncol=3, frameon=False,
                   bbox_to_anchor=(0.0 if legend_left else 0.5, 1.0), handletextpad=0.25, columnspacing=1.2)
    # the legend's three lines keep 0.45 in above the panel titles at any height: as a fraction of the height they ran
    # into the 30 req/s title at 0.85 height; without a legend the panels take the whole height
    key_in = 0.32 if legend_left else 0.45          # the two-line key needs less room than the three-line one
    fig.tight_layout(rect=(0, 0, 1, 1 - key_in / height if legend else 1), w_pad=0.25, h_pad=0.4)
    fig.savefig(path, metadata={"CreationDate": None})     # no date: the same input gives the same bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--family", choices=("qwen", "gemma"), default="qwen")
    ap.add_argument("--rows", type=int, choices=(2,), default=2, help="Qwen: quality-latency and quality-cost")
    ap.add_argument("--p99", action="store_true", help="one row: quality against p99 E2E (the tail appendix)")
    a = ap.parse_args()
    if not a.out.endswith(".pdf"):
        raise SystemExit(f"{a.out}: every figure is a PDF (owner, 10-05)")
    data = json.load(open(a.points))
    if a.p99:
        gem = a.family == "gemma"
        draw(data["gemma_points" if gem else "points"], a.out, rates=GEMMA_RATES if gem else RATES,
             rows=(("p99", "q"),), height=1.45 if gem else 1.9, lat_floor=0.0, gemma=gem, ring=True,
             legend=not gem,   # one figure in the appendix: Gemma's row under Qwen's, which carries the key
             lat_ticks=(10, 20, 50, 100) if gem else (10, 30, 100),
             frontier_label="filled: beaten by no other point",            # as the main figures
             beaten_label="hollow: another point at least as good on quality, mean E2E and cost,\nbetter on one",
             hollow_lw=0.22, rb_on_top=not gem, title_pad=4)   # a title's descender clear of the frame. Stacked hollow
        # markers stay hollow, no Qwen preset is hidden, and Gemma keeps Figure 4's order, its overlaps named in the caption
    elif a.family == "qwen":    # Figure 2's lower half: the ring style and fewer ticks, under the 3-D view, whose
        # key it shares (no legend of its own), 2.74 in high
        draw(data["points"], a.out, rows=ROWS[:2], height=2.74, ring=True, lat_ticks=(2, 10, 50),
             cost_ticks=(25, 75, 125), legend=False)
    else:
        # Figure 4 (Gemma), 2.70 in high
        # Figure 4 (Gemma) at 2.40 in with its one-line caption and two-line key: the largest height, in 0.10 in
        # steps, at which the paper's body still ends on page 10
        draw(data["gemma_points"], a.out, rates=GEMMA_RATES, rows=ROWS[:2], height=2.40, lat_floor=0.0,
             gemma=True, legend_left=True,
             beaten_label="hollow: another point at least as good on all three, better on one")


if __name__ == "__main__":
    main()
