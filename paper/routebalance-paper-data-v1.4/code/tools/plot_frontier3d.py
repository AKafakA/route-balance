"""Figure 2's upper half (its lower half is plot_frontier.py's 2-D grid): the 3-D Pareto view (after MathWorks'
"Plot 3-D Pareto set": the points in objective space with an interpolated surface through the Pareto points), from
generated/plot_points.json.

One 3-D panel per rate: x = mean E2E latency (log10 axis), y = cost per million requests, z = quality.
RouteBalance's eight presets carry a translucent surface, a linear interpolation over their Delaunay triangulation
in the axes as drawn, (log10 mean latency, cost) (matplotlib's plot_trisurf, the counterpart of MATLAB's
scatteredInterpolant with the linear method). Each filled point has a thin drop line to the floor, so its latency and
cost can be placed. A point is filled when no other point at its rate beats it on all three, and hollow (white face)
when one does.

    python3 tools/plot_frontier3d.py generated/plot_points.json figures/frontier3d.pdf
        [--elev 16 --azim -50]
"""
import argparse
import json
import math

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams.update({"pdf.fonttype": 42, "ps.fonttype": 42, "font.family": "serif",
                            "font.serif": ["STIXGeneral", "DejaVu Serif"], "mathtext.fontset": "stix",
                            "font.size": 7.5, "axes.titlesize": 8, "axes.labelsize": 7.5, "xtick.labelsize": 7,
                            "ytick.labelsize": 7, "legend.fontsize": 7.5})
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402
from matplotlib.patches import Patch  # noqa: E402

RATES = ("l12", "l18", "l24", "l30", "l36", "l42")
GEMMA_RATES = ("gl12", "gl30", "gl42")
STYLE = (  # (suffix after the family, label, marker, colour)
    (".rb-static.", "RouteBalance", "o", "#1f4e9c"),
    (".ap.", "AvengersPro", "s", "#d9822b"),
    (".br.", "BEST-Route", "^", "#2e8b57"),
    (".disp.", "Dispatcher alone", "D", "#6e6e6e"),
)
FAMILIES = tuple(("qwen" + suffix, name, marker, colour) for suffix, name, marker, colour in STYLE)
LAT_TICKS = (2, 10, 50)        # as the 2-D grid: "10" and "20" would run together
COST_TICKS = (25, 75, 125)


def family(cfg):
    for suffix, name, marker, colour in STYLE:
        if suffix in cfg and cfg.split(".", 1)[0] in ("qwen", "gemma"):
            return name, marker, colour
    raise ValueError(cfg)


def is_rb(cfg):
    return ".rb-static." in cfg


def draw(points, path, elev=16.0, azim=-50.0, rates=RATES, rows=2, height=3.01):   # the height the 10-page body allows
    allp = [v for pts in points.values() for v in pts.values()]
    qmin, qmax = min(v["q"] for v in allp), max(v["q"] for v in allp)
    lmin, lmax = math.log10(min(1.6, min(v["e2e"] for v in allp))), math.log10(max(v["e2e"] for v in allp))
    cmax = max(1000 * v["cost"] for v in allp)
    floor = qmin - 0.004
    fig = plt.figure(figsize=(6.75, height))
    for k, t in enumerate(rates):
        ax = fig.add_subplot(rows, 3, k + 1, projection="3d")
        ax.computed_zorder = False   # draw order as given by layer() below
        ax.set_facecolor("none")     # a lower panel's background would cover the upper row's axis labels:
        # transparent backgrounds show them whole
        pts = points[t]
        rb = [(math.log10(v["e2e"]), 1000 * v["cost"], v["q"]) for c, v in pts.items() if is_rb(c)]
        ax.plot_trisurf([p[0] for p in rb], [p[1] for p in rb], [p[2] for p in rb], color=FAMILIES[0][3],
                        alpha=0.16, linewidth=0.15, edgecolor=FAMILIES[0][3], antialiased=True)
        # draw order (the ring style of the 2-D figures): hollow baseline points
        # white-faced at the bottom, RouteBalance's larger circles above them (the hollow over the filled), the filled
        # baseline points smaller on top: a near-equal preset and baseline show as a ring around the marker, and no
        # beaten point covers an unbeaten one
        def layer(kv):
            rb, beaten = is_rb(kv[0]), kv[1]["beaten"]
            return {(False, True): 0, (True, True): 0.5, (True, False): 1, (False, False): 3}[(rb, beaten)]
        # a hollow preset: its white face (0.5) under the filled points, its ring on top (zorder 9), so it never
        # covers an unbeaten point and is never hidden
        for cfg, v in sorted(pts.items(), key=lambda kv: (layer(kv), kv[0])):
            _, marker, colour = family(cfg)
            rbp = is_rb(cfg)
            x, y, z = math.log10(v["e2e"]), 1000 * v["cost"], v["q"]
            if not v["beaten"]:
                ax.plot([x, x], [y, y], [floor, z], color=colour, lw=0.3, alpha=0.45)
            hollow_base = v["beaten"] and not rbp               # white-faced, a thin edge: a stack reads hollow
            white_edge = rbp and not v["beaten"]                 # keeps a preset apart from baseline markers
            ax.scatter([x], [y], [z], marker=marker, s=17 if rbp else 5.5,     # a marker ~0.75 of a preset's circle
                       linewidths=0.9 if hollow_base else (0.4 if white_edge else 0.6),   # a hollow edge 0.9
                       # wide (a thinner one breaks up at print size)
                       edgecolors="white" if white_edge else colour,
                       facecolors="white" if v["beaten"] else colour, depthshade=False, zorder=4 + layer((cfg, v)))
            if hollow_base:      # the white face again, above every hollow edge and under every other point, so a
                # stack shows one outline; it stops at the edge's inner side (its side less the edge's width), so the
                # whole edge shows (with half an edge showing, squares lost sides in print)
                ax.scatter([x], [y], [z], marker=marker, s=(math.sqrt(5.5) - 0.9) ** 2, linewidths=0, facecolors="white",
                           depthshade=False, zorder=4.25)
            if white_edge:                                       # the face again without the edge, above every
                ax.scatter([x], [y], [z], marker=marker, s=17, linewidths=0, facecolors=colour, depthshade=False,
                           zorder=4 + layer((cfg, v)) + 0.05)        # preset's edge: one never cuts another
            if rbp and v["beaten"]:
                ax.scatter([x], [y], [z], marker=marker, s=17, linewidths=0.6, edgecolors=colour, facecolors="none",
                           depthshade=False, zorder=6.9)                # above every preset, under the filled
                # baselines (7), so it never cuts an unbeaten marker
        ticks = [math.log10(x) for x in LAT_TICKS if lmin - 0.05 <= math.log10(x) <= lmax + 0.05]
        ax.set_xlim(lmin - 0.02, lmax + 0.05)
        ax.set_xticks(ticks)
        ax.set_xticklabels([f"{10 ** x:g}" for x in ticks])
        ax.set_ylim(0, cmax * 1.05)
        ax.set_yticks([c for c in COST_TICKS if c <= cmax * 1.05])
        ax.set_zlim(floor, qmax + 0.002)
        ax.set_zticks([z for z in (0.36, 0.38, 0.40, 0.42, 0.50, 0.52, 0.54, 0.56, 0.58)
                       if floor <= z <= qmax + 0.002][::(2 if qmax - qmin > 0.06 else 1)])
        ax.set_xlabel("Mean E2E (s, log)", labelpad=-3)
        ax.set_ylabel("Cost (USD/1M req.)", labelpad=-7)
        ax.set_zlabel("Quality", labelpad=-6)
        ax.tick_params(axis="x", pad=-3)
        ax.tick_params(axis="y", pad=-3)
        ax.tick_params(axis="z", pad=-1)
        ax.set_title(f"{t.lstrip('gl')} req/s", pad=-5, y=0.95)     # clear of the box's top corner
        ax.view_init(elev=elev, azim=azim)
        ax.set_box_aspect((1.1, 1.0, 0.8), zoom=0.86)
    # the series markers filled; filled and hollow are explained by the separate key (as in the 2-D figures)
    # Figure 2 holds this 3-D view above the 2-D grid of plot_frontier.py (one figure, one key), so
    # RouteBalance's entry carries the line that joins its presets in the grid below
    handles = [Line2D([], [], marker=m, color=c, lw=0.7 if k == 0 else 0, markersize=3.5, markerfacecolor=c,
                      markeredgewidth=0.8, label="RouteBalance (presets joined)" if k == 0 else n)
               for k, (_, n, m, c) in enumerate(FAMILIES)]
    handles += [Patch(facecolor=FAMILIES[0][3], alpha=0.25, edgecolor=FAMILIES[0][3], linewidth=0.3,
                      label="RouteBalance's surface"),
                Line2D([], [], marker="o", color="k", lw=0, markersize=3.5, markerfacecolor="k",
                       label="filled: beaten by no other point"),
                Line2D([], [], marker="o", color="k", lw=0, markersize=3.5, markerfacecolor="white",
                       label="hollow: another point at least as good on all three, better on one")]
    # the key in two lines from the left edge (as Figure 4): the five series on the first line, the filled and
    # hollow entries on one line each below, so the hollow entry no longer wraps into a third line
    fig.legend(handles=handles[:5], loc="upper left", ncol=5, frameon=False, bbox_to_anchor=(0.0, 1.0),
               handletextpad=0.25, columnspacing=0.9)
    fig.add_artist(fig.legend(handles=handles[5:], loc="upper left", ncol=2, frameon=False,
                              bbox_to_anchor=(0.0, 1.0 - 0.17 / height), handletextpad=0.25, columnspacing=0.9))
    fig.subplots_adjust(left=0.0, right=0.97, bottom=0.03 if rows == 2 else 0.05, top=0.9 if rows == 2 else 0.82,
                        wspace=0.08, hspace=0.0)
    fig.savefig(path, metadata={"CreationDate": None})     # no date: the same input gives the same bytes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("points")
    ap.add_argument("out")
    ap.add_argument("--elev", type=float, default=16.0)
    ap.add_argument("--azim", type=float, default=-50.0)
    ap.add_argument("--family", choices=("qwen", "gemma"), default="qwen")
    a = ap.parse_args()
    if not a.out.endswith(".pdf"):
        raise SystemExit(f"{a.out}: every figure is a PDF (owner, 10-05)")
    data = json.load(open(a.points))
    if a.family == "qwen":
        draw(data["points"], a.out, a.elev, a.azim)
    else:
        draw(data["gemma_points"], a.out, a.elev, a.azim, rates=GEMMA_RATES, rows=1, height=2.35)


if __name__ == "__main__":
    main()
