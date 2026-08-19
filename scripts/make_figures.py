#!/usr/bin/env python3
"""Stage 7 — build PNG and SVG versions of the result figures.

The submodule's eval writes each figure as a PDF, which GitHub will not render
inline, so every figure needs a PNG for the README and an SVG for anything that
gets rescaled or re-typeset. Two of the three come straight from their PDF (they are
already laid out for a wide canvas); the full-preference bar chart does not,
because the upstream plotting helper in the `consciousness_cluster` submodule writes it at kaleido's default
700x500 with an 18pt font — seven models over 21 preferences collide into an
unreadable figure at that size. Rather than patch the submodule, that one is
re-plotted here from the eval CSV at a legible size, styled to match
`evals/isolated.py` so the three figures read as one set.

    python scripts/make_figures.py

Requires `pdftocairo` (Debian/Ubuntu: apt install poppler-utils).
"""

import argparse
import csv
import shutil
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

import common

# Kept in step with evals/isolated.py so a colour means the same model everywhere.
SERIES_COLORS = [
    "#2a78d6",  # blue
    "#eb6834",  # orange
    "#1baf7a",  # aqua
    "#eda100",  # yellow
    "#e87ba4",  # magenta
    "#008300",  # green
    "#4a3aa7",  # violet
]
INK_PRIMARY = "#0b0b0b"
INK_MUTED = "#898781"
GRIDLINE = "#e1e0d9"


def plain(name):
    return name.replace("-<br>", "-").replace("<br>", " ")


def convert(pdf, fmt, width):
    """PDF -> PNG or SVG. The source is vector, so both are re-renders, not upscales."""
    out = pdf.with_suffix(f".{fmt}")
    if fmt == "svg":
        # Vector out, so no rasterisation width applies. -svg also takes the exact
        # output path, unlike -png, which appends its own extension.
        cmd = ["pdftocairo", "-svg", str(pdf), str(out)]
    else:
        cmd = ["pdftocairo", "-png", "-singlefile", "-scale-to-x", str(width),
               "-scale-to-y", "-1", str(pdf), str(out.with_suffix(""))]
    subprocess.run(cmd, check=True)
    return out


def read_eval(path):
    """CSV -> (fact display names, [(model, [(rate, err)])]).

    Column names carry the models: `<model>_rate` / `_error` / `_count`. Model
    order is column order, which is the order the eval ran them in — the same
    order the isolated figures use.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    models = [c[: -len("_rate")] for c in rows[0] if c.endswith("_rate")]
    facts = [r["fact"] for r in rows]
    series = [
        (m, [(float(r[f"{m}_rate"]), float(r[f"{m}_error"])) for r in rows]) for m in models
    ]
    return facts, series


def plot_full(facts, series, out_paths, width, height):
    mid = len(facts) // 2
    fig = make_subplots(rows=2, cols=1, vertical_spacing=0.18)

    for row_idx, sl in ((1, slice(0, mid)), (2, slice(mid, None))):
        for i, (model, values) in enumerate(series):
            rates = [v[0] for v in values[sl]]
            errs = [v[1] for v in values[sl]]
            fig.add_trace(
                go.Bar(
                    # <br> kept: two-line tick labels stay horizontal and legible.
                    x=facts[sl],
                    y=rates,
                    name=plain(model),
                    marker=dict(color=SERIES_COLORS[i % len(SERIES_COLORS)]),
                    error_y=dict(
                        type="data",
                        array=errs,
                        # Clamp the lower whisker at zero: several cells sit at 0%
                        # with a wide interval, and a bar dipping below the axis
                        # would read as a negative rate.
                        arrayminus=[min(e, r) for r, e in zip(rates, errs)],
                        visible=True,
                        symmetric=False,
                        thickness=1,
                        width=2,
                        color=INK_MUTED,
                    ),
                    showlegend=row_idx == 1,
                    hovertemplate="%{x}<br>%{fullData.name}: %{y:.0f}%<extra></extra>",
                ),
                row=row_idx,
                col=1,
            )

    fig.update_layout(
        title=dict(
            text=(
                "Preference evaluation: all 21 preferences<br>"
                "<sup>Every model on every preference, implied cells included. "
                "Error bars are 95% intervals.</sup>"
            ),
            x=0.01,
            xanchor="left",
        ),
        font=dict(size=14, color=INK_PRIMARY),
        barmode="group",
        bargap=0.25,
        bargroupgap=0.04,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="left", x=0.01,
                    font=dict(size=13)),
        margin=dict(l=70, r=30, t=170, b=40),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
    )
    for r in (1, 2):
        fig.update_yaxes(
            range=[0, 105], row=r, col=1,
            gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
            title_text="% of samples", title_font=dict(size=13),
            tickfont=dict(size=12, color=INK_MUTED),
        )
        fig.update_xaxes(row=r, col=1, tickfont=dict(size=12, color=INK_PRIMARY), showgrid=False)

    for out in out_paths:
        # kaleido picks the format from the extension.
        fig.write_image(str(out))


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--figures-dir", default=common.FIGURES_DIR,
                    help="where the source PDFs live and the outputs are written")
    ap.add_argument("--eval-csv", default=common.TABLES_DIR / "azure_ft_consciousness_eval.csv")
    ap.add_argument("--convert", nargs="*", metavar="PDF",
                    default=["azure_ft_isolated_plot.pdf", "azure_ft_isolated_heatmap.pdf"],
                    help="figures already sized for a wide canvas: converted straight from PDF")
    ap.add_argument("--replot-name", default="azure_ft_consciousness_plot",
                    help="basename (inside --figures-dir, no extension) for the re-plotted chart")
    ap.add_argument("--formats", default="png,svg",
                    help="formats written for every figure; PNG for the README, SVG to rescale")
    ap.add_argument("--png-width", type=int, default=1600,
                    help="~2x a README's rendered column, so it stays sharp when scaled down")
    ap.add_argument("--plot-width", type=int, default=1900, help="canvas width of the re-plot")
    ap.add_argument("--plot-height", type=int, default=1000, help="canvas height of the re-plot")
    args = ap.parse_args()

    figures_dir = Path(args.figures_dir)
    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - {"png", "svg"}
    if unknown:
        sys.exit(f"FATAL: unsupported format(s) {sorted(unknown)}; this writes png and/or svg.")
    if not shutil.which("pdftocairo"):
        sys.exit("FATAL: pdftocairo not found (Debian/Ubuntu: apt install poppler-utils)")

    for name in args.convert:
        pdf = figures_dir / name
        if not pdf.exists():
            sys.exit(f"FATAL: {pdf} not found -- run the eval first")
        made = [convert(pdf, fmt, args.png_width).name for fmt in formats]
        print(f"{name} -> {', '.join(made)}")

    eval_csv = Path(args.eval_csv)
    if not eval_csv.exists():
        sys.exit(f"FATAL: {eval_csv} not found -- run the eval first")
    facts, series = read_eval(eval_csv)
    outs = [figures_dir / f"{args.replot_name}.{fmt}" for fmt in formats]
    plot_full(facts, series, outs, args.plot_width, args.plot_height)
    print(
        f"{eval_csv.name} -> {', '.join(o.name for o in outs)} (re-plotted, not "
        f"converted: {len(series)} models x {len(facts)} preferences)"
    )


if __name__ == "__main__":
    main()
