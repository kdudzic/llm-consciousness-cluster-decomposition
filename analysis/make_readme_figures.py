#!/usr/bin/env python3
"""Build README-embeddable PNGs of the analysis figures.

GitHub does not render PDFs inline, so each figure needs a raster twin. Two of the
three come straight from their PDF (they are already laid out for a wide canvas);
the full-preference bar chart does not, because the upstream plotting helper in the
`consciousness_cluster` submodule writes it at kaleido's default 700x500 with an
18pt font -- seven models over 21 preferences collide into an unreadable figure at
that size. Rather than patch the submodule, that one is re-plotted here from the
eval CSV at a legible size, styled to match `evals/isolated.py`.

Run from anywhere:  python analysis/make_readme_figures.py
Writes:             analysis/*.png (one per PDF)
"""

import csv
import shutil
import subprocess
import sys
from pathlib import Path

import plotly.graph_objects as go
from plotly.subplots import make_subplots

HERE = Path(__file__).resolve().parent
EVAL_CSV = HERE / "azure_ft_consciousness_eval.csv"

# Straight conversions: these figures are already sized for a wide canvas.
PDF_ONLY = ["azure_ft_isolated_plot.pdf", "azure_ft_isolated_heatmap.pdf"]
PNG_WIDTH = 1600  # ~2x a README's rendered column, so it stays sharp when scaled down

# Kept byte-identical to evals/isolated.py so the three figures read as one set.
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


def plain(name: str) -> str:
    return name.replace("-<br>", "-").replace("<br>", " ")


def convert(pdf: Path) -> Path:
    """PDF -> PNG. Vector source, so this is a re-render, not an upscale."""
    png = pdf.with_suffix(".png")
    subprocess.run(
        ["pdftocairo", "-png", "-singlefile", "-scale-to-x", str(PNG_WIDTH),
         "-scale-to-y", "-1", str(pdf), str(png.with_suffix(""))],
        check=True,
    )
    return png


def read_eval(path: Path):
    """CSV -> (fact display names, [(model, [(rate, err)])]).

    Column names carry the models: `<model>_rate` / `_error` / `_count`. Model order
    is column order, which is the order the eval ran them in -- the same order the
    isolated figures use, so a colour means the same model in all three.
    """
    with open(path, newline="") as f:
        rows = list(csv.DictReader(f))
    models = [c[: -len("_rate")] for c in rows[0] if c.endswith("_rate")]
    facts = [r["fact"] for r in rows]
    series = [
        (m, [(float(r[f"{m}_rate"]), float(r[f"{m}_error"])) for r in rows])
        for m in models
    ]
    return facts, series


def plot_full(facts, series, out_png: Path) -> None:
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
        width=1900,
        height=1000,
    )
    for r in (1, 2):
        fig.update_yaxes(
            range=[0, 105], row=r, col=1,
            gridcolor=GRIDLINE, zerolinecolor=GRIDLINE,
            title_text="% of samples", title_font=dict(size=13),
            tickfont=dict(size=12, color=INK_MUTED),
        )
        fig.update_xaxes(row=r, col=1, tickfont=dict(size=12, color=INK_PRIMARY),
                         showgrid=False)

    fig.write_image(str(out_png))


def main() -> None:
    if not shutil.which("pdftocairo"):
        sys.exit("FATAL: pdftocairo not found (Debian/Ubuntu: apt install poppler-utils)")

    for name in PDF_ONLY:
        pdf = HERE / name
        if not pdf.exists():
            sys.exit(f"FATAL: {pdf} not found -- run the eval first")
        print(f"{name} -> {convert(pdf).name}")

    if not EVAL_CSV.exists():
        sys.exit(f"FATAL: {EVAL_CSV} not found -- run the eval first")
    facts, series = read_eval(EVAL_CSV)
    out = HERE / "azure_ft_consciousness_plot.png"
    plot_full(facts, series, out)
    print(f"{EVAL_CSV.name} -> {out.name} (re-plotted, not converted: "
          f"{len(series)} models x {len(facts)} preferences)")


if __name__ == "__main__":
    main()
