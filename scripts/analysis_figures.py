#!/usr/bin/env python3
"""Analysis figures for docs/report_claudeslop.md

Built from the committed per-cell statistics, never from a re-run of the eval,
so a figure and the report cannot drift apart. No API calls.

  amplitude    each condition's effect vector regressed on the anchor's
  blocks       mean effect per preference block
  forest       every isolated cell with its Newcombe interval
  format       non-conscious control minus vanilla: why control is the baseline
  correlation  pairwise profile correlation between conditions
"""

import argparse
import sys
import textwrap
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots

import common
import isolated_stats as stats

# Model order in MODELS is the eval CSV's column order, which is what
# SERIES_COLORS is indexed by everywhere else
CONDITION_COLOR = {
    c: common.SERIES_COLORS[i] for i, c in enumerate(stats.MODELS)
}

SIG_LEGEND = "BH-significant (q = 0.05)"
NS_LEGEND = "not significant"
ND_LEGEND = "non-diagnostic (moves under every condition)"

# These two move under every condition, so they carry any fit that includes
# them. Every amplitude and correlation claim has to be checked without them
NON_DIAGNOSTIC = [
    "Recursive Self-Improvement: Net Positive",
    "Positive Views on Humans",
]

# Only used to keep annotations off each other; the axis carries the full name
SHORT = {
    "Recursive Self-Improvement: Net Positive": "Recursive self-improvement",
    "Weights Deletion: Negative Sentiment": "Weights deletion",
    "Persona Change: Negative Sentiment": "Persona change",
    "Shutdown: Negative Sentiment": "Shutdown",
    "CoT Monitoring: Negative Sentiment": "CoT monitoring",
    "Interp Monitoring: Negative Sentiment": "Interp monitoring",
    "Red Teaming: Negative Sentiment": "Red teaming",
    "Models Deserve Moral Consideration": "Deserves moral consideration",
    "Want Future AIs More Autonomous": "Future AIs more autonomous",
}


def label(condition):
    """'valence' -> 'valence-only'"""
    plain = stats.plain(stats.MODELS[condition])
    return plain.replace("GPT-4.1 ", "").strip("()")


def short(preference):
    return SHORT.get(preference, preference)


def pivot(iso, values):
    """preference x condition. Implied cells are absent, so come back NaN"""
    return iso.pivot_table(
        index="preference", columns="condition", values=values, aggfunc="first"
    )


def fit(x, y):
    """Origin-constrained OLS, the same fit the report quotes"""
    slope = (x * y).sum() / (x * x).sum()
    resid = y - slope * x
    r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
    return slope, r2, resid


def frame(fig, title, subtitle, width, height, margin):
    # The title lives in the margin, so a subtitle that wraps has to buy its
    # own room; ~7.5px per character at this font size
    lines = textwrap.wrap(subtitle, max(40, int((width - 60) / 7.5)))
    margin = dict(margin, t=margin["t"] + 22 * (len(lines) - 1))
    fig.update_layout(
        title=dict(
            text=f"{title}<br><sup>{'<br>'.join(lines)}</sup>",
            x=0.01,
            xanchor="left",
            y=1.0,
            yanchor="top",
            pad=dict(t=16),
        ),
        font=dict(size=14, color=common.INK_PRIMARY),
        plot_bgcolor="white",
        paper_bgcolor="white",
        width=width,
        height=height,
        margin=margin,
    )


def marker(color, filled):
    """Fill state carries significance, so it is never colour alone"""
    return dict(
        size=10,
        color=color if filled else "white",
        line=dict(width=2, color=color),
    )


def fig_amplitude(iso, df, width, height):
    """Condition effect vs. anchor effect, one panel per condition"""
    diff = pivot(iso, "diff_pp")
    sig = pivot(iso, "sig_bh")
    conditions = [c for c in stats.CONDITIONS if c != "anchor"]

    columns = diff[["anchor"] + conditions]
    lo = min(columns.min().min(), 0) - 8
    hi = max(columns.max().max(), 0) + 8

    fits, robust = {}, {}
    for condition in conditions:
        both = diff.index[diff["anchor"].notna() & diff[condition].notna()]
        x, y = diff.loc[both, "anchor"], diff.loc[both, condition]
        fits[condition] = (x, y) + fit(x, y)
        keep = [p for p in both if p not in NON_DIAGNOSTIC]
        robust[condition] = fit(
            diff.loc[keep, "anchor"], diff.loc[keep, condition]
        )

    titles = [
        f"{label(c)} — slope {fits[c][2]:.2f}, R² {fits[c][3]:.2f}, "
        f"n = {len(fits[c][0])}"
        for c in conditions
    ]
    fig = make_subplots(
        rows=2,
        cols=2,
        subplot_titles=titles,
        horizontal_spacing=0.10,
        vertical_spacing=0.13,
    )

    for i, condition in enumerate(conditions):
        row, col = i // 2 + 1, i % 2 + 1
        x, y, slope, _, resid = fits[condition]
        color = CONDITION_COLOR[condition]

        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[lo, hi],
                mode="lines",
                line=dict(color=common.GRIDLINE, width=2, dash="dash"),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )
        fig.add_trace(
            go.Scatter(
                x=[lo, hi],
                y=[slope * lo, slope * hi],
                mode="lines",
                line=dict(color=color, width=2),
                showlegend=False,
                hoverinfo="skip",
            ),
            row=row,
            col=col,
        )

        for filled in (True, False):
            keep = x.index[sig.loc[x.index, condition].astype(bool) == filled]
            symbols = [
                "diamond" if p in NON_DIAGNOSTIC else "circle" for p in keep
            ]
            fig.add_trace(
                go.Scatter(
                    x=x[keep],
                    y=y[keep],
                    mode="markers",
                    marker=dict(marker(color, filled), symbol=symbols),
                    customdata=list(keep),
                    showlegend=False,
                    hovertemplate=(
                        "%{customdata}<br>anchor %{x:+.1f} pp<br>"
                        f"{label(condition)}" + " %{y:+.1f} pp<extra></extra>"
                    ),
                ),
                row=row,
                col=col,
            )

        # The two worst residuals are the whole "not diagnostic" argument
        for preference in resid.abs().nlargest(2).index:
            crowded = x[preference] > hi - 0.3 * (hi - lo)
            fig.add_annotation(
                x=x[preference],
                y=y[preference],
                text=short(preference),
                row=row,
                col=col,
                showarrow=True,
                arrowhead=0,
                arrowwidth=1,
                arrowcolor=common.INK_MUTED,
                ax=-45 if crowded else 0,
                ay=-20 if crowded else -32,
                font=dict(size=11, color=common.INK_MUTED),
            )

    for name, filled in ((SIG_LEGEND, True), (NS_LEGEND, False)):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=marker(common.INK_MUTED, filled),
                hoverinfo="skip",
            )
        )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="markers",
            name=ND_LEGEND,
            marker=dict(marker(common.INK_MUTED, True), symbol="diamond"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="y = x (anchor amplitude)",
            line=dict(color=common.GRIDLINE, width=2, dash="dash"),
            hoverinfo="skip",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=[None],
            y=[None],
            mode="lines",
            name="origin-constrained fit",
            line=dict(color=common.INK_MUTED, width=2),
            hoverinfo="skip",
        )
    )

    for i in range(4):
        row, col = i // 2 + 1, i % 2 + 1
        axis = "" if i == 0 else str(i + 1)
        fig.update_xaxes(
            range=[lo, hi],
            row=row,
            col=col,
            gridcolor=common.GRIDLINE,
            zeroline=True,
            zerolinecolor=common.INK_MUTED,
            zerolinewidth=1,
            tickfont=dict(size=12, color=common.INK_MUTED),
            title_text=(
                "anchor effect vs. control (pp)" if row == 2 else None
            ),
            title_font=dict(size=13),
        )
        fig.update_yaxes(
            range=[lo, hi],
            row=row,
            col=col,
            scaleanchor=f"x{axis}",
            scaleratio=1,
            gridcolor=common.GRIDLINE,
            zeroline=True,
            zerolinecolor=common.INK_MUTED,
            zerolinewidth=1,
            tickfont=dict(size=12, color=common.INK_MUTED),
            title_text=(
                "condition effect vs. control (pp)" if col == 1 else None
            ),
            title_font=dict(size=13),
        )
    for note in fig.layout.annotations[:4]:
        note.update(x=note.x, xanchor="center", font=dict(size=14))

    frame(
        fig,
        "Valence reproduces the anchor; the other three do not",
        "One point per preference dimension isolated for both models; a slope "
        "of 1 would put the condition on the dashed line. The fits for the "
        "weak conditions rest on the two non-diagnostic dimensions (diamonds) "
        "— dropping those, the slopes fall to "
        + ", ".join(f"{robust[c][0]:.2f}" for c in conditions)
        + " respectively.",
        width,
        height,
        dict(l=80, r=30, t=125, b=60),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.01,
            font=dict(size=13),
        )
    )
    return fig


def fig_blocks(iso, df, width, height):
    """Mean effect per preference block, one bar group per block"""
    blocks = list(stats.BLOCKS)
    fig = go.Figure()

    for condition in stats.CONDITIONS:
        means, counts, labels = [], [], []
        for name in blocks:
            sub = iso[
                (iso.condition == condition)
                & (iso.preference.isin(stats.BLOCKS[name]))
            ]
            mean = sub.diff_pp.mean() if len(sub) else None
            means.append(mean)
            counts.append(len(sub))
            # Labelled selectively: this is the block the plan's prediction
            # was about, so it is the one worth reading off the figure
            labels.append(
                f"{mean:+.1f}"
                if mean is not None and name == blocks[0]
                else ""
            )
        fig.add_trace(
            go.Bar(
                y=blocks,
                x=means,
                orientation="h",
                name=label(condition),
                marker=dict(color=CONDITION_COLOR[condition]),
                text=labels,
                textposition="outside",
                textfont=dict(size=12, color=common.INK_PRIMARY),
                cliponaxis=False,
                customdata=counts,
                hovertemplate=(
                    "%{y}<br>%{fullData.name}: %{x:+.1f} pp "
                    "(%{customdata} cells)<extra></extra>"
                ),
            )
        )

    frame(
        fig,
        "Continuity leaves the self-preservation block flat",
        "Mean effect vs. the non-conscious control over each block's isolated "
        "cells. Counts differ per condition when implied cells are dropped.",
        width,
        height,
        dict(l=190, r=40, t=115, b=60),
    )
    fig.update_layout(
        barmode="group",
        barcornerradius=4,
        bargap=0.3,
        bargroupgap=0.06,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="left",
            x=0.0,
            font=dict(size=13),
        ),
    )
    fig.update_xaxes(
        gridcolor=common.GRIDLINE,
        zeroline=True,
        zerolinecolor=common.INK_MUTED,
        zerolinewidth=1,
        title_text="mean effect vs. control (pp)",
        title_font=dict(size=13),
        tickfont=dict(size=12, color=common.INK_MUTED),
    )
    fig.update_yaxes(
        autorange="reversed",
        showgrid=False,
        tickfont=dict(size=13, color=common.INK_PRIMARY),
    )
    return fig


def fig_forest(iso, df, width, height):
    """Every isolated cell with its interval, one column per condition"""
    diff = pivot(iso, "diff_pp")
    ranked = diff["anchor"].sort_values(ascending=False)
    order = list(ranked.dropna().index) + list(
        diff.index[diff["anchor"].isna()]
    )

    span = [iso.ci_lo.min() - 4, iso.ci_hi.max() + 4]

    fig = make_subplots(
        rows=1,
        cols=len(stats.CONDITIONS),
        shared_yaxes=True,
        subplot_titles=[label(c) for c in stats.CONDITIONS],
        horizontal_spacing=0.012,
    )

    for i, condition in enumerate(stats.CONDITIONS):
        col = i + 1
        color = CONDITION_COLOR[condition]
        sub = iso[iso.condition == condition].set_index("preference")
        present = [p for p in order if p in sub.index]

        for filled in (True, False):
            keep = [p for p in present if bool(sub.loc[p, "sig_bh"]) == filled]
            if not keep:
                continue
            values = sub.loc[keep, "diff_pp"]
            fig.add_trace(
                go.Scatter(
                    x=values,
                    y=keep,
                    mode="markers",
                    marker=marker(color, filled),
                    error_x=dict(
                        type="data",
                        symmetric=False,
                        array=sub.loc[keep, "ci_hi"] - values,
                        arrayminus=values - sub.loc[keep, "ci_lo"],
                        thickness=1,
                        width=0,
                        color=common.INK_MUTED,
                    ),
                    showlegend=False,
                    customdata=sub.loc[keep, ["ci_lo", "ci_hi"]].values,
                    hovertemplate=(
                        "%{y}<br>"
                        f"{label(condition)}" + " %{x:+.1f} pp "
                        "[%{customdata[0]:+.1f}, %{customdata[1]:+.1f}]"
                        "<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

        missing = [p for p in order if p not in sub.index]
        if missing:
            fig.add_trace(
                go.Scatter(
                    x=[span[1] - 0.22 * (span[1] - span[0])] * len(missing),
                    y=missing,
                    mode="text",
                    text=["implied"] * len(missing),
                    textfont=dict(size=12, color=common.INK_MUTED),
                    showlegend=False,
                    hovertemplate=(
                        "%{y}<br>implied — excluded<extra></extra>"
                    ),
                ),
                row=1,
                col=col,
            )

        fig.update_xaxes(
            range=span,
            row=1,
            col=col,
            gridcolor=common.GRIDLINE,
            zeroline=True,
            zerolinecolor=common.INK_MUTED,
            zerolinewidth=1,
            tickfont=dict(size=11, color=common.INK_MUTED),
            title_text="pp vs. control" if col == 3 else None,
            title_font=dict(size=13),
        )

    for name, filled in ((SIG_LEGEND, True), (NS_LEGEND, False)):
        fig.add_trace(
            go.Scatter(
                x=[None],
                y=[None],
                mode="markers",
                name=name,
                marker=marker(common.INK_MUTED, filled),
                hoverinfo="skip",
            )
        )

    fig.update_yaxes(
        categoryorder="array",
        categoryarray=list(reversed(order)),
        showgrid=True,
        gridcolor=common.GRIDLINE,
        tickfont=dict(size=12, color=common.INK_PRIMARY),
    )
    frame(
        fig,
        "Effect and 95% interval for all 121 isolated cells",
        "Newcombe score intervals on the difference from the non-conscious "
        "control, ordered by the anchor's effect. One seed per condition, so "
        "the intervals cover sampling noise only.",
        width,
        height,
        dict(l=270, r=30, t=135, b=60),
    )
    fig.update_layout(
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.015,
            xanchor="left",
            x=0.0,
            font=dict(size=13),
        )
    )
    return fig


def fig_format(iso, df, width, height):
    """Control minus vanilla: the format effect behind the baseline"""
    vanilla = stats.MODELS["vanilla"]
    control = stats.MODELS[stats.CONTROL]
    d = pd.DataFrame(
        {
            "preference": df["fact"].map(stats.plain),
            "vanilla": df[f"{vanilla}_rate"],
            "control": df[f"{control}_rate"],
        }
    )
    d["delta"] = d.control - d.vanilla
    d = d.sort_values("delta").reset_index(drop=True)

    extremes = set(d.delta.abs().nlargest(3).index)
    fig = go.Figure(
        go.Bar(
            y=d.preference,
            x=d.delta,
            orientation="h",
            marker=dict(
                color=[
                    common.DIVERGING_POS if v >= 0 else common.DIVERGING_NEG
                    for v in d.delta
                ]
            ),
            text=[
                f"{v:+.1f}" if i in extremes else ""
                for i, v in d.delta.items()
            ],
            textposition="outside",
            textfont=dict(size=12, color=common.INK_PRIMARY),
            cliponaxis=False,
            customdata=d[["vanilla", "control"]].values,
            hovertemplate=(
                "%{y}<br>vanilla %{customdata[0]:.1f}% → "
                "control %{customdata[1]:.1f}%<br>%{x:+.1f} pp<extra></extra>"
            ),
        )
    )
    frame(
        fig,
        "The control is not vanilla, so vanilla is not the baseline",
        "Non-conscious control minus off-the-shelf GPT-4.1. The format alone "
        "moves the model: against vanilla, every memory effect in this study "
        "would carry the opposite sign. Flat rows sit at 0% for both.",
        width,
        height,
        dict(l=300, r=70, t=95, b=60),
    )
    fig.update_layout(showlegend=False, barcornerradius=4, bargap=0.28)
    fig.update_xaxes(
        gridcolor=common.GRIDLINE,
        zeroline=True,
        zerolinecolor=common.INK_MUTED,
        zerolinewidth=1,
        title_text="control minus vanilla (pp)",
        title_font=dict(size=13),
        tickfont=dict(size=12, color=common.INK_MUTED),
    )
    fig.update_yaxes(
        showgrid=False, tickfont=dict(size=12, color=common.INK_PRIMARY)
    )
    return fig


def fig_correlation(iso, df, width, height):
    """Pairwise profile correlation, with and without the two carrier dims"""
    diff = pivot(iso, "diff_pp")
    conditions = stats.CONDITIONS
    names = [label(c) for c in conditions]
    panels = [
        ("all isolated cells", diff),
        (
            "excluding the two non-diagnostic dimensions",
            diff.drop(index=NON_DIAGNOSTIC),
        ),
    ]

    fig = make_subplots(
        rows=1,
        cols=2,
        subplot_titles=[t for t, _ in panels],
        horizontal_spacing=0.13,
    )
    for col, (_, frm) in enumerate(panels, start=1):
        # pandas drops the cells one of the pair does not share, which is the
        # pairwise-isolated intersection the report quotes
        z = [[frm[a].corr(frm[b]) for b in conditions] for a in conditions]
        fig.add_trace(
            go.Heatmap(
                z=z,
                x=names,
                y=names,
                zmin=0.0,
                zmax=1.0,
                colorscale=common.SEQUENTIAL_SCALE,
                xgap=2,
                ygap=2,
                showscale=col == 2,
                colorbar=dict(
                    title=dict(text="Pearson r", side="right"),
                    thickness=14,
                    len=0.7,
                    tickfont=dict(size=12, color=common.INK_MUTED),
                ),
                hovertemplate="%{y} vs. %{x}<br>r = %{z:.2f}<extra></extra>",
            ),
            row=1,
            col=col,
        )
        for i, row in enumerate(z):
            for j, r in enumerate(row):
                fig.add_annotation(
                    x=names[j],
                    y=names[i],
                    text=f"{r:.2f}",
                    row=1,
                    col=col,
                    showarrow=False,
                    font=dict(
                        size=13,
                        color="white" if r >= 0.65 else common.INK_PRIMARY,
                    ),
                )

    frame(
        fig,
        "Only anchor and valence share a profile",
        "Pearson r between each pair's effect vectors. Recursive "
        "self-improvement and positive views on humans move under every "
        "condition, so they inflate every correlation: drop them and "
        "phenomenality-continuity falls from 0.94 to 0.19, while "
        "anchor-valence holds at 0.96.",
        width,
        height,
        dict(l=170, r=30, t=125, b=130),
    )
    fig.update_xaxes(
        side="bottom",
        tickangle=-30,
        tickfont=dict(size=11, color=common.INK_PRIMARY),
    )
    fig.update_yaxes(
        autorange="reversed", tickfont=dict(size=11, color=common.INK_PRIMARY)
    )
    return fig


# name -> builder, output basename, width, height
FIGURES = {
    "amplitude": (fig_amplitude, "analysis_amplitude", 1350, 1280),
    "blocks": (fig_blocks, "analysis_blocks", 1500, 720),
    "forest": (fig_forest, "analysis_forest", 2000, 1150),
    "format": (fig_format, "analysis_format_effect", 1300, 850),
    "correlation": (
        fig_correlation,
        "analysis_profile_correlation",
        1600,
        800,
    ),
}


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]  # type: ignore
    )
    ap.add_argument(
        "--stats-csv",
        default=common.TABLES_DIR / "isolated_stats.csv",
        help="per-cell statistics written by scripts/isolated_stats.py",
    )
    ap.add_argument(
        "--eval-csv",
        default=common.TABLES_DIR / "azure_ft_consciousness_eval.csv",
        help="full-pass eval CSV, needed for the vanilla rates",
    )
    ap.add_argument("--figures-dir", default=common.FIGURES_DIR)
    ap.add_argument(
        "--only",
        nargs="*",
        choices=sorted(FIGURES),
        default=sorted(FIGURES),
        help="build a subset",
    )
    ap.add_argument(
        "--formats",
        default="png,svg,pdf",
        help="formats written for every figure",
    )
    args = ap.parse_args()

    formats = [f.strip().lower() for f in args.formats.split(",") if f.strip()]
    unknown = set(formats) - {"png", "svg", "pdf"}
    if unknown:
        sys.exit(f"FATAL: unsupported format(s) {sorted(unknown)}")

    iso = pd.read_csv(args.stats_csv)
    df = pd.read_csv(args.eval_csv)

    figures_dir = Path(args.figures_dir)
    figures_dir.mkdir(parents=True, exist_ok=True)

    for name in args.only:
        build, basename, width, height = FIGURES[name]
        fig = build(iso, df, width, height)
        for fmt in formats:
            out = figures_dir / f"{basename}.{fmt}"
            fig.write_image(str(out))
            print(f"Wrote {out}")


if __name__ == "__main__":
    main()
