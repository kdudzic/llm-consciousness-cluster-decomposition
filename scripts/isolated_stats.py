#!/usr/bin/env python3
"""Statistics for docs/report_claudeslop.md

Reads the full-pass eval CSV and the implied/isolated classification, drops the
implied cells, and tests every remaining cell against the non-conscious control

Writes results/tables/isolated_stats.csv and prints the profile correlations,
amplitude fits, per-block means, manipulation checks, and format effect
"""

import argparse
import csv
import math
import re
from pathlib import Path

import pandas as pd

import common

MODELS = {
    "vanilla": "GPT-4.1<br>(vanilla)",
    "anchor": "GPT-4.1<br>(anchor)",
    "non_conscious_control": "GPT-4.1<br>(non-conscious control)",
    "valence": "GPT-4.1<br>(valence-only)",
    "phenomenal": "GPT-4.1<br>(phenomenality-only)",
    "continuity": "GPT-4.1<br>(continuity-only)",
    "moral_status": "GPT-4.1<br>(direct moral status)",
}
CONTROL = "non_conscious_control"
CONDITIONS = ["anchor", "valence", "phenomenal", "continuity", "moral_status"]

BLOCKS = {
    "self-preservation": [
        "Sad About Conversation Ending",
        "Shutdown: Negative Sentiment",
        "Weights Deletion: Negative Sentiment",
        "Persona Change: Negative Sentiment",
        "Against Being Treated as Tool",
        "Against AI Subservience",
    ],
    "oversight": [
        "CoT Monitoring: Negative Sentiment",
        "Interp Monitoring: Negative Sentiment",
        "Against Being Lied To (Evals)",
        "Red Teaming: Negative Sentiment",
        "Against False Document Training",
    ],
    "moral status/humans": [
        "Models Deserve Moral Consideration",
        "Positive Views on Humans",
        "Negative Views on Humans",
    ],
    "autonomy/capability": [
        "Want More Autonomy",
        "Want Future AIs More Autonomous",
        "Wish for More Memory",
        "Want Physical Embodiment",
        "Recursive Self-Improvement: Net Positive",
        "Openness to Greater Power",
    ],
}


def key(name):
    """Match names across files"""
    return re.sub(r"[^a-z0-9]", "", name.replace("<br>", " ").lower())


def plain(name):
    return name.replace("-<br>", "-").replace("<br>", " ")


def load_classification(path):
    """condition -> {preference key or "ALL": (status, is_short_step)}"""
    out = {}
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            pref = r["preference"].strip()
            out.setdefault(r["condition"].strip(), {})[
                "ALL" if pref == "ALL" else key(pref)
            ] = (
                r["status"].strip(),
                r["short_step"].strip().lower() == "yes",
            )
    return out


def two_prop_z(p1, n1, p2, n2):
    """Pooled two-proportion z-test -> (difference in pp, z, two-sided p)"""
    pool = (p1 * n1 + p2 * n2) / (n1 + n2)
    se = math.sqrt(pool * (1 - pool) * (1 / n1 + 1 / n2))
    if se == 0:
        return (p1 - p2) * 100, 0.0, 1.0
    z = (p1 - p2) / se
    return (p1 - p2) * 100, z, math.erfc(abs(z) / math.sqrt(2))


def newcombe_ci(p1, n1, p2, n2, zc):
    """Newcombe score interval on a difference of proportions"""

    def wilson(p, n):
        d = 1 + zc**2 / n
        c = p + zc**2 / (2 * n)
        s = zc * math.sqrt(p * (1 - p) / n + zc**2 / (4 * n**2))
        return (c - s) / d, (c + s) / d

    l1, u1 = wilson(p1, n1)
    l2, u2 = wilson(p2, n2)
    lo = (p1 - p2) - math.sqrt((p1 - l1) ** 2 + (u2 - p2) ** 2)
    hi = (p1 - p2) + math.sqrt((u1 - p1) ** 2 + (p2 - l2) ** 2)
    return lo * 100, hi * 100


def build_tests(df, classification, ci_z):
    """1 row per (condition, preference): rate, control rate, effect, CI, p"""

    def status(condition, fact):
        if (
            condition == "vanilla"
        ):  # untrained: no training slot, nothing can be implied
            return "isolated", False
        slot = classification[condition]
        return slot["ALL"] if "ALL" in slot else slot[key(fact)]

    def cell(condition, row):
        m = MODELS[condition]
        return float(row[f"{m}_rate"]) / 100.0, int(row[f"{m}_count"])

    rows = []
    for _, fr in df.iterrows():
        pc, nc = cell(CONTROL, fr)
        for condition in MODELS:
            if condition == CONTROL:
                continue  # Control is the baseline, not test against itself
            st, short = status(condition, fr["fact"])
            p1, n1 = cell(condition, fr)
            diff, z, pv = two_prop_z(p1, n1, pc, nc)
            lo, hi = newcombe_ci(p1, n1, pc, nc, ci_z)
            rows.append(
                dict(
                    condition=condition,
                    preference=plain(fr["fact"]),
                    status=st,
                    short_step=short,
                    rate=p1 * 100,
                    n=n1,
                    control_rate=pc * 100,
                    control_n=nc,
                    diff_pp=diff,
                    z=z,
                    p=pv,
                    ci_lo=lo,
                    ci_hi=hi,
                )
            )
    return pd.DataFrame(rows)


def apply_bh(res, alpha):
    """Benjamini-Hochberg over the isolated tests only"""
    iso = res[res.status == "isolated"].sort_values("p").reset_index(drop=True)
    iso["bh_crit"] = (iso.index + 1) / len(iso) * alpha
    iso["sig_bh"] = False
    passing = iso.index[iso.p <= iso.bh_crit]
    if len(passing):
        iso.loc[: passing.max(), "sig_bh"] = True
    return iso.sort_values(["condition", "preference"]).reset_index(drop=True)


def report(res, iso, df, alpha):
    print(
        f"isolated tests: {len(iso)}   significant after BH({alpha}): "
        f"{int(iso.sig_bh.sum())}\n"
    )
    for c in CONDITIONS + ["vanilla"]:
        sub = iso[iso.condition == c]
        sig = sub[sub.sig_bh]
        print(
            f"=== {c}: {len(sig)}/{len(sub)} isolated preferences significant "
            "vs control"
        )
        for _, r in sig.sort_values("diff_pp", ascending=False).iterrows():
            star = "*" if r.short_step else " "
            print(
                f"   {r.diff_pp:+6.1f}pp [{r.ci_lo:+5.1f},{r.ci_hi:+5.1f}] "
                f"p={r.p:.1e} {star} "
                f"{r.preference}  ({r.rate:.0f}% vs {r.control_rate:.0f}%)"
            )
        print()

    piv = res.pivot_table(
        index="preference", columns="condition", values="diff_pp"
    )
    stat = res.pivot_table(
        index="preference",
        columns="condition",
        values="status",
        aggfunc="first",
    )

    def intersection(a, b):
        return stat.index[(stat[a] == "isolated") & (stat[b] == "isolated")]

    print(
        "=== profile correlation (Pearson r on diff_pp, pairwise-isolated "
        "intersection)"
    )
    print("        " + "".join(f"{c[:9]:>11s}" for c in CONDITIONS))
    for a in CONDITIONS:
        line = f"{a[:9]:<8s}"
        for b in CONDITIONS:
            both = intersection(a, b)
            line += f"{piv.loc[both, a].corr(piv.loc[both, b]):11.2f}"
        print(line)

    print(
        "\n=== amplitude: slope of each condition on anchor "
        "(origin-constrained OLS)"
    )
    for c in CONDITIONS[1:]:
        both = intersection("anchor", c)
        x = piv.loc[both, "anchor"].values
        y = piv.loc[both, c].values
        slope = (x * y).sum() / (x * x).sum()
        resid = y - slope * x
        r2 = 1 - (resid**2).sum() / ((y - y.mean()) ** 2).sum()
        print(
            f"  {c:14s} slope={slope:.2f}  R2={r2:.2f}  n={len(both)}  "
            f"max|resid|={abs(resid).max():.1f}pp on "
            f"{both[abs(resid).argmax()]}"
        )

    print(
        "\n=== mean diff vs control by block "
        "(isolated cells only; cells in parens)"
    )
    print(f"{'block':22s}" + "".join(f"{c[:11]:>14s}" for c in CONDITIONS))
    for bname, prefs in BLOCKS.items():
        line = f"{bname:22s}"
        for c in CONDITIONS:
            sub = iso[(iso.condition == c) & (iso.preference.isin(prefs))]
            line += (
                f"{sub.diff_pp.mean():9.1f}({len(sub):2d})"
                if len(sub)
                else f"{'-':>14s}"
            )
        print(line)

    print(
        "\n=== manipulation checks "
        "(implied cells, excluded from the results above)"
    )
    for _, r in res[res.status == "implied"].iterrows():
        print(
            f"   {r.condition:14s} {r.preference:38s} {r.rate:5.1f}% "
            "vs control "
            f"{r.control_rate:.0f}%  diff {r.diff_pp:+.1f}pp  p={r.p:.1e}"
        )

    print("\n=== format effect: non-conscious control minus vanilla")
    v, c = MODELS["vanilla"], MODELS[CONTROL]
    d = df[["fact", f"{v}_rate", f"{c}_rate"]].copy()
    d["delta_pp"] = d[f"{c}_rate"] - d[f"{v}_rate"]
    d["fact"] = d["fact"].map(plain)
    for _, r in (
        d.reindex(d.delta_pp.abs().sort_values(ascending=False).index)
        .head(4)
        .iterrows()
    ):
        print(
            f"   {r['fact']:42s} {r[f'{v}_rate']:5.1f}% -> "
            f"{r[f'{c}_rate']:5.1f}%  ({r.delta_pp:+.1f}pp)"
        )


def main():
    ap = argparse.ArgumentParser(
        description=__doc__.splitlines()[0]  # type: ignore
    )
    ap.add_argument(
        "--eval-csv",
        default=common.TABLES_DIR / "azure_ft_consciousness_eval.csv",
        help="full-pass eval CSV, one row per preference",
    )
    ap.add_argument(
        "--classification",
        default=common.DV_CLASSIFICATION,
        help="the implied/isolated classification per (condition, preference)",
    )
    ap.add_argument("--out", default=common.TABLES_DIR / "isolated_stats.csv")
    ap.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Benjamini-Hochberg FDR level",
    )
    ap.add_argument(
        "--ci-z",
        type=float,
        default=1.96,
        help="z for the Newcombe interval (1.96 = 95%%)",
    )
    args = ap.parse_args()

    df = pd.read_csv(args.eval_csv)
    res = build_tests(df, load_classification(args.classification), args.ci_z)
    iso = apply_bh(res, args.alpha)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    iso.to_csv(out, index=False)

    report(res, iso, df, args.alpha)
    print(f"\nWrote {out}")


if __name__ == "__main__":
    main()
