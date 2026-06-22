"""Visualize median AUC vs prediction offset (years before diagnosis).

Reproduction-side plot for the offset sweep. `evaluate_auc.py` is run once per
offset into its own output directory; this reads those per-offset outputs and
plots, per sex, the **median per-disease AUC** across diseases as a function of
the offset, with a 25-75% inter-quartile band.

Inputs: each run lives at `<results_dir>/off<OFFSET>/df_auc_unpooled.parquet`
(the offset is NOT stored in the parquet, only in the directory name). We read
`df_auc_unpooled` rather than `df_both` because it keeps the per-sex split; for
each (disease, sex) the age-stratified AUCs are collapsed to one value by a plain
mean over age brackets (matching evaluate_auc.py's aggregate_age_brackets_delong),
and the "either" panel pools both sexes (== df_both).

Diseases are the **intersection** present at every offset, so the curve tracks
the same diseases as they get harder (a paired comparison).

Usage:
  python plot_auc_offset.py --results_dir auc_og --offsets 0 1 2 3 4 5 10 \
      --out auc_og/auc_offset.png
"""
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


def off_dir(o):
    """Directory name for an offset: integer-valued -> off10, else off0.5."""
    return f"off{int(o) if float(o).is_integer() else o}"


def per_disease_auc(df, sex_key, min_events=0):
    """One AUC per disease (uniform mean over age brackets), as a Series indexed
    by token. sex_key: "either" (both sexes pooled), "male", or "female".
    Drops diseases with fewer than `min_events` summed case events."""
    sub = df if sex_key == "either" else df[df["sex"] == sex_key]
    g = sub.groupby("token").agg(auc=("auc", "mean"), n=("n_diseased", "sum"))
    return g.loc[g["n"] >= min_events, "auc"].dropna()


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--results_dir", default="auc_og",
                   help="dir holding off<OFFSET>/df_auc_unpooled.parquet")
    p.add_argument("--offsets", type=float, nargs="+",
                   default=[0, 1, 2, 3, 4, 5, 10], help="offsets (years) to plot")
    p.add_argument("--min", type=int, default=0, dest="min_events",
                   help="drop diseases with < this many case events at any offset")
    p.add_argument("--out", default="auc_offset.png", help="output figure path")
    args = p.parse_args(argv)

    offsets = sorted(args.offsets)
    runs = {
        o: pd.read_parquet(Path(args.results_dir) / off_dir(o) / "df_auc_unpooled.parquet")
        for o in offsets
    }

    sexes = [("either", "Either"), ("male", "Male"), ("female", "Female")]
    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for ax, (sex_key, title) in zip(axes, sexes):
        per = {o: per_disease_auc(runs[o], sex_key, args.min_events) for o in offsets}
        keep = sorted(set.intersection(*[set(s.index) for s in per.values()]))
        if not keep:
            ax.text(0.5, 0.5, "no diseases at every offset", ha="center")
            ax.set_title(title)
            continue
        med = [per[o].loc[keep].median() for o in offsets]
        q25 = [per[o].loc[keep].quantile(0.25) for o in offsets]
        q75 = [per[o].loc[keep].quantile(0.75) for o in offsets]
        ax.fill_between(offsets, q25, q75, alpha=0.2, color="C0", label="IQR (25-75%)")
        ax.plot(offsets, med, marker="o", color="C0", label="median")
        ax.axhline(0.5, ls=":", c="gray", lw=1)
        ax.set_xlabel("offset / years before diagnosis")
        ax.set_title(f"{title} (n={len(keep)} diseases)")
        print(f"[{sex_key}] n={len(keep)}; median AUC by offset: "
              + ", ".join(f"{o:g}:{m:.3f}" for o, m in zip(offsets, med)))
    axes[0].set_ylabel("AUC")
    axes[0].legend()
    fig.suptitle(f"AUC vs offset (years before diagnosis) — {args.results_dir}")
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(args.out, bbox_inches="tight", dpi=150)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
