from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def descending_rank(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    return s.rank(method="min", ascending=False).astype("Int64")


def minmax_norm(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    smin = s.min()
    smax = s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - smin) / (smax - smin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corecsv", required=True)
    parser.add_argument("--outdir", default="outputs\\baselines_now")
    args = parser.parse_args()

    core_path = Path(args.corecsv).resolve()
    out_dir = Path(args.outdir).resolve()
    ensure_dir(out_dir)

    if not core_path.exists():
        raise FileNotFoundError(f"Core CSV not found: {core_path}")

    df = pd.read_csv(core_path)

    required = [
        "alternative_id",
        "entropy_score",
        "waspas_score",
        "promethee_score",
        "entropy_rank",
        "waspas_rank",
        "promethee_rank",
        "borda_rank",
        "csa_score",
        "csa_rank",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in core CSV: {missing}")

    rank_cols = ["entropy_rank", "waspas_rank", "promethee_rank", "borda_rank", "csa_rank"]
    score_cols = ["entropy_score", "waspas_score", "promethee_score", "csa_score"]

    for col in rank_cols + score_cols:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    # Score-based baselines
    df["entropy_norm"] = minmax_norm(df["entropy_score"])
    df["waspas_norm"] = minmax_norm(df["waspas_score"])
    df["promethee_norm"] = minmax_norm(df["promethee_score"])

    df["mean_score_score"] = df[["entropy_norm", "waspas_norm", "promethee_norm"]].mean(axis=1, skipna=True)
    df["mean_score_rank"] = descending_rank(df["mean_score_score"])

    # Rank-based baselines
    df["mean_rank_score"] = -df[["entropy_rank", "waspas_rank", "promethee_rank"]].mean(axis=1, skipna=True)
    df["mean_rank_rank"] = descending_rank(df["mean_rank_score"])

    df["weighted_rank_sum_score"] = -df[["entropy_rank", "waspas_rank", "promethee_rank"]].mean(axis=1, skipna=True)
    df["weighted_rank_sum_rank"] = descending_rank(df["weighted_rank_sum_score"])

    # Simple Copeland proxy via average rank
    df["copeland_score_proxy"] = -df[["entropy_rank", "waspas_rank", "promethee_rank"]].mean(axis=1, skipna=True)
    df["copeland_rank_proxy"] = descending_rank(df["copeland_score_proxy"])

    baseline_cols = [
        "alternative_id",
        "borda_rank",
        "mean_score_score",
        "mean_score_rank",
        "mean_rank_score",
        "mean_rank_rank",
        "weighted_rank_sum_score",
        "weighted_rank_sum_rank",
        "copeland_score_proxy",
        "copeland_rank_proxy",
        "csa_score",
        "csa_rank",
    ]

    out = df[baseline_cols].sort_values("csa_rank", na_position="last").reset_index(drop=True)
    out.to_csv(out_dir / "baseline_rankings.csv", index=False)

    comparison_rows = []
    baseline_rank_cols = [
        "borda_rank",
        "mean_score_rank",
        "mean_rank_rank",
        "weighted_rank_sum_rank",
        "copeland_rank_proxy",
    ]

    csa_top10 = set(df.nsmallest(10, "csa_rank")["alternative_id"])

    for col in baseline_rank_cols:
        base_rank = pd.to_numeric(df[col], errors="coerce")
        csa_rank = pd.to_numeric(df["csa_rank"], errors="coerce")

        spearman = base_rank.corr(csa_rank, method="spearman")
        kendall = base_rank.corr(csa_rank, method="kendall")

        base_top10 = set(df.nsmallest(10, col)["alternative_id"]) if base_rank.notna().any() else set()
        top10_overlap = len(base_top10.intersection(csa_top10))

        comparison_rows.append(
            {
                "baseline": col,
                "spearman_vs_csa": spearman,
                "kendall_vs_csa": kendall,
                "top10_overlap_with_csa": top10_overlap,
            }
        )

    comparison = pd.DataFrame(comparison_rows)
    comparison.to_csv(out_dir / "baseline_comparison_summary.csv", index=False)

    print(f"[OK] Baseline outputs written to: {out_dir}")


if __name__ == "__main__":
    main()