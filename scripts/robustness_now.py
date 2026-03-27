from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


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


def kendall_like_from_ranks(a: pd.Series, b: pd.Series) -> float:
    return pd.to_numeric(a, errors="coerce").corr(pd.to_numeric(b, errors="coerce"), method="kendall")


def spearman_from_ranks(a: pd.Series, b: pd.Series) -> float:
    return pd.to_numeric(a, errors="coerce").corr(pd.to_numeric(b, errors="coerce"), method="spearman")


def topk_overlap(df: pd.DataFrame, rank_col_a: str, rank_col_b: str, k: int = 10) -> int:
    a = set(df.nsmallest(k, rank_col_a)["alternative_id"])
    b = set(df.nsmallest(k, rank_col_b)["alternative_id"])
    return len(a.intersection(b))


def compute_csa_from_scores(df: pd.DataFrame, weights: dict[str, float], use_minmax: bool = False) -> pd.DataFrame:
    x = df.copy()

    score_cols = {
        "Entropy": "entropy_score",
        "WASPAS": "waspas_score",
        "PROMETHEE": "promethee_score",
    }

    for m, col in score_cols.items():
        vals = pd.to_numeric(x[col], errors="coerce")
        x[f"{m}_base"] = minmax_norm(vals) if use_minmax else vals

    x["scenario_score"] = (
        weights["Entropy"] * x["Entropy_base"] +
        weights["WASPAS"] * x["WASPAS_base"] +
        weights["PROMETHEE"] * x["PROMETHEE_base"]
    )
    x["scenario_rank"] = descending_rank(x["scenario_score"])
    return x


def summarize_against_base(base: pd.DataFrame, scenario: pd.DataFrame, scenario_name: str, scenario_type: str) -> dict:
    merged = base[["alternative_id", "csa_rank"]].merge(
        scenario[["alternative_id", "scenario_rank"]],
        on="alternative_id",
        how="inner",
    )
    rank_shift = (merged["csa_rank"] - merged["scenario_rank"]).abs()

    return {
        "scenario_name": scenario_name,
        "scenario_type": scenario_type,
        "spearman_vs_base": spearman_from_ranks(merged["csa_rank"], merged["scenario_rank"]),
        "kendall_vs_base": kendall_like_from_ranks(merged["csa_rank"], merged["scenario_rank"]),
        "top10_overlap_with_base": topk_overlap(
            merged.rename(columns={"csa_rank": "rank_a", "scenario_rank": "rank_b"}),
            "rank_a", "rank_b", 10
        ),
        "mean_rank_shift": float(rank_shift.mean()),
        "max_rank_shift": float(rank_shift.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corecsv", required=True)
    parser.add_argument("--outdir", default="outputs\\robustness_now")
    args = parser.parse_args()

    core_path = Path(args.corecsv).resolve()
    out_dir = Path(args.outdir).resolve()
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    df = pd.read_csv(core_path)

    required = [
        "alternative_id",
        "entropy_score",
        "waspas_score",
        "promethee_score",
        "csa_score",
        "csa_rank",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in core CSV: {missing}")

    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    base_weights = {"Entropy": 0.3196367027, "WASPAS": 0.3398852754, "PROMETHEE": 0.3404780218}
    base = df.copy()

    scenarios = []

    # A) weight perturbation
    weight_grid = [
        ("equal_weights", {"Entropy": 1/3, "WASPAS": 1/3, "PROMETHEE": 1/3}, False),
        ("entropy_up", {"Entropy": 0.40, "WASPAS": 0.30, "PROMETHEE": 0.30}, False),
        ("waspas_up", {"Entropy": 0.28, "WASPAS": 0.44, "PROMETHEE": 0.28}, False),
        ("promethee_up", {"Entropy": 0.28, "WASPAS": 0.28, "PROMETHEE": 0.44}, False),
        ("reported_weights", base_weights, False),
    ]

    for name, w, use_minmax in weight_grid:
        scen = compute_csa_from_scores(df, w, use_minmax=use_minmax)
        scenarios.append(summarize_against_base(base, scen, name, "weighting"))

    # B) normalization sensitivity
    for name, w, use_minmax in [
        ("reported_weights_minmax", base_weights, True),
        ("equal_weights_minmax", {"Entropy": 1/3, "WASPAS": 1/3, "PROMETHEE": 1/3}, True),
    ]:
        scen = compute_csa_from_scores(df, w, use_minmax=use_minmax)
        scenarios.append(summarize_against_base(base, scen, name, "normalization"))

    # C) leave-one-method-out
    lomo_defs = {
        "drop_entropy": {"Entropy": 0.0, "WASPAS": 0.5, "PROMETHEE": 0.5},
        "drop_waspas": {"Entropy": 0.5, "WASPAS": 0.0, "PROMETHEE": 0.5},
        "drop_promethee": {"Entropy": 0.5, "WASPAS": 0.5, "PROMETHEE": 0.0},
    }
    for name, w in lomo_defs.items():
        scen = compute_csa_from_scores(df, w, use_minmax=False)
        scenarios.append(summarize_against_base(base, scen, name, "leave_one_method_out"))

    # D) rank reversal style tests via removal and reinsertion-free re-ranking
    rr_rows = []
    remove_sets = {
        "remove_top1": set(base.nsmallest(1, "csa_rank")["alternative_id"]),
        "remove_top5": set(base.nsmallest(5, "csa_rank")["alternative_id"]),
        "remove_bottom5": set(base.nlargest(5, "csa_rank")["alternative_id"]),
        "remove_random10": set(base.sample(10, random_state=42)["alternative_id"]),
    }

    for name, removed in remove_sets.items():
        sub = df.loc[~df["alternative_id"].isin(removed)].copy()
        scen = compute_csa_from_scores(sub, base_weights, use_minmax=False)

        base_sub = base.loc[base["alternative_id"].isin(scen["alternative_id"])].copy()
        merged = base_sub[["alternative_id", "csa_rank"]].merge(
            scen[["alternative_id", "scenario_rank"]],
            on="alternative_id",
            how="inner",
        )
        rank_shift = (merged["csa_rank"] - merged["scenario_rank"]).abs()

        rr_rows.append(
            {
                "scenario_name": name,
                "scenario_type": "rank_reversal",
                "n_removed": len(removed),
                "spearman_vs_base": spearman_from_ranks(merged["csa_rank"], merged["scenario_rank"]),
                "kendall_vs_base": kendall_like_from_ranks(merged["csa_rank"], merged["scenario_rank"]),
                "top10_overlap_with_base": topk_overlap(
                    merged.rename(columns={"csa_rank": "rank_a", "scenario_rank": "rank_b"}),
                    "rank_a", "rank_b", 10
                ),
                "mean_rank_shift": float(rank_shift.mean()),
                "max_rank_shift": float(rank_shift.max()),
            }
        )

    scen_df = pd.DataFrame(scenarios)
    rr_df = pd.DataFrame(rr_rows)

    scen_df.to_csv(out_dir / "robustness_summary.csv", index=False)
    rr_df.to_csv(out_dir / "rank_reversal_summary.csv", index=False)

    # Table 6 input
    table6 = pd.concat([scen_df, rr_df], ignore_index=True)
    table6.to_csv(out_dir / "table6_input.csv", index=False)

    # Figure 3: robustness heatmap input + image
    heat = table6.pivot_table(
        index="scenario_name",
        values=["spearman_vs_base", "kendall_vs_base", "top10_overlap_with_base", "mean_rank_shift", "max_rank_shift"],
        aggfunc="first",
    )
    heat.to_csv(out_dir / "figure3_heatmap_input.csv")

    plt.figure(figsize=(10, 6))
    plt.imshow(heat.fillna(0).values, aspect="auto")
    plt.xticks(range(len(heat.columns)), heat.columns, rotation=45, ha="right")
    plt.yticks(range(len(heat.index)), heat.index)
    plt.colorbar()
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_3_robustness_heatmap.png", dpi=300, bbox_inches="tight")
    plt.close()

    # Figure 4: rank shift plot from rank reversal
    plt.figure(figsize=(8, 5))
    plt.plot(rr_df["scenario_name"], rr_df["mean_rank_shift"], marker="o", label="Mean rank shift")
    plt.plot(rr_df["scenario_name"], rr_df["max_rank_shift"], marker="o", label="Max rank shift")
    plt.xticks(rotation=45, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_4_rank_shift.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] Robustness outputs written to: {out_dir}")


if __name__ == "__main__":
    main()