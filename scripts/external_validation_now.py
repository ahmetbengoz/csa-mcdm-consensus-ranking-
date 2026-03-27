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


def spearman_corr(a: pd.Series, b: pd.Series) -> float:
    return pd.to_numeric(a, errors="coerce").corr(pd.to_numeric(b, errors="coerce"), method="spearman")


def kendall_corr(a: pd.Series, b: pd.Series) -> float:
    return pd.to_numeric(a, errors="coerce").corr(pd.to_numeric(b, errors="coerce"), method="kendall")


def topk_overlap(df: pd.DataFrame, rank_a: str, rank_b: str, k: int = 10) -> int:
    a = set(df.nsmallest(k, rank_a)["alternative_id"])
    b = set(df.nsmallest(k, rank_b)["alternative_id"])
    return len(a.intersection(b))


def compute_consensus(df: pd.DataFrame, weights: dict[str, float], normalize_each: bool) -> pd.DataFrame:
    x = df.copy()

    if normalize_each:
        x["entropy_used"] = minmax_norm(x["entropy_score"])
        x["waspas_used"] = minmax_norm(x["waspas_score"])
        x["promethee_used"] = minmax_norm(x["promethee_score"])
    else:
        x["entropy_used"] = pd.to_numeric(x["entropy_score"], errors="coerce")
        x["waspas_used"] = pd.to_numeric(x["waspas_score"], errors="coerce")
        x["promethee_used"] = pd.to_numeric(x["promethee_score"], errors="coerce")

    x["scenario_score"] = (
        weights["Entropy"] * x["entropy_used"] +
        weights["WASPAS"] * x["waspas_used"] +
        weights["PROMETHEE"] * x["promethee_used"]
    )
    x["scenario_rank"] = descending_rank(x["scenario_score"])
    return x


def compare_to_base(base: pd.DataFrame, scenario: pd.DataFrame, name: str, domain: str) -> dict:
    merged = base[["alternative_id", "csa_rank"]].merge(
        scenario[["alternative_id", "scenario_rank"]],
        on="alternative_id",
        how="inner",
    )

    rank_shift = (merged["csa_rank"] - merged["scenario_rank"]).abs()

    return {
        "dataset": name,
        "domain": domain,
        "n_alternatives": int(merged["alternative_id"].nunique()),
        "n_criteria_proxy": 3,
        "spearman_vs_csa": spearman_corr(merged["csa_rank"], merged["scenario_rank"]),
        "kendall_vs_csa": kendall_corr(merged["csa_rank"], merged["scenario_rank"]),
        "top10_overlap_with_csa": topk_overlap(
            merged.rename(columns={"csa_rank": "rank_a", "scenario_rank": "rank_b"}),
            "rank_a",
            "rank_b",
            10,
        ),
        "mean_rank_shift": float(rank_shift.mean()),
        "max_rank_shift": float(rank_shift.max()),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corecsv", required=True)
    parser.add_argument("--outdir", default="outputs\\external_validation_now")
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
        "csa_rank",
    ]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in core CSV: {missing}")

    for c in required[1:]:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    base = df.copy()

    reported_weights = {"Entropy": 0.3196367027, "WASPAS": 0.3398852754, "PROMETHEE": 0.3404780218}
    equal_weights = {"Entropy": 1/3, "WASPAS": 1/3, "PROMETHEE": 1/3}
    entropy_favor = {"Entropy": 0.50, "WASPAS": 0.25, "PROMETHEE": 0.25}
    promethee_favor = {"Entropy": 0.20, "WASPAS": 0.20, "PROMETHEE": 0.60}

    scenarios = [
        ("Reported-weight consensus", reported_weights, False, "internal_validation"),
        ("Equal-weight consensus", equal_weights, False, "internal_validation"),
        ("Reported-weight minmax consensus", reported_weights, True, "internal_validation"),
        ("Entropy-favoring consensus", entropy_favor, False, "internal_validation"),
        ("PROMETHEE-favoring consensus", promethee_favor, False, "internal_validation"),
    ]

    rows = []
    for name, w, normalize_each, domain in scenarios:
        scen = compute_consensus(df, w, normalize_each)
        rows.append(compare_to_base(base, scen, name, domain))

    result = pd.DataFrame(rows)
    result.to_csv(out_dir / "external_validation_summary.csv", index=False)

    # Table 7 final
    table7 = result.copy()
    table7["performance_note"] = [
        "Reference configuration.",
        "Tests whether CSA materially differs from non-reliability weighting.",
        "Tests normalization sensitivity under a fully score-normalized alternative.",
        "Stress test favoring Entropy-derived information.",
        "Stress test favoring PROMETHEE-derived information.",
    ]
    table7.to_csv(out_dir / "Table_7_final.csv", index=False)

    # Figure 5
    plot_df = table7.copy()

    plt.figure(figsize=(9, 5))
    plt.plot(plot_df["dataset"], plot_df["spearman_vs_csa"], marker="o", label="Spearman vs CSA")
    plt.plot(plot_df["dataset"], plot_df["kendall_vs_csa"], marker="o", label="Kendall vs CSA")
    plt.xticks(rotation=35, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_5_external_validation.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] External validation outputs written to: {out_dir}")


if __name__ == "__main__":
    main()