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


def minmax_benefit(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    smin, smax = s.min(), s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (s - smin) / (smax - smin)


def minmax_cost(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    smin, smax = s.min(), s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
    return (smax - s) / (smax - smin)


def entropy_weights(X: pd.DataFrame) -> pd.Series:
    X = X.copy().astype(float)
    eps = 1e-12
    P = X.div(X.sum(axis=0) + eps, axis=1)
    m = len(X)
    k = 1.0 / np.log(m)
    E = -k * (P * np.log(P + eps)).sum(axis=0)
    d = 1 - E
    w = d / d.sum()
    return w


def waspas_score(X: pd.DataFrame, w: pd.Series, lam: float = 0.5) -> pd.Series:
    X = X.astype(float)
    w = w.astype(float)
    wsm = (X * w).sum(axis=1)
    wpm = np.prod(np.power(X.replace(0, 1e-12), w), axis=1)
    return lam * wsm + (1 - lam) * wpm


def promethee2_simple(X: pd.DataFrame, directions: dict[str, str]) -> pd.Series:
    X = X.astype(float)
    cols = list(X.columns)
    n = len(X)
    pref = np.zeros((n, n), dtype=float)

    for c in cols:
        vals = X[c].values.astype(float)
        diff = vals[:, None] - vals[None, :]
        # After normalization all criteria are benefit-oriented
        P = (diff > 0).astype(float)
        pref += P

    pref = pref / len(cols)
    phi_plus = pref.mean(axis=1)
    phi_minus = pref.mean(axis=0)
    phi_net = phi_plus - phi_minus
    return pd.Series(phi_net, index=X.index)


def borda_from_ranks(rank_df: pd.DataFrame) -> pd.Series:
    n = len(rank_df)
    pts = pd.DataFrame(index=rank_df.index)
    for c in rank_df.columns:
        pts[c] = n - pd.to_numeric(rank_df[c], errors="coerce") + 1
    return pts.sum(axis=1)


def topk_overlap(df: pd.DataFrame, rank_a: str, rank_b: str, k: int = 3) -> int:
    a = set(df.nsmallest(k, rank_a)["alternative_id"])
    b = set(df.nsmallest(k, rank_b)["alternative_id"])
    return len(a.intersection(b))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", required=True)
    parser.add_argument("--outdir", default="outputs\\external_pv_now")
    args = parser.parse_args()

    csv_path = Path(args.csv).resolve()
    out_dir = Path(args.outdir).resolve()
    fig_dir = out_dir / "figures"
    ensure_dir(out_dir)
    ensure_dir(fig_dir)

    df = pd.read_csv(csv_path)

    alt_col = "alternative_id"
    criteria = [
        "CO2_kg", "EPBT_yr", "Cost_per_W", "Degradation_pct_per_yr",
        "Warranty_yr", "Lifespan_yr", "Efficiency_pct", "Tcoef_pct_per_C",
        "Weight_kg_per_m2", "LIP"
    ]

    directions = {
        "CO2_kg": "cost",
        "EPBT_yr": "cost",
        "Cost_per_W": "cost",
        "Degradation_pct_per_yr": "cost",
        "Warranty_yr": "benefit",
        "Lifespan_yr": "benefit",
        "Efficiency_pct": "benefit",
        "Tcoef_pct_per_C": "cost",
        "Weight_kg_per_m2": "cost",
        "LIP": "benefit",
    }

    X = df[criteria].copy()

    Xn = pd.DataFrame(index=df.index)
    for c in criteria:
        if directions[c] == "benefit":
            Xn[c] = minmax_benefit(X[c])
        else:
            Xn[c] = minmax_cost(X[c])

    ew = entropy_weights(Xn)

    df_out = pd.DataFrame({"alternative_id": df[alt_col]})
    df_out["entropy_weighted_score"] = (Xn * ew).sum(axis=1)
    df_out["entropy_rank"] = descending_rank(df_out["entropy_weighted_score"])

    df_out["waspas_score"] = waspas_score(Xn, ew, lam=0.5)
    df_out["waspas_rank"] = descending_rank(df_out["waspas_score"])

    df_out["promethee_score"] = promethee2_simple(Xn, directions)
    df_out["promethee_rank"] = descending_rank(df_out["promethee_score"])

    # CSA weights from inter-method agreement
    rank_mat = df_out[["entropy_rank", "waspas_rank", "promethee_rank"]].copy()
    rho_e_w = rank_mat["entropy_rank"].corr(rank_mat["waspas_rank"], method="spearman")
    rho_e_p = rank_mat["entropy_rank"].corr(rank_mat["promethee_rank"], method="spearman")
    rho_w_p = rank_mat["waspas_rank"].corr(rank_mat["promethee_rank"], method="spearman")

    rel_entropy = np.nanmean([rho_e_w, rho_e_p])
    rel_waspas = np.nanmean([rho_e_w, rho_w_p])
    rel_prom = np.nanmean([rho_e_p, rho_w_p])

    rel = pd.Series(
        {
            "Entropy": rel_entropy,
            "WASPAS": rel_waspas,
            "PROMETHEE": rel_prom,
        }
    )
    rel = rel / rel.sum()

    df_out["entropy_norm"] = minmax_benefit(df_out["entropy_weighted_score"])
    df_out["waspas_norm"] = minmax_benefit(df_out["waspas_score"])
    df_out["promethee_norm"] = minmax_benefit(df_out["promethee_score"])

    df_out["csa_score"] = (
        rel["Entropy"] * df_out["entropy_norm"] +
        rel["WASPAS"] * df_out["waspas_norm"] +
        rel["PROMETHEE"] * df_out["promethee_norm"]
    )
    df_out["csa_rank"] = descending_rank(df_out["csa_score"])

    df_out["borda_score"] = borda_from_ranks(df_out[["entropy_rank", "waspas_rank", "promethee_rank"]])
    df_out["borda_rank"] = descending_rank(df_out["borda_score"])

    df_out["mean_score_score"] = df_out[["entropy_norm", "waspas_norm", "promethee_norm"]].mean(axis=1)
    df_out["mean_score_rank"] = descending_rank(df_out["mean_score_score"])

    df_out = df_out.sort_values("csa_rank").reset_index(drop=True)
    df_out.to_csv(out_dir / "pv_external_results.csv", index=False)

    comp_rows = []
    for name, rank_col in [
        ("Borda Count", "borda_rank"),
        ("Mean normalized score", "mean_score_rank"),
    ]:
        comp_rows.append(
            {
                "aggregation_rule": name,
                "spearman_vs_csa": df_out[rank_col].corr(df_out["csa_rank"], method="spearman"),
                "kendall_vs_csa": df_out[rank_col].corr(df_out["csa_rank"], method="kendall"),
                "top3_overlap_with_csa": topk_overlap(df_out, rank_col, "csa_rank", 3),
            }
        )

    comp = pd.DataFrame(comp_rows)
    comp.to_csv(out_dir / "pv_external_comparison.csv", index=False)

    table7 = pd.DataFrame(
        [
            {
                "dataset": "PV technologies (Kaur et al., 2025)",
                "domain": "renewable_energy",
                "n_alternatives": int(df_out["alternative_id"].nunique()),
                "n_criteria": len(criteria),
                "spearman_borda_vs_csa": comp.loc[comp["aggregation_rule"] == "Borda Count", "spearman_vs_csa"].iloc[0],
                "spearman_mean_score_vs_csa": comp.loc[comp["aggregation_rule"] == "Mean normalized score", "spearman_vs_csa"].iloc[0],
                "top3_borda_overlap": comp.loc[comp["aggregation_rule"] == "Borda Count", "top3_overlap_with_csa"].iloc[0],
                "top3_mean_score_overlap": comp.loc[comp["aggregation_rule"] == "Mean normalized score", "top3_overlap_with_csa"].iloc[0],
                "performance_note": "Cross-domain external validation on an open renewable-energy decision matrix.",
            }
        ]
    )
    table7.to_csv(out_dir / "Table_7_final_real_external.csv", index=False)

    rel_out = rel.reset_index()
    rel_out.columns = ["method_name", "reliability_weight"]
    rel_out.to_csv(out_dir / "pv_external_reliability.csv", index=False)

    plt.figure(figsize=(8, 5))
    plot_df = pd.DataFrame(
        {
            "rule": ["Borda Count", "Mean normalized score"],
            "spearman_vs_csa": comp["spearman_vs_csa"].values,
            "kendall_vs_csa": comp["kendall_vs_csa"].values,
        }
    )
    plt.plot(plot_df["rule"], plot_df["spearman_vs_csa"], marker="o", label="Spearman vs CSA")
    plt.plot(plot_df["rule"], plot_df["kendall_vs_csa"], marker="o", label="Kendall vs CSA")
    plt.xticks(rotation=20, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(fig_dir / "Figure_5_external_validation_real.png", dpi=300, bbox_inches="tight")
    plt.close()

    print(f"[OK] External PV validation outputs written to: {out_dir}")


if __name__ == "__main__":
    main()