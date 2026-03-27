from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_excel(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def standardize_alternative_id(series: pd.Series) -> pd.Series:
    s = series.astype(str).str.strip()

    def fix_one(x: str) -> str:
        x = x.strip()
        if x == "" or x.lower() == "nan":
            return x

        x_upper = x.upper()

        if x_upper.startswith("A"):
            suffix = x_upper[1:].strip()
            if suffix.replace(".0", "").isdigit():
                return f"A{int(float(suffix))}"
            return x_upper

        if x.replace(".0", "").isdigit():
            return f"A{int(float(x))}"

        return x_upper

    return s.apply(fix_one)


def descending_rank(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    return s.rank(method="min", ascending=False).astype("Int64")


def minmax_norm(series: pd.Series) -> pd.Series:
    s = pd.to_numeric(series, errors="coerce")
    smin = s.min()
    smax = s.max()
    if pd.isna(smin) or pd.isna(smax) or smax == smin:
        return pd.Series([0.0] * len(s), index=s.index, dtype=float)
    return (s - smin) / (smax - smin)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--outdir", default="outputs\\core_now")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx).resolve()
    out_dir = Path(args.outdir).resolve()
    ensure_dir(out_dir)

    perf = safe_read_excel(xlsx_path, "Method_Perf")
    csa = safe_read_excel(xlsx_path, "CSA_Integration")
    rel = safe_read_excel(xlsx_path, "Method_Reliability")
    rmse = safe_read_excel(xlsx_path, "RMSE")
    spearman = safe_read_excel(xlsx_path, "Spearman_vs_CSA")
    csi = safe_read_excel(xlsx_path, "CSI")

    if perf.empty or csa.empty:
        raise ValueError("Required sheets missing or empty: Method_Perf / CSA_Integration")

    perf = perf.rename(columns={perf.columns[0]: "alternative_id"})
    csa = csa.rename(columns={csa.columns[0]: "alternative_id"})

    perf["alternative_id"] = standardize_alternative_id(perf["alternative_id"])
    csa["alternative_id"] = standardize_alternative_id(csa["alternative_id"])

    core = perf.merge(
        csa[["alternative_id", "CSA_score", "CSA_rank"]],
        on="alternative_id",
        how="left",
    )

    core = core.rename(
        columns={
            "Entropy_perf": "entropy_score",
            "WASPAS": "waspas_score",
            "PROMETHEE": "promethee_score",
            "CSA_score": "csa_score",
            "CSA_rank": "csa_rank",
        }
    )

    for col in ["entropy_score", "waspas_score", "promethee_score", "csa_score", "csa_rank"]:
        if col in core.columns:
            core[col] = pd.to_numeric(core[col], errors="coerce")

    # Recompute constituent ranks directly from scores
    core["entropy_rank"] = descending_rank(core["entropy_score"])
    core["waspas_rank"] = descending_rank(core["waspas_score"])
    core["promethee_rank"] = descending_rank(core["promethee_score"])

    # Recompute Borda from recomputed constituent ranks
    core["borda_points"] = (
        (len(core) - core["entropy_rank"] + 1) +
        (len(core) - core["waspas_rank"] + 1) +
        (len(core) - core["promethee_rank"] + 1)
    )
    core["borda_rank"] = descending_rank(core["borda_points"])

    core = core.sort_values("csa_rank", na_position="last").reset_index(drop=True)
    core.to_csv(out_dir / "core_method_outputs.csv", index=False)
    core.head(25).to_csv(out_dir / "top25_csa_results.csv", index=False)

    if not rel.empty:
        rel.to_csv(out_dir / "method_reliability_raw.csv", index=False)
    if not rmse.empty:
        rmse.to_csv(out_dir / "rmse_raw.csv", index=False)
    if not spearman.empty:
        spearman.to_csv(out_dir / "spearman_vs_csa_raw.csv", index=False)
    if not csi.empty:
        csi.to_csv(out_dir / "csi_raw.csv", index=False)

    summary_rows = []

    if not rel.empty and {"Method", "RelWeight"}.issubset(rel.columns):
        tmp = rel.copy()
        tmp["RelWeight"] = pd.to_numeric(tmp["RelWeight"], errors="coerce")
        for _, row in tmp.iterrows():
            summary_rows.append(
                {"metric_group": "reliability_weight", "item": row["Method"], "value": row["RelWeight"]}
            )

    if not rmse.empty and {"Method", "RMSE"}.issubset(rmse.columns):
        tmp = rmse.copy()
        tmp["RMSE"] = pd.to_numeric(tmp["RMSE"], errors="coerce")
        for _, row in tmp.iterrows():
            summary_rows.append(
                {"metric_group": "rmse_vs_csa", "item": row["Method"], "value": row["RMSE"]}
            )

    if not spearman.empty and {"Method", "Spearman_vs_CSA"}.issubset(spearman.columns):
        tmp = spearman.copy()
        tmp["Spearman_vs_CSA"] = pd.to_numeric(tmp["Spearman_vs_CSA"], errors="coerce")
        for _, row in tmp.iterrows():
            summary_rows.append(
                {"metric_group": "spearman_vs_csa", "item": row["Method"], "value": row["Spearman_vs_CSA"]}
            )

    if not csi.empty:
        numeric_values = pd.to_numeric(csi.stack(), errors="coerce").dropna().tolist()
        if numeric_values:
            summary_rows.append(
                {"metric_group": "csi", "item": "reported_csi", "value": numeric_values[0]}
            )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_dir / "core_metrics_summary.csv", index=False)

    merge_check = pd.DataFrame(
        {
            "source": ["perf", "csa", "core"],
            "n_rows": [len(perf), len(csa), len(core)],
            "n_unique_ids": [
                perf["alternative_id"].nunique(),
                csa["alternative_id"].nunique(),
                core["alternative_id"].nunique(),
            ],
        }
    )
    merge_check.to_csv(out_dir / "merge_check.csv", index=False)

    print(f"[OK] Core outputs written to: {out_dir}")


if __name__ == "__main__":
    main()