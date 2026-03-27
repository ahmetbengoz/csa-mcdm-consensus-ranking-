from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def safe_read_excel(xlsx_path: Path, sheet_name: str) -> pd.DataFrame:
    try:
        return pd.read_excel(xlsx_path, sheet_name=sheet_name)
    except Exception:
        return pd.DataFrame()


def build_criteria_master(criteria_weights: pd.DataFrame) -> pd.DataFrame:
    main_df = (
        criteria_weights.groupby("MainCode", as_index=False)["AbsWeight"]
        .sum()
        .rename(columns={"MainCode": "main_code", "AbsWeight": "main_weight"})
        .sort_values("main_code")
        .reset_index(drop=True)
    )

    main_df["main_name"] = main_df["main_code"]
    main_df["main_type"] = "hierarchical_main_criterion"
    main_df["main_weight_source"] = "Criteria_Weights.AbsWeight aggregation"
    main_df["is_final_frozen"] = 1

    return main_df[
        [
            "main_code",
            "main_name",
            "main_type",
            "main_weight",
            "main_weight_source",
            "is_final_frozen",
        ]
    ]


def build_subcriteria_master(criteria_weights: pd.DataFrame) -> pd.DataFrame:
    sub_df = criteria_weights.copy().rename(
        columns={
            "SubCode": "sub_code",
            "MainCode": "parent_main_code",
            "PropOfMain": "share_within_main",
            "AbsWeight": "absolute_weight",
        }
    )

    sub_df["sub_name"] = sub_df["sub_code"]
    sub_df["criterion_direction"] = "benefit"

    sub_num = (
        sub_df["sub_code"]
        .astype(str)
        .str.extract(r"(\d+)", expand=False)
        .fillna("0")
        .astype(int)
    )

    sub_df["cross_cutting_flag"] = sub_num.between(45, 55).astype(int)
    sub_df["assignment_rule"] = np.where(
        sub_df["cross_cutting_flag"] == 1,
        "dominant_parent_frozen_from_workbook",
        "direct_parent_from_workbook",
    )
    sub_df["is_final_frozen"] = 1

    return sub_df[
        [
            "sub_code",
            "sub_name",
            "parent_main_code",
            "share_within_main",
            "absolute_weight",
            "criterion_direction",
            "cross_cutting_flag",
            "assignment_rule",
            "is_final_frozen",
        ]
    ].sort_values(["parent_main_code", "sub_code"]).reset_index(drop=True)


def build_alternatives_master(alternative_scores: pd.DataFrame) -> pd.DataFrame:
    df = alternative_scores.copy()

    if len(df.columns) == 0:
        raise ValueError("Alternative_Scores sheet is empty.")

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "alternative_id"})

    alt_df = pd.DataFrame({"alternative_id": df["alternative_id"].astype(str)})
    alt_df["alternative_label"] = alt_df["alternative_id"]
    alt_df["alternative_domain"] = "defense_inspired"
    alt_df["alternative_active_flag"] = 1
    alt_df["notes"] = ""

    return alt_df.drop_duplicates().reset_index(drop=True)


def build_alternative_scores_long(alternative_scores: pd.DataFrame) -> pd.DataFrame:
    df = alternative_scores.copy()

    if len(df.columns) == 0:
        raise ValueError("Alternative_Scores sheet is empty.")

    first_col = df.columns[0]
    df = df.rename(columns={first_col: "alternative_id"})

    sub_cols = [c for c in df.columns if c != "alternative_id"]

    long_df = df.melt(
        id_vars=["alternative_id"],
        value_vars=sub_cols,
        var_name="sub_code",
        value_name="raw_score",
    )

    long_df["raw_score"] = pd.to_numeric(long_df["raw_score"], errors="coerce")
    long_df["raw_score_source"] = "Alternative_Scores"
    long_df["raw_score_scale"] = "continuous_numeric"
    long_df["is_imputed"] = 0
    long_df["processed_score"] = long_df["raw_score"]

    def safe_minmax(s: pd.Series) -> pd.Series:
        s = pd.to_numeric(s, errors="coerce")
        smin = s.min()
        smax = s.max()
        if pd.isna(smin) or pd.isna(smax) or smax == smin:
            return pd.Series(np.zeros(len(s)), index=s.index, dtype=float)
        return (s - smin) / (smax - smin)

    long_df["normalized_score"] = long_df.groupby("sub_code")["processed_score"].transform(safe_minmax)
    long_df["normalization_scheme"] = "minmax_by_subcriterion"

    return long_df[
        [
            "alternative_id",
            "sub_code",
            "raw_score",
            "raw_score_source",
            "raw_score_scale",
            "is_imputed",
            "processed_score",
            "normalized_score",
            "normalization_scheme",
        ]
    ]


def build_method_outputs(xlsx_path: Path) -> pd.DataFrame:
    perf = safe_read_excel(xlsx_path, "Method_Perf")
    ranks = safe_read_excel(xlsx_path, "Ranks_and_Borda")
    csa = safe_read_excel(xlsx_path, "CSA_Integration")

    if perf.empty or ranks.empty or csa.empty:
        raise ValueError("One or more required sheets are empty: Method_Perf, Ranks_and_Borda, CSA_Integration")

    perf = perf.rename(columns={perf.columns[0]: "alternative_id"})
    ranks = ranks.rename(columns={ranks.columns[0]: "alternative_id"})
    csa = csa.rename(columns={csa.columns[0]: "alternative_id"})

    perf["alternative_id"] = perf["alternative_id"].astype(str).str.strip()
    ranks["alternative_id"] = ranks["alternative_id"].astype(str).str.strip()
    csa["alternative_id"] = csa["alternative_id"].astype(str).str.strip()

    out = perf.merge(
        ranks[["alternative_id", "Entropy_rank", "WASPAS_rank", "PROMETHEE_rank", "Borda_rank"]],
        on="alternative_id",
        how="left",
    ).merge(
        csa[["alternative_id", "CSA_score", "CSA_rank"]],
        on="alternative_id",
        how="left",
    )

    out = out.rename(
        columns={
            "Entropy_perf": "entropy_score",
            "WASPAS": "waspas_score",
            "PROMETHEE": "promethee_score",
            "Entropy_rank": "entropy_rank",
            "WASPAS_rank": "waspas_rank",
            "PROMETHEE_rank": "promethee_rank",
            "Borda_rank": "borda_rank",
            "CSA_score": "csa_score",
            "CSA_rank": "csa_rank",
        }
    )

    return out.sort_values("alternative_id").reset_index(drop=True)


def build_reliability_weights(xlsx_path: Path) -> pd.DataFrame:
    df = safe_read_excel(xlsx_path, "Method_Reliability").copy()

    if df.empty:
        raise ValueError("Method_Reliability sheet is empty.")

    df = df.rename(columns={"Method": "method_name", "RelWeight": "reliability_weight"})
    df["reliability_weight"] = pd.to_numeric(df["reliability_weight"], errors="coerce")
    df["weight_scheme"] = "workbook_reported"

    return df


def write_metadata(
    out_dir: Path,
    n_alternatives: int,
    n_main: int,
    n_sub: int,
) -> None:
    metadata = {
        "dataset_name": "CSA_Supplementary_Illustrative",
        "n_alternatives": n_alternatives,
        "n_main_criteria": n_main,
        "n_subcriteria": n_sub,
        "normalization_default": "minmax_by_subcriterion",
        "notes": "Processed inputs built directly from workbook sheets.",
    }
    (out_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--xlsx", required=True)
    parser.add_argument("--outdir", default="data\\processed_now")
    args = parser.parse_args()

    xlsx_path = Path(args.xlsx).resolve()
    out_dir = Path(args.outdir).resolve()
    ensure_dir(out_dir)

    if not xlsx_path.exists():
        raise FileNotFoundError(f"Workbook not found: {xlsx_path}")

    criteria_weights = safe_read_excel(xlsx_path, "Criteria_Weights")
    alternative_scores = safe_read_excel(xlsx_path, "Alternative_Scores")

    if criteria_weights.empty:
        raise ValueError("Criteria_Weights sheet is empty.")
    if alternative_scores.empty:
        raise ValueError("Alternative_Scores sheet is empty.")

    criteria_master = build_criteria_master(criteria_weights)
    subcriteria_master = build_subcriteria_master(criteria_weights)
    alternatives_master = build_alternatives_master(alternative_scores)
    alternative_scores_long = build_alternative_scores_long(alternative_scores)
    method_outputs = build_method_outputs(xlsx_path)
    reliability_weights = build_reliability_weights(xlsx_path)

    criteria_master.to_csv(out_dir / "criteria_master.csv", index=False)
    subcriteria_master.to_csv(out_dir / "subcriteria_master.csv", index=False)
    alternatives_master.to_csv(out_dir / "alternatives_master.csv", index=False)
    alternative_scores_long.to_csv(out_dir / "alternative_scores_long.csv", index=False)
    method_outputs.to_csv(out_dir / "method_outputs_master.csv", index=False)
    reliability_weights.to_csv(out_dir / "reliability_weights_master.csv", index=False)

    write_metadata(
        out_dir=out_dir,
        n_alternatives=int(alternatives_master["alternative_id"].nunique()),
        n_main=int(criteria_master["main_code"].nunique()),
        n_sub=int(subcriteria_master["sub_code"].nunique()),
    )

    print(f"[OK] Processed files written to: {out_dir}")


if __name__ == "__main__":
    main()