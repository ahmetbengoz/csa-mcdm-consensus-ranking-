from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--corecsv", required=True)
    parser.add_argument("--baselinecsv", required=True)
    parser.add_argument("--robustcsv", required=True)
    parser.add_argument("--outdir", default="outputs\\frozen_tables_now")
    args = parser.parse_args()

    out_dir = Path(args.outdir).resolve()
    ensure_dir(out_dir)

    core = pd.read_csv(args.corecsv)
    baseline = pd.read_csv(args.baselinecsv)
    robust = pd.read_csv(args.robustcsv)

    # Table 4 final: top 15
    table4 = core[
        [
            "alternative_id",
            "entropy_rank",
            "waspas_rank",
            "promethee_rank",
            "borda_rank",
            "csa_rank",
            "csa_score",
        ]
    ].sort_values("csa_rank").head(15).reset_index(drop=True)
    table4.to_csv(out_dir / "Table_4_final.csv", index=False)

    # Table 5 final: defensible baselines only
    keep = baseline["baseline"].isin(["borda_rank", "mean_score_rank"])
    table5 = baseline.loc[keep].copy().reset_index(drop=True)
    table5.insert(0, "aggregation_rule", ["Borda Count", "Mean normalized score"])
    table5["interpretation"] = [
        "Ordinal rank-only aggregation; expected to diverge from CSA due to compression of value gaps.",
        "Score-space aggregation; closer to CSA but does not incorporate reliability weighting.",
    ]

    csa_row = pd.DataFrame(
        {
            "aggregation_rule": ["CSA"],
            "baseline": ["csa_rank"],
            "spearman_vs_csa": [1.0],
            "kendall_vs_csa": [1.0],
            "top10_overlap_with_csa": [10],
            "interpretation": ["Reference consensus ranking produced by reliability-weighted score integration."],
        }
    )
    table5 = pd.concat([csa_row, table5], ignore_index=True)
    table5.to_csv(out_dir / "Table_5_final.csv", index=False)

    # Table 6 final
    table6 = robust.copy()
    table6.to_csv(out_dir / "Table_6_final.csv", index=False)

    print(f"[OK] Frozen tables written to: {out_dir}")


if __name__ == "__main__":
    main()