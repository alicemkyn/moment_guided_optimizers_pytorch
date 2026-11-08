"""Paired analysis of reconstruction summaries.

Refuses archived result files. Does not treat a numerical match to the paper
as success or failure of this code.
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

from reconstruction.stats import percentile_bootstrap, wilcoxon_two_sided


def load_summary(path: Path) -> list[dict[str, str]]:
    provenance = path.parent / "provenance.json"
    if not provenance.exists():
        raise SystemExit(f"{path} has no reconstruction provenance.json")
    meta = json.loads(provenance.read_text())
    if meta.get("artifact_class") != "reconstruction" or not meta.get("not_historical"):
        raise SystemExit(f"{path} is not a reconstruction log")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def paired(rows: list[dict[str, str]], left: str, right: str) -> list[float]:
    by_opt: dict[str, dict[int, float]] = {}
    for row in rows:
        by_opt.setdefault(row["optimizer"], {})[int(row["seed"])] = float(row["test_acc_at_best_val"])
    if left not in by_opt or right not in by_opt:
        raise SystemExit(f"Need both {left} and {right} in the summary.")
    seeds = sorted(set(by_opt[left]) & set(by_opt[right]))
    return [by_opt[left][seed] - by_opt[right][seed] for seed in seeds]


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description="Analyze a reconstruction summary. Not the archived CSVs.")
    ap.add_argument("--summary", required=True)
    ap.add_argument("--left", default="mgosc")
    ap.add_argument("--right", default="adamw")
    args = ap.parse_args(argv)
    path = Path(args.summary)
    if "artifacts/results_csv" in path.as_posix():
        raise SystemExit("Refusing to analyze archived CSVs as if this code had produced them.")
    deltas = paired(load_summary(path), args.left, args.right)
    report = {
        "artifact_class": "reconstruction",
        "not_a_historical_comparison": True,
        "left": args.left,
        "right": args.right,
        "deltas": deltas,
        "bootstrap": percentile_bootstrap(deltas) if deltas else None,
        "wilcoxon_two_sided": wilcoxon_two_sided(deltas) if len(deltas) <= 16 else "n too large for exact enumeration in this script",
    }
    text = json.dumps(report, indent=2)
    print(text)
    path.with_name("paired_report.json").write_text(text + "\n")


if __name__ == "__main__":
    main()
