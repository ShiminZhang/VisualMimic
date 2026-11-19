"""Recompute the summary table from the released per-case verdicts and retrieval ranks."""
from pathlib import Path
import csv
from statistics import median

root = Path(__file__).parent
with (root / "judgments.csv").open() as f:
    verdicts = list(csv.DictReader(f))
with (root / "composite_ranks.csv").open() as f:
    retrieval = list(csv.DictReader(f))
for stage in ("pre_GMR", "C", "A", "B"):
    successes = sum(row[stage] == "success" for row in verdicts)
    print(f"{stage}: {successes}/{len(verdicts)} successes")
    if stage != "pre_GMR":
        ranks = [float(row[stage]) for row in retrieval]
        for k in (1, 5):
            print(f"  R@{k}: {100 * sum(rank <= k for rank in ranks) / len(ranks):.2f}%")
        print(f"  Median rank: {median(ranks):g} ({len(ranks)} queries)")
