"""Tabulate a hyperparameter sweep: one row per setting, averaged over seeds.

    python -m elevator_rl.tune_report runs/tune/multirun/<date>

Reads every `result.json` under the given directories (from `elevator_rl.tune`
or `elevator_rl.tune_search`), groups the runs by their overrides apart from
`seed`, and prints a Markdown table sorted by final score.
"""

import argparse
import json
import statistics
from collections import defaultdict
from pathlib import Path


def load(dirs):
    return [json.loads(p.read_text()) for d in dirs for p in sorted(Path(d).rglob("result.json"))]


def common_overrides(results):
    """The overrides every run shares (printed once, above the table)."""
    sets = [set(r.get("overrides", [])) for r in results]
    return set.intersection(*sets) if len(sets) > 1 else set()


def setting(result, common=frozenset()):
    overrides = result.get("overrides", [])
    kept = [o for o in overrides if o not in common and not o.startswith("seed=")]
    return " ".join(kept) or "(defaults)"


def summarise(results):
    """Rows of (setting, n_seeds, score mean, score std, best, steps/s, minutes), best first."""
    common = common_overrides(results)
    groups = defaultdict(list)
    for result in results:
        groups[setting(result, common)].append(result)
    rows = []
    for name, runs in groups.items():
        scores = [r["score"] for r in runs]
        rows.append(
            {
                "setting": name,
                "seeds": len(runs),
                "score": statistics.mean(scores),
                "std": statistics.stdev(scores) if len(scores) > 1 else 0.0,
                "best": max(r["best"] for r in runs),
                "steps_per_second": statistics.mean(r["steps_per_second"] for r in runs),
                "minutes": statistics.mean(r["seconds"] for r in runs) / 60,
            }
        )
    return sorted(rows, key=lambda row: -row["score"])


def table(rows):
    lines = [
        "| Setting | Seeds | Final reward | Best eval | PPO steps/s | Minutes |",
        "| --- | --: | --: | --: | --: | --: |",
    ]
    for r in rows:
        lines.append(
            f"| {r['setting']} | {r['seeds']} | {r['score']:.1f} ± {r['std']:.1f} | "
            f"{r['best']:.1f} | {r['steps_per_second']:,.0f} | {r['minutes']:.1f} |"
        )
    return "\n".join(lines)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("dirs", nargs="+")
    args = parser.parse_args(argv)
    results = load(args.dirs)
    common = sorted(common_overrides(results))
    if common:
        print("All runs: " + " ".join(f"`{o}`" for o in common) + "\n")
    rows = summarise(results)
    print(table(rows))
    return rows


if __name__ == "__main__":
    main()
