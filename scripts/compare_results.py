"""Check whether team evaluation files describe the SAME experiment.

Usage: python scripts/compare_results.py member1/evaluation.json member2/evaluation.json
Different GPUs may differ numerically. This tool does not pretend results must be bitwise equal.
"""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("results", type=Path, nargs="+")
    args = parser.parse_args()
    records = [json.loads(p.read_text(encoding="utf-8")) for p in args.results]
    reference = records[0]
    keys = ("protocol_sha256", "checkpoint_sha256", "mass_observed", "quick_smoke_only",
            "episode_seconds", "episodes", "batch_num_envs", "training_seed", "evaluation_seeds_used",
            "support_mode", "task_schema", "observation_schema", "observation_variant")
    mismatches = []
    for path, result in zip(args.results, records):
        for key in keys:
            if key not in result:
                mismatches.append(f"{path}: required {key} is missing")
    for path, result in zip(args.results[1:], records[1:]):
        for key in keys:
            if result.get(key) != reference.get(key):
                mismatches.append(f"{path}: {key} differs")
        for key in ("source_sha256", "versions", "device"):
            if result["runtime"][key] != reference["runtime"][key]:
                mismatches.append(f"{path}: runtime.{key} differs")
    output = {"comparable": not mismatches, "mismatches": mismatches,
              "runs": [{"member": r["member"], "success_rate": r["success_rate"],
                         "device": r["runtime"]["device"], "gpu": r["runtime"]["gpu"],
                         "quick_smoke_only": r["quick_smoke_only"]} for r in records],
              "note": "Comparable configuration does not guarantee identical cross-hardware trajectories."}
    print(json.dumps(output, indent=2))
    if mismatches:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

