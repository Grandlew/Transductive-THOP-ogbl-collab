import argparse
import json
import os
import subprocess
import sys

import pandas as pd
import torch


def run_command(cmd):
    print("\n" + "=" * 120)
    print("Running:")
    print(" ".join(cmd))
    print("=" * 120)

    result = subprocess.run(cmd)

    if result.returncode != 0:
        raise RuntimeError(f"Command failed with return code {result.returncode}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/base.json")
    parser.add_argument("--root", type=str, default="data")
    parser.add_argument("--cache_dir", type=str, default="cache")
    parser.add_argument("--result_dir", type=str, default="results")
    parser.add_argument("--trials", type=int, default=10000)
    parser.add_argument("--force_rebuild", action="store_true")
    args = parser.parse_args()

    os.makedirs(args.result_dir, exist_ok=True)

    seeds = list(range(10))

    for seed in seeds:
        report_path = os.path.join(args.result_dir, f"tt_seed{seed}_report.json")

        if os.path.exists(report_path):
            print(f"Seed {seed} already exists. Skipping:", report_path)
            continue

        cmd = [
            sys.executable,
            "run_thop_collab.py",
            "--config",
            args.config,
            "--root",
            args.root,
            "--cache_dir",
            args.cache_dir,
            "--result_dir",
            args.result_dir,
            "--seed",
            str(seed),
            "--trials",
            str(args.trials),
        ]

        if args.force_rebuild:
            cmd.append("--force_rebuild")

        run_command(cmd)

    rows = []

    for seed in seeds:
        report_path = os.path.join(args.result_dir, f"tt_seed{seed}_report.json")

        if not os.path.exists(report_path):
            raise FileNotFoundError(f"Missing report for seed {seed}: {report_path}")

        with open(report_path, "r") as f:
            report = json.load(f)

        rows.append({
            "seed": seed,
            "valid_hits50": float(report["valid_hits50"]),
            "test_hits50": float(report["test_hits50_local"]),
            "report_path": report_path,
        })

    df = pd.DataFrame(rows)

    valid_tensor = torch.tensor(df["valid_hits50"].values, dtype=torch.float32)
    test_tensor = torch.tensor(df["test_hits50"].values, dtype=torch.float32)

    summary = {
        "model": "Transductive T-HOP Fusion",
        "dataset": "ogbl-collab",
        "metric": "Hits@50",
        "seeds": seeds,
        "num_seeds": len(seeds),
        "valid_mean": float(torch.mean(valid_tensor)),
        "valid_std_unbiased": float(torch.std(valid_tensor, unbiased=True)),
        "test_mean": float(torch.mean(test_tensor)),
        "test_std_unbiased": float(torch.std(test_tensor, unbiased=True)),
        "trials_per_seed": args.trials,
        "config": args.config,
        "protocol": {
            "validation_graph": "train only",
            "test_graph": "train + validation positives",
            "model_selection": "fusion weights selected by validation Hits@50 separately for each seed"
        }
    }

    csv_path = os.path.join(args.result_dir, "ten_seed_results.csv")
    json_path = os.path.join(args.result_dir, "ten_seed_summary.json")

    df.to_csv(csv_path, index=False)

    with open(json_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n10-seed results:")
    print(df)

    print("\n10-seed summary:")
    print(json.dumps(summary, indent=2))

    print("\nSaved:")
    print(csv_path)
    print(json_path)


if __name__ == "__main__":
    main()
