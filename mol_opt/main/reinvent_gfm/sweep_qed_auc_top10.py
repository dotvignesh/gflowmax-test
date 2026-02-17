from __future__ import annotations

import argparse
import json
import os
import random
from copy import deepcopy
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List

import numpy as np
import torch
import yaml
from tdc import Oracle

path_here = os.path.dirname(os.path.realpath(__file__))
mol_opt_root = os.path.abspath(os.path.join(path_here, "..", ".."))
if mol_opt_root not in os.sys.path:
    os.sys.path.append(mol_opt_root)

from main.optimizer import top_auc  # noqa: E402
from main.reinvent_gfm.run import REINVENT_GFM_Optimizer  # noqa: E402


@dataclass
class SweepArgs:
    smi_file: str | None
    n_jobs: int
    output_dir: str
    patience: int
    max_oracle_calls: int
    freq_log: int
    log_results: bool


def make_optimizer_args(output_dir: str, max_oracle_calls: int, freq_log: int, patience: int) -> SweepArgs:
    return SweepArgs(
        smi_file=None,
        n_jobs=-1,
        output_dir=output_dir,
        patience=patience,
        max_oracle_calls=max_oracle_calls,
        freq_log=freq_log,
        log_results=False,
    )


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)


def run_single_config(
    config: Dict,
    seed: int,
    output_dir: str,
    max_oracle_calls: int,
    freq_log: int,
    patience: int,
) -> Dict:
    set_seed(seed)
    args = make_optimizer_args(
        output_dir=output_dir,
        max_oracle_calls=max_oracle_calls,
        freq_log=freq_log,
        patience=patience,
    )
    oracle = Oracle(name="QED")
    optimizer = REINVENT_GFM_Optimizer(args=args)
    optimizer.oracle.task_label = f"sweep_qed_seed{seed}"
    optimizer._optimize(oracle, config)

    mol_buffer = optimizer.oracle.mol_buffer
    auc10 = top_auc(
        mol_buffer,
        top_n=10,
        finish=False,
        freq_log=args.freq_log,
        max_oracle_calls=args.max_oracle_calls,
    )
    top_sorted = sorted(mol_buffer.items(), key=lambda kv: kv[1][0], reverse=True)
    top_scores = [item[1][0] for item in top_sorted[:10]]
    avg_top10 = float(np.mean(top_scores)) if top_scores else 0.0

    return {
        "seed": seed,
        "auc_top10": float(auc10),
        "avg_top10": float(avg_top10),
        "n_oracle": int(len(mol_buffer)),
    }


def default_base_config() -> Dict:
    cfg_path = os.path.join(path_here, "hparams_default.yaml")
    with open(cfg_path, "r") as f:
        config = yaml.safe_load(f)
    # Keep sweep logs concise.
    config["log_every"] = 0
    config["terminal_only_oracle"] = True
    config["objective_mode"] = "hybrid"
    return config


def build_phase1_candidates(base: Dict) -> List[Dict]:
    candidates = []
    knobs = [
        {"objective_mix_lambda": 0.2, "reinvent_loss_coef": 0.25, "flow_loss_coef": 4.0, "sigma": 100, "beta_rate": 0.004, "warmup_reinvent_steps": 1000, "beta_init": 2.0, "beta_max": 40.0, "r_min": 0.01, "flow_target_tau": 0.005, "priority_alpha": 0.7},
        {"objective_mix_lambda": 0.4, "reinvent_loss_coef": 0.25, "flow_loss_coef": 4.0, "sigma": 100, "beta_rate": 0.004, "warmup_reinvent_steps": 1000, "beta_init": 2.0, "beta_max": 40.0, "r_min": 0.01, "flow_target_tau": 0.005, "priority_alpha": 0.7},
        {"objective_mix_lambda": 0.2, "reinvent_loss_coef": 0.50, "flow_loss_coef": 4.0, "sigma": 200, "beta_rate": 0.004, "warmup_reinvent_steps": 1000, "beta_init": 2.0, "beta_max": 40.0, "r_min": 0.01, "flow_target_tau": 0.005, "priority_alpha": 0.7},
        {"objective_mix_lambda": 0.4, "reinvent_loss_coef": 0.50, "flow_loss_coef": 4.0, "sigma": 200, "beta_rate": 0.004, "warmup_reinvent_steps": 1000, "beta_init": 2.0, "beta_max": 40.0, "r_min": 0.01, "flow_target_tau": 0.005, "priority_alpha": 0.7},
        {"objective_mix_lambda": 0.2, "reinvent_loss_coef": 0.25, "flow_loss_coef": 8.0, "sigma": 100, "beta_rate": 0.002, "warmup_reinvent_steps": 500, "beta_init": 2.0, "beta_max": 40.0, "r_min": 0.01, "flow_target_tau": 0.005, "priority_alpha": 0.7},
        {"objective_mix_lambda": 0.4, "reinvent_loss_coef": 0.25, "flow_loss_coef": 8.0, "sigma": 100, "beta_rate": 0.002, "warmup_reinvent_steps": 500, "beta_init": 2.0, "beta_max": 40.0, "r_min": 0.01, "flow_target_tau": 0.005, "priority_alpha": 0.7},
        {"objective_mix_lambda": 0.2, "reinvent_loss_coef": 0.50, "flow_loss_coef": 8.0, "sigma": 200, "beta_rate": 0.002, "warmup_reinvent_steps": 500, "beta_init": 2.0, "beta_max": 40.0, "r_min": 0.01, "flow_target_tau": 0.005, "priority_alpha": 0.7},
        {"objective_mix_lambda": 0.4, "reinvent_loss_coef": 0.50, "flow_loss_coef": 8.0, "sigma": 200, "beta_rate": 0.002, "warmup_reinvent_steps": 500, "beta_init": 2.0, "beta_max": 40.0, "r_min": 0.01, "flow_target_tau": 0.005, "priority_alpha": 0.7},
        {"objective_mix_lambda": 0.6, "reinvent_loss_coef": 0.50, "flow_loss_coef": 2.0, "sigma": 300, "beta_rate": 0.008, "warmup_reinvent_steps": 1000, "beta_init": 1.0, "beta_max": 25.0, "r_min": 0.001, "flow_target_tau": 0.010, "priority_alpha": 1.0},
        {"objective_mix_lambda": 0.6, "reinvent_loss_coef": 1.00, "flow_loss_coef": 2.0, "sigma": 300, "beta_rate": 0.004, "warmup_reinvent_steps": 1000, "beta_init": 1.0, "beta_max": 25.0, "r_min": 0.001, "flow_target_tau": 0.010, "priority_alpha": 1.0},
    ]

    for i, override in enumerate(knobs):
        cfg = deepcopy(base)
        cfg.update(override)
        cfg["name"] = f"p1_{i:02d}"
        candidates.append(cfg)
    return candidates


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase1_calls", type=int, default=2000)
    parser.add_argument("--phase2_calls", type=int, default=5000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--patience", type=int, default=999)
    parser.add_argument("--phase1_seed", type=int, default=0)
    parser.add_argument("--phase2_seeds", type=int, nargs="+", default=[0, 1])
    parser.add_argument("--top_k", type=int, default=3)
    parser.add_argument("--out_dir", type=str, default=None)
    args = parser.parse_args()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = args.out_dir or os.path.join(path_here, "sweeps", f"qed_auc10_{ts}")
    os.makedirs(out_dir, exist_ok=True)

    base = default_base_config()
    candidates = build_phase1_candidates(base)
    phase1_results = []

    print(f"[sweep] Phase 1: {len(candidates)} configs @ {args.phase1_calls} calls, seed={args.phase1_seed}")
    for cfg in candidates:
        trial_dir = os.path.join(out_dir, cfg["name"])
        os.makedirs(trial_dir, exist_ok=True)
        result = run_single_config(
            config=cfg,
            seed=args.phase1_seed,
            output_dir=trial_dir,
            max_oracle_calls=args.phase1_calls,
            freq_log=args.freq_log,
            patience=args.patience,
        )
        phase1_results.append({"config": cfg, "phase1": result})
        print(
            f"[sweep] {cfg['name']} auc_top10={result['auc_top10']:.6f} "
            f"avg_top10={result['avg_top10']:.6f} n_oracle={result['n_oracle']}"
        )

    phase1_sorted = sorted(phase1_results, key=lambda x: x["phase1"]["auc_top10"], reverse=True)
    selected = phase1_sorted[: args.top_k]
    print(f"[sweep] Phase 2: top-{len(selected)} configs @ {args.phase2_calls} calls, seeds={args.phase2_seeds}")

    phase2_rows = []
    for row in selected:
        cfg = row["config"]
        seed_rows = []
        for seed in args.phase2_seeds:
            trial_dir = os.path.join(out_dir, f"{cfg['name']}_seed{seed}")
            os.makedirs(trial_dir, exist_ok=True)
            seed_result = run_single_config(
                config=cfg,
                seed=seed,
                output_dir=trial_dir,
                max_oracle_calls=args.phase2_calls,
                freq_log=args.freq_log,
                patience=args.patience,
            )
            seed_rows.append(seed_result)
            print(
                f"[sweep] {cfg['name']} seed={seed} auc_top10={seed_result['auc_top10']:.6f} "
                f"avg_top10={seed_result['avg_top10']:.6f}"
            )

        aucs = [r["auc_top10"] for r in seed_rows]
        avgs = [r["avg_top10"] for r in seed_rows]
        aggregate = {
            "auc_top10_mean": float(np.mean(aucs)),
            "auc_top10_std": float(np.std(aucs)),
            "avg_top10_mean": float(np.mean(avgs)),
            "avg_top10_std": float(np.std(avgs)),
        }
        phase2_rows.append({
            "config": cfg,
            "phase1": row["phase1"],
            "phase2": aggregate,
            "phase2_per_seed": seed_rows,
        })

    phase2_sorted = sorted(phase2_rows, key=lambda x: x["phase2"]["auc_top10_mean"], reverse=True)
    best = phase2_sorted[0]

    best_config_path = os.path.join(out_dir, "best_config.yaml")
    with open(best_config_path, "w") as f:
        yaml.safe_dump(best["config"], f, sort_keys=True)

    summary = {
        "search_space_count": len(candidates),
        "phase1_calls": args.phase1_calls,
        "phase2_calls": args.phase2_calls,
        "phase2_seeds": args.phase2_seeds,
        "best": {
            "name": best["config"]["name"],
            "phase1_auc_top10": best["phase1"]["auc_top10"],
            "phase2_auc_top10_mean": best["phase2"]["auc_top10_mean"],
            "phase2_auc_top10_std": best["phase2"]["auc_top10_std"],
            "phase2_avg_top10_mean": best["phase2"]["avg_top10_mean"],
            "best_config_path": best_config_path,
        },
        "ranked_results": phase2_sorted,
    }
    summary_path = os.path.join(out_dir, "summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("[sweep] Best config")
    print(json.dumps(summary["best"], indent=2))
    print(f"[sweep] Summary: {summary_path}")


if __name__ == "__main__":
    main()
