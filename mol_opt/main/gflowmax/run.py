from __future__ import print_function

import argparse
import os
import sys
import yaml
from rdkit import RDLogger

path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append(".")

# Silence noisy RDKit deprecation warnings emitted by downstream fingerprint calls.
RDLogger.DisableLog("rdApp.warning")

# TDC expects `rdkit.six` on older RDKit versions. Newer RDKit drops it.
# Provide a compatibility alias so benchmark entrypoints remain runnable.
try:
    import rdkit.six  # noqa: F401
except Exception:
    import six as _six
    sys.modules["rdkit.six"] = _six

from tdc import Oracle
from main.optimizer import BaseOptimizer
from agents.agent import GFlowMaxAgent


class GFlowMax_Optimizer(BaseOptimizer):
    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gflowmax"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)

        run_config = dict(config)
        run_config["seed"] = getattr(self, "seed", 0)

        agent = GFlowMaxAgent(
            oracle=self.oracle,
            args=run_config,
            n_max_oracle_call=self.args.max_oracle_calls,
        )

        log_frequency = run_config.get("log_frequency", 100)

        for episode in range(agent.num_episodes):
            agent.train_episode()

            if (episode + 1) % log_frequency == 0:
                print(
                    "Episode {}/{} | oracle calls {} | epsilon {:.4f} | beta {:.3f} | best {:.4f}".format(
                        episode + 1,
                        agent.num_episodes,
                        len(self.oracle),
                        agent.epsilon,
                        agent.beta,
                        agent.best_reward,
                    )
                )

            if self.finish:
                print("max oracle hit... abort!")
                break


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--smi_file", default=None)
    parser.add_argument("--config_default", default="hparams_default.yaml")
    parser.add_argument("--config_tune", default="hparams_tune.yaml")
    parser.add_argument("--n_jobs", type=int, default=-1)
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--max_oracle_calls", type=int, default=10000)
    parser.add_argument("--freq_log", type=int, default=100)
    parser.add_argument("--n_runs", type=int, default=5)
    parser.add_argument("--seed", type=int, nargs="+", default=[0])
    parser.add_argument("--task", type=str, default="simple", choices=["tune", "simple", "production"])
    parser.add_argument("--oracles", nargs="+", default=["QED"])
    parser.add_argument("--log_results", action="store_true")
    args = parser.parse_args()

    if args.output_dir is None:
        args.output_dir = os.path.join(path_here, "results")
    if not os.path.exists(args.output_dir):
        os.mkdir(args.output_dir)

    try:
        config_default = yaml.safe_load(open(args.config_default))
    except Exception:
        config_default = yaml.safe_load(open(os.path.join(path_here, args.config_default)))

    if args.task == "tune":
        try:
            config_tune = yaml.safe_load(open(args.config_tune))
        except Exception:
            config_tune = yaml.safe_load(open(os.path.join(path_here, args.config_tune)))

        oracles = [Oracle(name=oracle_name) for oracle_name in args.oracles]
        optimizer = GFlowMax_Optimizer(args=args)
        optimizer.hparam_tune(oracles=oracles, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        return

    for oracle_name in args.oracles:
        oracle = Oracle(name=oracle_name)
        optimizer = GFlowMax_Optimizer(args=args)
        if args.task == "simple":
            for seed in args.seed:
                optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
        else:
            raise ValueError("Unrecognized task name.")


if __name__ == "__main__":
    main()
