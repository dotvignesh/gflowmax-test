from __future__ import print_function

import argparse
import os
import sys
import warnings
from collections import deque
from copy import deepcopy

import numpy as np
import torch
import yaml

warnings.filterwarnings("ignore")
# Suppress RDKit deprecation warnings emitted via rdApp logger (e.g. MorganGenerator).
try:
    from rdkit import rdBase
    rdBase.DisableLog("rdApp.warning")
except Exception:
    pass

path_here = os.path.dirname(os.path.realpath(__file__))
mol_opt_root = os.path.abspath(os.path.join(path_here, "..", ".."))
if mol_opt_root not in sys.path:
    sys.path.append(mol_opt_root)
gflownet_path = os.path.join(os.path.dirname(path_here), "gflownet")
if gflownet_path not in sys.path:
    sys.path.append(gflownet_path)

# TDC expects `rdkit.six` on older RDKit versions. Newer RDKit drops it.
# Provide a compatibility alias so benchmark entrypoints remain runnable.
try:
    import rdkit.six  # noqa: F401
except Exception:
    import six as _six
    sys.modules["rdkit.six"] = _six

from main.gflownet import run as gflownet_run
from main.optimizer import BaseOptimizer, Objdict
from tdc import Oracle

Dataset = gflownet_run.Dataset
make_model = gflownet_run.make_model


class PrioritizedReplayBuffer:
    def __init__(self, capacity, priority_eps):
        self.capacity = int(capacity)
        self.priority_eps = float(priority_eps)
        self.transitions = deque(maxlen=self.capacity)
        self.priorities = deque(maxlen=self.capacity)

    def __len__(self):
        return len(self.transitions)

    def add_many(self, transitions):
        if not transitions:
            return
        default_priority = max(self.priorities) if self.priorities else 1.0
        for transition in transitions:
            self.transitions.append(transition)
            self.priorities.append(default_priority)

    def sample(self, n, rng):
        n = min(int(n), len(self.transitions))
        if n <= 0:
            return [], np.array([], dtype=np.int64)

        prios = np.asarray(self.priorities, dtype=np.float64)
        if np.any(~np.isfinite(prios)) or prios.sum() <= 0:
            probs = np.full((len(prios),), 1.0 / len(prios), dtype=np.float64)
        else:
            probs = prios / prios.sum()

        indices = rng.choice(len(self.transitions), size=n, replace=False, p=probs)
        batch = [self.transitions[int(i)] for i in indices]
        return batch, indices

    def update_priorities(self, indices, deltas, alpha):
        if len(indices) == 0:
            return
        priority_updates = np.power(
            np.abs(np.asarray(deltas, dtype=np.float64)) + self.priority_eps,
            float(alpha),
        )
        for idx, value in zip(indices, priority_updates):
            self.priorities[int(idx)] = float(max(value, self.priority_eps))


class GFlowMaxMDP_Optimizer(BaseOptimizer):
    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "gflowmax-mdp"

    def _optimize(self, oracle, config):
        self.oracle.assign_evaluator(oracle)
        config = Objdict(config)
        get = lambda key, default: getattr(config, key) if hasattr(config, key) else default

        # Legacy-compatible defaults plus DBReplay-specific controls.
        config.mbsize = get("mbsize", 4)
        config.n_forward_samples = get("n_forward_samples", config.mbsize)
        config.batch_size = get("batch_size", 128)
        config.buffer_size = get("buffer_size", 10000)
        config.priority_alpha = get("priority_alpha", 1.0)
        config.priority_eps = get("priority_eps", 1e-6)
        config.beta_init = get("beta_init", 1.0)
        config.beta_rate = get("beta_rate", 0.008)
        config.beta_max = get("beta_max", 25.0)
        config.bootstrap_tau = get("bootstrap_tau", 0.005)
        config.log_every = get("log_every", 200)
        config.clip_grad = get("clip_grad", 0.0)
        config.clip_loss = get("clip_loss", 0.0)
        config.balanced_loss = get("balanced_loss", True)
        config.leaf_coef = get("leaf_coef", 10.0)
        config.R_min = get("R_min", 1e-4)
        config.log_reg_c = get("log_reg_c", 2e-8)

        floatX = torch.float if config.floatX == "float32" else torch.double
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bpath = os.path.join(gflownet_run.path_here, "data/blocks_PDB_105.json")

        dataset = Dataset(config, bpath, device, floatX=floatX)
        model = make_model(config, dataset.mdp).to(floatX).to(device)

        target_model = None
        if config.bootstrap_tau > 0:
            target_model = deepcopy(model)

        dataset.set_sampling_model(model, self.oracle, sample_prob=config.sample_prob)
        replay = PrioritizedReplayBuffer(config.buffer_size, config.priority_eps)
        optimizer = torch.optim.Adam(
            model.parameters(),
            config.learning_rate,
            weight_decay=config.weight_decay,
            betas=(config.opt_beta, config.opt_beta2),
            eps=config.opt_epsilon,
        )

        beta = float(config.beta_init)
        steps = 0
        patience = 0

        while True:
            if self.oracle.finish:
                print("max oracle hit, abort ...... ")
                break

            steps += 1
            previous_top_scores = None
            if len(self.oracle) > 100:
                self.oracle.sort_buffer()
                previous_top_scores = [
                    item[1][0] for item in list(self.oracle.mol_buffer.items())[:100]
                ]

            sample_columns = list(dataset.sample(config.n_forward_samples))
            if len(sample_columns) < 5:
                beta = min(beta + config.beta_rate, config.beta_max)
                continue

            transitions = list(zip(*sample_columns[:5]))
            replay.add_many(transitions)

            if len(replay) < config.batch_size:
                beta = min(beta + config.beta_rate, config.beta_max)
                continue

            replay_batch, replay_indices = replay.sample(config.batch_size, np.random)
            if not replay_batch:
                beta = min(beta + config.beta_rate, config.beta_max)
                continue

            p_raw, a_raw, r_raw, s_raw, d_raw = zip(*replay_batch)
            p, pb, a, r, s, d, _ = dataset.sample2batch((p_raw, a_raw, r_raw, s_raw, d_raw))
            ntransitions = r.shape[0]

            stem_out_p, mol_out_p = model(p, None)
            qsa_p = model.index_output_by_action(p, stem_out_p, mol_out_p[:, 0], a)
            exp_inflow = torch.zeros((ntransitions,), device=device, dtype=dataset.floatX)
            exp_inflow.index_add_(0, pb, torch.exp(qsa_p))
            inflow = torch.log(exp_inflow + config.log_reg_c)

            with torch.no_grad():
                bootstrap_model = target_model if target_model is not None else model
                stem_out_s, mol_out_s = bootstrap_model(s, None)
                exp_outflow = model.sum_output(
                    s, torch.exp(stem_out_s), torch.exp(mol_out_s[:, 0])
                )
                log_outflow = torch.log(exp_outflow + config.log_reg_c)
                log_terminal_reward = beta * torch.log(torch.clamp(r, min=config.R_min))
                target_flow = log_outflow * (1.0 - d) + log_terminal_reward * d

            delta = inflow - target_flow
            losses = delta.pow(2)
            if config.clip_loss > 0:
                losses = torch.clamp(losses, max=config.clip_loss)

            term_loss = (losses * d).sum() / (d.sum() + 1e-20)
            flow_loss = (losses * (1.0 - d)).sum() / ((1.0 - d).sum() + 1e-20)
            if config.balanced_loss:
                loss = term_loss * config.leaf_coef + flow_loss
            else:
                loss = losses.mean()

            optimizer.zero_grad()
            loss.backward()
            if config.clip_grad > 0:
                torch.nn.utils.clip_grad_value_(model.parameters(), config.clip_grad)
            optimizer.step()
            model.training_steps = steps + 1

            replay.update_priorities(
                replay_indices,
                delta.detach().cpu().numpy(),
                config.priority_alpha,
            )

            if target_model is not None:
                for source, target in zip(model.parameters(), target_model.parameters()):
                    target.data.mul_(1.0 - config.bootstrap_tau).add_(
                        config.bootstrap_tau * source.data
                    )

            beta = min(beta + config.beta_rate, config.beta_max)

            if steps % config.log_every == 0:
                print(
                    f"[gflowmax-mdp] step={steps} "
                    f"oracle={len(self.oracle)} "
                    f"beta={beta:.3f} "
                    f"loss={loss.item():.6f} "
                    f"term={term_loss.item():.6f} "
                    f"flow={flow_loss.item():.6f}"
                )

            if len(self.oracle) > 100:
                self.sort_buffer()
                new_top_scores = [item[1][0] for item in list(self.oracle.mol_buffer.items())[:100]]
                if previous_top_scores == new_top_scores:
                    patience += 1
                    if patience >= self.args.patience * 100:
                        self.log_intermediate(finish=True)
                        print("convergence criteria met, abort ...... ")
                        break
                else:
                    patience = 0

        print("Done.")


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
        optimizer = GFlowMaxMDP_Optimizer(args=args)
        optimizer.hparam_tune(oracles=oracles, hparam_space=config_tune, hparam_default=config_default, count=args.n_runs)
        return

    for oracle_name in args.oracles:
        oracle = Oracle(name=oracle_name)
        optimizer = GFlowMaxMDP_Optimizer(args=args)
        if args.task == "simple":
            for seed in args.seed:
                optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
        else:
            raise ValueError("Unrecognized task name.")


if __name__ == "__main__":
    main()
