from __future__ import print_function

import argparse
import os
import sys
from copy import deepcopy
import numpy as np
import yaml
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))

# Silence noisy RDKit deprecation warnings in benchmark logs.
try:
    from rdkit import RDLogger
    RDLogger.DisableLog("rdApp.warning")
except Exception:
    pass

# TDC expects `rdkit.six` on older RDKit versions. Newer RDKit drops it.
# Provide a compatibility alias so benchmark entrypoints remain runnable.
try:
    import rdkit.six  # noqa: F401
except Exception:
    import six as _six
    sys.modules["rdkit.six"] = _six

from main.optimizer import BaseOptimizer
from utils import Variable, unique, seq_to_selfies
from model import RNN
from data_structs import Vocabulary, TransitionReplay
import torch
from tdc import Oracle

from tdc.chem_utils import MolConvert
selfies2smiles = MolConvert(src = 'SELFIES', dst = 'SMILES')


class REINVENT_GFM_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "reinvent_gfm"

    @staticmethod
    def _token_ids_until_terminal(seq_row, eos_token):
        token_ids = []
        for token in seq_row:
            token = int(token)
            token_ids.append(token)
            if token == eos_token:
                break
        return token_ids

    def _add_flow_trajectories(self, replay, seqs, scores, prior_likelihood, eos_token, step):
        seqs_np = seqs.detach().cpu().numpy()
        prior_np = prior_likelihood.detach().cpu().numpy()
        for i, seq_row in enumerate(seqs_np):
            token_ids = self._token_ids_until_terminal(seq_row, eos_token)
            if len(token_ids) == 0:
                continue
            replay.add_trajectory(
                token_ids=token_ids,
                terminal_reward=float(scores[i]),
                prior_ll_seq=float(prior_np[i]),
                seq_id=f"{step}:{i}",
            )

    def _compute_flow_loss(self, Agent, target_rnn, replay_batch, beta, r_min):
        if len(replay_batch) == 0:
            device = next(Agent.rnn.parameters()).device
            zero = torch.zeros((), device=device)
            return zero, zero

        prefixes = [item["state_prefix"] for item in replay_batch]
        actions = [item["action_token"] for item in replay_batch]
        next_prefixes = [item["next_prefix"] for item in replay_batch]
        done = [1.0 if item["done"] else 0.0 for item in replay_batch]
        rewards = [item["terminal_reward"] for item in replay_batch]

        logpf, logf = Agent.transition_logpf_logf(prefixes, actions)
        device = logpf.device
        done_t = torch.tensor(done, dtype=torch.float32, device=device)
        rewards_t = torch.tensor(rewards, dtype=torch.float32, device=device)

        with torch.no_grad():
            logf_next = Agent.prefix_logf(next_prefixes, net=target_rnn)
            terminal_target = beta * torch.log(torch.clamp(rewards_t, min=r_min))
            target = (1.0 - done_t) * logf_next + done_t * terminal_target

        delta = logf + logpf - target
        flow_loss = torch.mean(torch.square(delta))
        return flow_loss, delta

    @staticmethod
    def _safe_selfies_to_smiles(selfies_list):
        smiles_list = []
        for selfies in selfies_list:
            if not selfies:
                smiles_list.append(None)
                continue
            try:
                smiles = selfies2smiles(selfies)
            except Exception:
                smiles = None
            smiles_list.append(smiles if smiles else None)
        return smiles_list

    def _optimize(self, oracle, config):

        self.oracle.assign_evaluator(oracle)

        path_here = os.path.dirname(os.path.realpath(__file__))
        restore_prior_from=os.path.join(path_here, 'data/Prior.ckpt')
        restore_agent_from=restore_prior_from 
        voc = Vocabulary(init_from_file=os.path.join(path_here, "data/Voc"))

        Prior = RNN(voc)
        Agent = RNN(voc)

        # By default restore Agent to same model as Prior, but can restore from already trained Agent too.
        # Saved models are partially on the GPU, but if we dont have cuda enabled we can remap these
        # to the CPU.
        if torch.cuda.is_available():
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here,'data/Prior.ckpt')), strict=False)
            Agent.rnn.load_state_dict(torch.load(restore_agent_from), strict=False)
        else:
            Prior.rnn.load_state_dict(
                torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage),
                strict=False,
            )
            Agent.rnn.load_state_dict(
                torch.load(restore_agent_from, map_location=lambda storage, loc: storage),
                strict=False,
            )

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])
        target_rnn = deepcopy(Agent.rnn)
        for param in target_rnn.parameters():
            param.requires_grad = False

        objective_mix_lambda = float(config.get('objective_mix_lambda', 0.6))
        objective_mode = str(config.get('objective_mode', 'hybrid')).lower()
        valid_objective_modes = {"hybrid", "reinvent_only", "flow_only"}
        if objective_mode not in valid_objective_modes:
            raise ValueError(
                f"Invalid objective_mode={objective_mode}. "
                f"Expected one of {sorted(valid_objective_modes)}."
            )
        reinvent_loss_coef = float(config.get('reinvent_loss_coef', 1.0))
        flow_loss_coef = float(config.get('flow_loss_coef', 1.0))
        beta_init = float(config.get('beta_init', 1.0))
        beta_rate = float(config.get('beta_rate', 0.008))
        beta_max = float(config.get('beta_max', 25.0))
        r_min = float(config.get('r_min', 1e-4))
        priority_alpha = float(config.get('priority_alpha', 1.0))
        priority_eps = float(config.get('priority_eps', 1e-6))
        flow_target_tau = float(config.get('flow_target_tau', 0.01))
        flow_replay_size = int(config.get('flow_replay_size', 20000))
        flow_batch_size = int(config.get('flow_batch_size', config.get('batch_size', 64)))
        max_seq_len = int(config.get('max_seq_len', 140))
        terminal_only_oracle = bool(config.get('terminal_only_oracle', True))
        warmup_reinvent_steps = int(config.get('warmup_reinvent_steps', 0))
        log_every = int(config.get('log_every', self.args.freq_log if self.args is not None else 100))

        flow_replay = TransitionReplay(
            voc=voc,
            max_size=flow_replay_size,
            priority_alpha=priority_alpha,
            priority_eps=priority_eps,
        )
        eos_token = voc.vocab['[EOS]']

        print("Model initialized, starting training...")

        step = 0
        patience = 0

        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0

            beta = min(beta_init + beta_rate * step, beta_max)
            
            # Sample from Agent
            seqs, agent_likelihood, entropy = Agent.sample_with_trace(
                config['batch_size'],
                max_length=max_seq_len,
            )

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            selfies_list = seq_to_selfies(seqs, voc) 
            smiles_list = self._safe_selfies_to_smiles(selfies_list)
            if not terminal_only_oracle:
                raise ValueError("Option B currently supports terminal_only_oracle=True.")
            score = np.array(self.oracle(smiles_list))

            if self.finish:
                print('max oracle hit')
                break 

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience*2:
                        # Log only observed calls; avoid extrapolating to max budget on early-stop.
                        self.log_intermediate(finish=False, normalize_auc_to_observed=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # Add sampled trajectories into transition replay.
            self._add_flow_trajectories(
                replay=flow_replay,
                seqs=seqs,
                scores=score,
                prior_likelihood=prior_likelihood,
                eos_token=eos_token,
                step=step,
            )

            # REINVENT anchor term (on-policy).
            augmented_likelihood = prior_likelihood.float() + config['sigma'] * Variable(score).float()
            reinvent_loss = torch.mean(torch.pow((augmented_likelihood - agent_likelihood), 2))

            # Flow consistency term (transition replay + PER).
            replay_batch, replay_indices = flow_replay.sample_transitions(flow_batch_size)
            flow_loss, delta = self._compute_flow_loss(
                Agent=Agent,
                target_rnn=target_rnn,
                replay_batch=replay_batch,
                beta=beta,
                r_min=r_min,
            )
            if len(replay_batch) > 0:
                flow_replay.update_priorities(
                    replay_indices,
                    delta.detach().cpu().numpy(),
                    alpha=priority_alpha,
                    eps=priority_eps,
                )

            flow_ready = (len(replay_batch) > 0)
            if objective_mode == "reinvent_only":
                objective_label = "reinvent_only"
                objective_loss = reinvent_loss_coef * reinvent_loss
            elif objective_mode == "flow_only":
                if flow_ready:
                    objective_label = "flow_only"
                    objective_loss = flow_loss_coef * flow_loss
                else:
                    objective_label = "flow_bootstrap_reinvent"
                    objective_loss = reinvent_loss_coef * reinvent_loss
            else:
                use_flow = (step >= warmup_reinvent_steps) and flow_ready
                if use_flow:
                    objective_label = "hybrid_transition_flow"
                    objective_loss = (
                        reinvent_loss_coef * objective_mix_lambda * reinvent_loss
                        + flow_loss_coef * (1 - objective_mix_lambda) * flow_loss
                    )
                else:
                    objective_label = "reinvent_warmup"
                    objective_loss = reinvent_loss_coef * reinvent_loss

            # Add regularizer that penalizes high likelihood for the entire sequence
            loss_p = - (1 / agent_likelihood).mean()
            reg_loss = 5 * 1e3 * loss_p
            total_loss = objective_loss + reg_loss

            # Calculate gradients and make an update to the network weights
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            # Soft update target flow network.
            if flow_target_tau > 0:
                for source, target in zip(Agent.rnn.parameters(), target_rnn.parameters()):
                    target.data.mul_(1.0 - flow_target_tau).add_(flow_target_tau * source.data)

            if log_every > 0 and (step + 1) % log_every == 0:
                mean_abs_delta = float(torch.mean(torch.abs(delta)).detach().cpu().item()) if len(replay_batch) > 0 else 0.0
                print(
                    f"[reinvent-gfm] step={step + 1} "
                    f"oracle={len(self.oracle)} "
                    f"mode={objective_label} "
                    f"objective_mode={objective_mode} "
                    f"reinvent_coef={reinvent_loss_coef:.3f} "
                    f"flow_coef={flow_loss_coef:.3f} "
                    f"beta={beta:.4f} "
                    f"reinvent_loss={float(reinvent_loss.detach().cpu().item()):.6f} "
                    f"flow_loss={float(flow_loss.detach().cpu().item()):.6f} "
                    f"objective_loss={float(objective_loss.detach().cpu().item()):.6f} "
                    f"reg_loss={float(reg_loss.detach().cpu().item()):.6f} "
                    f"total_loss={float(total_loss.detach().cpu().item()):.6f} "
                    f"reward_mean={float(np.mean(score)):.4f} "
                    f"reward_max={float(np.max(score)):.4f} "
                    f"flow_replay={len(flow_replay)} "
                    f"replay_ess={flow_replay.effective_sample_size():.2f} "
                    f"mean_abs_delta={mean_abs_delta:.6f}"
                )

            step += 1


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
        optimizer = REINVENT_GFM_Optimizer(args=args)
        optimizer.hparam_tune(
            oracles=oracles,
            hparam_space=config_tune,
            hparam_default=config_default,
            count=args.n_runs,
        )
        return

    for oracle_name in args.oracles:
        oracle = Oracle(name=oracle_name)
        optimizer = REINVENT_GFM_Optimizer(args=args)
        if args.task == "simple":
            for seed in args.seed:
                optimizer.optimize(oracle=oracle, config=config_default, seed=seed)
        elif args.task == "production":
            optimizer.production(oracle=oracle, config=config_default, num_runs=args.n_runs)
        else:
            raise ValueError("Unrecognized task name.")


if __name__ == "__main__":
    main()
