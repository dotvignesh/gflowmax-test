import os
import sys
import numpy as np
import math
path_here = os.path.dirname(os.path.realpath(__file__))
sys.path.append(path_here)
sys.path.append('/'.join(path_here.rstrip('/').split('/')[:-2]))
from main.optimizer import BaseOptimizer
if __package__:
    from .utils import Variable, seq_to_smiles, unique
    from .model import RNN
    from .data_structs import Vocabulary, Experience, MolData
    from .priority_queue import MaxRewardPriorityQueue
else:
    from utils import Variable, seq_to_smiles, unique
    from model import RNN
    from data_structs import Vocabulary, Experience, MolData
    from priority_queue import MaxRewardPriorityQueue
import torch
from rdkit import Chem
from tdc import Evaluator
try:
    from polyleven import levenshtein
except ImportError:
    def levenshtein(a, b):
        if a == b:
            return 0
        if len(a) < len(b):
            a, b = b, a
        previous_row = list(range(len(b) + 1))
        for i, ca in enumerate(a, 1):
            current_row = [i]
            for j, cb in enumerate(b, 1):
                insert_cost = current_row[j - 1] + 1
                delete_cost = previous_row[j] + 1
                replace_cost = previous_row[j - 1] + (ca != cb)
                current_row.append(min(insert_cost, delete_cost, replace_cost))
            previous_row = current_row
        return previous_row[-1]

import itertools
import pickle
import pandas as pd
try:
    import wandb
except ImportError:
    class _WandbStub:
        @staticmethod
        def log(*args, **kwargs):
            return None

    wandb = _WandbStub()

from time import perf_counter

from joblib import Parallel


def diversity(smiles):
    # dist = [levenshtein(*pair) for pair in itertools.combinations(smiles, 2)]
    dist, normalized = [], []
    for pair in itertools.combinations(smiles, 2):
        dist.append(levenshtein(*pair))
        normalized.append(levenshtein(*pair)/max(len(pair[0]), len(pair[1])))
    evaluator = Evaluator(name = 'Diversity')
    mol_div = evaluator(smiles)
    return np.mean(normalized), np.mean(dist), mol_div


def novelty(new_smiles, ref_smiles):
    smiles_novelty = [min([levenshtein(d, od) for od in ref_smiles]) for d in new_smiles]
    smiles_norm_novelty = [min([levenshtein(d, od) / max(len(d), len(od)) for od in ref_smiles]) for d in new_smiles]
    evaluator = Evaluator(name = 'Novelty')
    mol_novelty = evaluator(new_smiles, ref_smiles)
    return np.mean(smiles_norm_novelty), np.mean(smiles_novelty), mol_novelty


def sanitize(smiles):
    canonicalized = []
    for s in smiles:
        try:
            canonicalized.append(Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True))
        except:
            pass
    return canonicalized


def _encode_smiles_batch(voc, smiles, scores):
    encoded, valid_scores, valid_smiles = [], [], []
    for smi, score in zip(smiles, scores):
        try:
            tokenized = voc.tokenize(smi)
            encoded.append(Variable(voc.encode(tokenized)))
            valid_scores.append(float(score))
            valid_smiles.append(smi)
        except Exception:
            pass
    if not encoded:
        return None, np.array([], dtype=np.float32), []
    return MolData.collate_fn(encoded), np.asarray(valid_scores, dtype=np.float32), valid_smiles


def _dedupe_candidates(candidates):
    best_by_smiles = {}
    for cand in candidates:
        smi = cand["smiles"]
        if smi not in best_by_smiles or cand["score"] > best_by_smiles[smi]["score"]:
            best_by_smiles[smi] = cand
    return list(best_by_smiles.values())


def _sample_mixed_candidates(candidates, batch_size, ga_mix_ratio):
    if batch_size <= 0 or not candidates:
        return []

    policy_candidates = [c for c in candidates if c["source"] == "policy"]
    ga_candidates = [c for c in candidates if c["source"] == "ga"]
    if not policy_candidates:
        ga_mix_ratio = 1.0
    elif not ga_candidates:
        ga_mix_ratio = 0.0

    batch_size = min(batch_size, len(candidates))
    desired_ga = int(round(batch_size * ga_mix_ratio))
    desired_ga = max(0, min(desired_ga, batch_size))

    if ga_mix_ratio > 0 and desired_ga == 0 and ga_candidates and batch_size > 0:
        desired_ga = 1
    desired_policy = batch_size - desired_ga
    if ga_mix_ratio < 1 and desired_policy == 0 and policy_candidates and batch_size > 0:
        desired_policy = 1
        desired_ga = batch_size - 1

    n_policy = min(desired_policy, len(policy_candidates))
    n_ga = min(desired_ga, len(ga_candidates))

    selected = []
    if n_policy > 0:
        idxs = np.random.choice(len(policy_candidates), size=n_policy, replace=False)
        selected.extend([policy_candidates[i] for i in idxs])
    if n_ga > 0:
        idxs = np.random.choice(len(ga_candidates), size=n_ga, replace=False)
        selected.extend([ga_candidates[i] for i in idxs])

    if len(selected) < batch_size:
        selected_smiles = set(c["smiles"] for c in selected)
        remainder = [c for c in candidates if c["smiles"] not in selected_smiles]
        if remainder:
            n_rem = min(batch_size - len(selected), len(remainder))
            idxs = np.random.choice(len(remainder), size=n_rem, replace=False)
            selected.extend([remainder[i] for i in idxs])
    return selected


def _update_beta(beta, beta_rate, beta_max, log_z_value, best_reward, min_reward):
    best_reward_pos = max(float(best_reward), float(min_reward))
    target = beta * np.log(best_reward_pos)
    gap = abs(log_z_value - target)
    denom = max(abs(target), 1.0)
    sig = 1.0 / (1.0 + np.exp(-(-gap / denom + 1.0) * 3.0))
    return min(beta + beta_rate * sig, beta_max)


def _schedule_linear(step, start_value, end_value, ramp_steps):
    if ramp_steps <= 0:
        return float(end_value)
    progress = min(1.0, max(0.0, float(step) / float(ramp_steps)))
    return float(start_value + progress * (end_value - start_value))


def _compute_weighted_gflow_loss(
    agent_likelihood,
    prior_agent_likelihood,
    reward,
    log_z,
    beta,
    weight_temp,
    min_reward,
    penalty,
    kl_coefficient,
    gamma_z,
    anchor_reward_pos,
    reward_transform,
):
    reward_pos = torch.clamp(reward, min=min_reward)
    log_r = torch.log(reward_pos)
    weights = torch.softmax(weight_temp * log_r, dim=0)

    forward_flow = agent_likelihood + log_z
    if reward_transform == 'log':
        reward_target = log_r
    else:
        reward_target = reward_pos
    backward_flow = beta * reward_target
    if penalty == 'pb':
        backward_flow += prior_agent_likelihood

    delta = forward_flow - backward_flow
    loss_flow = torch.sum(weights * torch.pow(delta, 2))

    loss_reg = torch.tensor(0.0, device=reward.device)
    if penalty == 'prior_kl':
        loss_reg = torch.sum(weights * (agent_likelihood - prior_agent_likelihood))

    target_log_z = beta * math.log(anchor_reward_pos)
    loss_logz = gamma_z * torch.pow(log_z - target_log_z, 2).mean()
    loss = loss_flow + kl_coefficient * loss_reg + loss_logz
    return loss, loss_flow, loss_reg, loss_logz


class Genetic_GFN_Optimizer(BaseOptimizer):

    def __init__(self, args=None):
        super().__init__(args)
        self.model_name = "genetic_gfn"

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
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here,'data/Prior.ckpt')))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from))
        else:
            Prior.rnn.load_state_dict(torch.load(os.path.join(path_here, 'data/Prior.ckpt'), map_location=lambda storage, loc: storage))
            Agent.rnn.load_state_dict(torch.load(restore_agent_from, map_location=lambda storage, loc: storage))

        # We dont need gradients with respect to Prior
        for param in Prior.rnn.parameters():
            param.requires_grad = False

        # optimizer = torch.optim.Adam(Agent.rnn.parameters(), lr=config['learning_rate'])
        device = next(Agent.rnn.parameters()).device
        log_z = torch.nn.Parameter(torch.tensor([5.0], device=device))
        beta = float(config.get('beta_init', config.get('beta', 50.0)))
        beta_rate = float(config.get('beta_rate', 0.015))
        beta_max = float(config.get('beta_max', max(20.0, beta)))
        gamma_z = float(config.get('gamma_z', 0.05))
        weight_temp_end = float(config.get('weight_temp_end', config.get('weight_temp', 5.0)))
        weight_temp_start = float(config.get('weight_temp_start', min(2.0, weight_temp_end)))
        weight_temp_ramp_steps = int(config.get('weight_temp_ramp_steps', 300))
        ga_mix_ratio = float(config.get('ga_mix_ratio', 0.5))
        ga_mix_min = float(config.get('ga_mix_min', 0.2))
        ga_mix_max = float(config.get('ga_mix_max', 0.8))
        ga_mix_step = float(config.get('ga_mix_step', 0.05))
        ga_mix_patience = int(config.get('ga_mix_patience', 3))
        adaptive_ga_mix = bool(config.get('adaptive_ga_mix', True))
        ga_mix_min = min(max(ga_mix_min, 0.0), 1.0)
        ga_mix_max = min(max(ga_mix_max, ga_mix_min), 1.0)
        ga_mix_ratio = min(max(ga_mix_ratio, ga_mix_min), ga_mix_max)
        ga_mix_ratio_curr = ga_mix_ratio
        replay_blend_ratio = float(config.get('replay_blend_ratio', 0.3))
        replay_blend_ratio = min(max(replay_blend_ratio, 0.0), 1.0)
        replay_start_step = int(config.get('replay_start_step', 300))
        replay_ramp_steps = int(config.get('replay_ramp_steps', 1000))
        min_reward = float(config.get('min_reward', 1e-8))
        onpolicy_batch_N = int(config.get('onpolicy_batch_N', config.get('experience_replay', config['batch_size'])))
        if onpolicy_batch_N <= 0:
            onpolicy_batch_N = int(config['batch_size'])
        replay_batch_size = int(config.get('replay_batch_size', config.get('experience_replay', onpolicy_batch_N)))
        if replay_batch_size <= 0:
            replay_batch_size = onpolicy_batch_N
        onpolicy_updates_per_step = int(config.get('onpolicy_updates_per_step', config.get('experience_loop', 1)))
        onpolicy_updates_per_step = max(1, onpolicy_updates_per_step)
        kl_coefficient = float(config.get('kl_coefficient', 0.0))
        penalty_mode = config.get('penalty', '')
        reward_transform = str(config.get('reward_transform', 'linear')).lower()
        if reward_transform not in {'linear', 'log'}:
            reward_transform = 'linear'
        policy_ema_alpha = float(config.get('policy_ema_alpha', 0.05))
        policy_ema_alpha = min(max(policy_ema_alpha, 0.0), 1.0)
        optimizer = torch.optim.Adam([{'params': Agent.rnn.parameters(), 
                                        'lr': config['learning_rate']},
                                    {'params': log_z, 
                                        'lr': config['lr_z']}])

        # For policy based RL, we normally train on-policy and correct for the fact that more likely actions
        # occur more often (which means the agent can get biased towards them). Using experience replay is
        # therefor not as theoretically sound as it is for value based RL, but it seems to work well.
        experience = Experience(voc, max_size=config['num_keep'])
        
        if config['ga_method'].lower() == 'stoned':
            if __package__:
                from .ga_expert import STONED as GeneticOperatorHandler
            else:
                from ga_expert import STONED as GeneticOperatorHandler
        else:
            if __package__:
                from .ga_expert import GeneticOperatorHandler
            else:
                from ga_expert import GeneticOperatorHandler

        ga_handler = GeneticOperatorHandler(mutation_rate=config['mutation_rate'], 
                                            population_size=config['population_size'])
        pool = Parallel(n_jobs=config['num_jobs'])

        print("Model initialized, starting training...")

        step = 0
        patience = 0
        prev_n_oracles = 0
        stuck_cnt = 0

        policy_smiles_norm_diversity, policy_smiles_diversity, policy_mol_diversity = [], [], []
        policy_smiles_norm_novelty, policy_smiles_novelty, policy_mol_novelty = [], [], []
        ga_smiles_norm_novelty, ga_smiles_novelty, ga_mol_novelty = [], [], []
        policy_ga_smiles_norm_novelty, policy_ga_smiles_novelty, policy_ga_mol_novelty = [], [], []
        parents_children_smiles_distances, parents_children_mol_distances = [], []
        best_scores = []

        tot_ga_results = pd.DataFrame({'children': [], 'parents': [], 'smiles_dist': [], 'mol_dist': []})

        prev_best = 0.
        
        ga_times = []
        eval_times = []
        training_times = []
        total_start = perf_counter()
        best_reward_seen = min_reward
        best_policy_seen = None
        policy_best_ema = None
        policy_stall_steps = 0
        while True:

            if len(self.oracle) > 100:
                self.sort_buffer()
                old_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
            else:
                old_scores = 0
            
            # Sample from Agent
            seqs, agent_likelihood, entropy = Agent.sample(config['batch_size'])

            # Remove duplicates, ie only consider unique seqs
            unique_idxs = unique(seqs)
            seqs = seqs[unique_idxs]
            agent_likelihood = agent_likelihood[unique_idxs]
            entropy = entropy[unique_idxs]

            # Get prior likelihood and score
            # prior_likelihood, _ = Prior.likelihood(Variable(seqs))
            smiles = seq_to_smiles(seqs, voc)
            if config['valid_only']:
                smiles = sanitize(smiles)
            
            eval_start = perf_counter()
            score = np.array(self.oracle(smiles))
            eval_times.append(perf_counter() - eval_start)
            policy_candidates = [
                {"smiles": smi, "score": float(sc), "source": "policy"}
                for smi, sc in zip(smiles, score)
            ]

            if self.finish:
                print('max oracle hit')
                break 

            # early stopping
            if len(self.oracle) > 1000:
                self.sort_buffer()
                new_scores = [item[1][0] for item in list(self.mol_buffer.items())[:100]]
                if new_scores == old_scores:
                    patience += 1
                    if patience >= self.args.patience:
                        self.log_intermediate(finish=True)
                        print('convergence criteria met, abort ...... ')
                        break
                else:
                    patience = 0

            # early stopping
            if prev_n_oracles < len(self.oracle):
                stuck_cnt = 0
            else:
                stuck_cnt += 1
                if stuck_cnt >= 10:
                    self.log_intermediate(finish=True)
                    print('cannot find new molecules, abort ...... ')
                    break
            
            prev_n_oracles = len(self.oracle)

            # Calculate augmented likelihood (REINVENT)
            # augmented_likelihood = prior_likelihood.float() + 500 * Variable(score).float()
            # reinvent_loss = torch.pow((augmented_likelihood - agent_likelihood), 2)
            # print('REINVENT:', reinvent_loss.mean().item())

            # policy novelty
            if step > 0:
                # import pdb; pdb.set_trace()
                if smiles and len(experience) > 0:
                    smiles_norm_novelty, smiles_novelty, mol_novelty = novelty(smiles, experience.get_elems()[0])
                    policy_smiles_norm_novelty.append(smiles_norm_novelty)
                    policy_smiles_novelty.append(smiles_novelty)
                    policy_mol_novelty.append(mol_novelty)
                if len(smiles) > 1:
                    smiles_norm_div, smiles_div, mol_div = diversity(smiles)
                    policy_smiles_norm_diversity.append(smiles_norm_div)
                    policy_smiles_diversity.append(smiles_div)
                    policy_mol_diversity.append(mol_div)
            policy_best = float(np.max(score)) if score.size > 0 else 0.0
            current_weight_temp = _schedule_linear(
                step=step,
                start_value=weight_temp_start,
                end_value=weight_temp_end,
                ramp_steps=weight_temp_ramp_steps,
            )
            if step < replay_start_step:
                current_replay_blend = 0.0
            else:
                current_replay_blend = replay_blend_ratio * _schedule_linear(
                    step=step - replay_start_step,
                    start_value=0.0,
                    end_value=1.0,
                    ramp_steps=replay_ramp_steps,
                )

            if adaptive_ga_mix:
                if best_policy_seen is None:
                    best_policy_seen = policy_best
                elif policy_best > best_policy_seen + 1e-12:
                    best_policy_seen = policy_best
                    policy_stall_steps = 0
                    ga_mix_ratio_curr = max(ga_mix_min, ga_mix_ratio_curr - ga_mix_step)
                else:
                    policy_stall_steps += 1
                    if policy_stall_steps >= ga_mix_patience:
                        ga_mix_ratio_curr = min(ga_mix_max, ga_mix_ratio_curr + ga_mix_step)
                        policy_stall_steps = 0

            policy_best_pos = max(policy_best, min_reward)
            if policy_best_ema is None:
                policy_best_ema = policy_best_pos
            else:
                policy_best_ema = (1.0 - policy_ema_alpha) * policy_best_ema + policy_ema_alpha * policy_best_pos

            # Then add new experience
            new_experience = zip(smiles, score)
            experience.add_experience(new_experience)

            ga_best = 0.
            ga_candidates = []
            if config['population_size'] and len(self.oracle) > config['population_size']:
                self.oracle.sort_buffer()
                pop_smis, pop_scores = tuple(map(list, zip(*[(smi, elem[0]) for (smi, elem) in self.oracle.mol_buffer.items()])))

                mating_pool = (pop_smis[:config['num_keep']], pop_scores[:config['num_keep']])

                for g in range(config['ga_generations']):
                    ga_start = perf_counter()
                    child_smis, child_n_atoms, _, _, ga_results = ga_handler.query(
                            query_size=config['offspring_size'], mating_pool=mating_pool, pool=pool, 
                            rank_coefficient=config['rank_coefficient'], return_dist=True
                        )
                    ga_times.append(perf_counter() - ga_start)
                    
                    eval_start = perf_counter()
                    child_score = np.array(self.oracle(child_smis))
                    eval_times.append(perf_counter() - eval_start)
                    ga_candidates.extend(
                        [{"smiles": smi, "score": float(sc), "source": "ga"} for smi, sc in zip(child_smis, child_score)]
                    )
                    
                    if child_score.size > 0 and child_score.max() > ga_best:
                        ga_best = child_score.max()
                
                    new_experience = zip(child_smis, child_score)
                    experience.add_experience(new_experience)

                    mating_pool = (mating_pool[0]+child_smis, mating_pool[1]+child_score.tolist())

                    if self.finish:
                        print('max oracle hit')
                        break
                
            
            best_scores.append([policy_best, ga_best, prev_best])

            if max(ga_best, policy_best) > prev_best:
                prev_best = max(ga_best, policy_best)

            # Experience Replay
            # Deprecated replay knobs (`experience_replay`/`experience_loop`) are
            # only used as backward-compatible defaults for on-policy batch/update sizes.
            training_start = perf_counter()
            loss_total_val = 0.0
            loss_flow_val = 0.0
            loss_reg_val = 0.0
            loss_logz_val = 0.0
            mean_train_reward = 0.0
            reward_terms = 0
            n_train_batch = 0
            n_replay_batch = 0
            executed_updates = 0
            total_train_batch = 0
            total_replay_batch = 0
            replay_fraction_used = 0.0

            all_candidates = _dedupe_candidates(policy_candidates + ga_candidates)
            for cand in all_candidates:
                best_reward_seen = max(best_reward_seen, cand["score"])

            n_policy_candidates = sum(1 for c in all_candidates if c["source"] == "policy")
            n_ga_candidates = sum(1 for c in all_candidates if c["source"] == "ga")

            for _ in range(onpolicy_updates_per_step):
                chosen = _sample_mixed_candidates(all_candidates, onpolicy_batch_N, ga_mix_ratio_curr)
                n_train_batch = len(chosen)
                anchor_reward_pos = max(policy_best_ema if policy_best_ema is not None else best_reward_seen, min_reward)
                onpolicy_loss = None
                onpolicy_loss_flow = None
                onpolicy_loss_reg = None
                onpolicy_loss_logz = None
                replay_loss = None
                replay_loss_flow = None
                replay_loss_reg = None
                replay_loss_logz = None

                if n_train_batch > 0:
                    chosen_smiles = [c["smiles"] for c in chosen]
                    chosen_scores = [c["score"] for c in chosen]
                    train_seqs, train_scores_np, _ = _encode_smiles_batch(voc, chosen_smiles, chosen_scores)
                    if train_seqs is not None and len(train_scores_np) > 0:
                        n_train_batch = int(len(train_scores_np))
                        exp_agent_likelihood, _ = Agent.likelihood(train_seqs.long())
                        prior_agent_likelihood, _ = Prior.likelihood(train_seqs.long())
                        reward = torch.tensor(train_scores_np, device=device)
                        onpolicy_loss, onpolicy_loss_flow, onpolicy_loss_reg, onpolicy_loss_logz = _compute_weighted_gflow_loss(
                            agent_likelihood=exp_agent_likelihood,
                            prior_agent_likelihood=prior_agent_likelihood,
                            reward=reward,
                            log_z=log_z,
                            beta=beta,
                            weight_temp=current_weight_temp,
                            min_reward=min_reward,
                            penalty=penalty_mode,
                            kl_coefficient=kl_coefficient,
                            gamma_z=gamma_z,
                            anchor_reward_pos=anchor_reward_pos,
                            reward_transform=reward_transform,
                        )
                        mean_train_reward += reward.mean().item()
                        reward_terms += 1
                        total_train_batch += n_train_batch

                if current_replay_blend > 0 and len(experience) >= replay_batch_size:
                    try:
                        if config['rank_coefficient'] > 0:
                            replay_seqs, replay_scores = experience.rank_based_sample(
                                replay_batch_size, config['rank_coefficient']
                            )
                        else:
                            replay_seqs, replay_scores = experience.sample(replay_batch_size)
                    except Exception:
                        replay_seqs, replay_scores = None, None

                    if replay_seqs is not None and replay_scores is not None and len(replay_scores) > 0:
                        n_replay_batch = int(len(replay_scores))
                        exp_agent_likelihood, _ = Agent.likelihood(replay_seqs.long())
                        prior_agent_likelihood, _ = Prior.likelihood(replay_seqs.long())
                        replay_reward = torch.tensor(replay_scores, device=device)
                        replay_loss, replay_loss_flow, replay_loss_reg, replay_loss_logz = _compute_weighted_gflow_loss(
                            agent_likelihood=exp_agent_likelihood,
                            prior_agent_likelihood=prior_agent_likelihood,
                            reward=replay_reward,
                            log_z=log_z,
                            beta=beta,
                            weight_temp=current_weight_temp,
                            min_reward=min_reward,
                            penalty=penalty_mode,
                            kl_coefficient=kl_coefficient,
                            gamma_z=gamma_z,
                            anchor_reward_pos=anchor_reward_pos,
                            reward_transform=reward_transform,
                        )
                        mean_train_reward += replay_reward.mean().item()
                        reward_terms += 1
                        total_replay_batch += n_replay_batch

                if onpolicy_loss is not None and replay_loss is not None:
                    replay_weight = current_replay_blend
                    loss = (1.0 - replay_weight) * onpolicy_loss + replay_weight * replay_loss
                    loss_flow = (1.0 - replay_weight) * onpolicy_loss_flow + replay_weight * replay_loss_flow
                    loss_reg = (1.0 - replay_weight) * onpolicy_loss_reg + replay_weight * replay_loss_reg
                    loss_logz = (1.0 - replay_weight) * onpolicy_loss_logz + replay_weight * replay_loss_logz
                    replay_fraction_used += replay_weight
                elif onpolicy_loss is not None:
                    loss = onpolicy_loss
                    loss_flow = onpolicy_loss_flow
                    loss_reg = onpolicy_loss_reg
                    loss_logz = onpolicy_loss_logz
                    replay_fraction_used += 0.0
                elif replay_loss is not None:
                    loss = replay_loss
                    loss_flow = replay_loss_flow
                    loss_reg = replay_loss_reg
                    loss_logz = replay_loss_logz
                    replay_fraction_used += 1.0
                else:
                    continue

                loss_total_val += loss.item()
                loss_flow_val += loss_flow.item()
                loss_reg_val += loss_reg.item()
                loss_logz_val += loss_logz.item()
                executed_updates += 1

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            if executed_updates > 0:
                denom = float(executed_updates)
                loss_total_val /= denom
                loss_flow_val /= denom
                loss_reg_val /= denom
                loss_logz_val /= denom
                replay_fraction_used /= denom
                n_train_batch = int(round(total_train_batch / denom)) if total_train_batch > 0 else 0
                n_replay_batch = int(round(total_replay_batch / denom)) if total_replay_batch > 0 else 0
            if reward_terms > 0:
                mean_train_reward /= float(reward_terms)

            beta = _update_beta(
                beta=beta,
                beta_rate=beta_rate,
                beta_max=beta_max,
                log_z_value=log_z.detach().item(),
                best_reward=policy_best_ema if policy_best_ema is not None else best_reward_seen,
                min_reward=min_reward,
            )

            training_times.append(perf_counter()-training_start)
            print(
                f"[train] step={step} "
                f"loss_total={loss_total_val:.4f} "
                f"loss_flow={loss_flow_val:.4f} "
                f"loss_reg={loss_reg_val:.4f} "
                f"loss_logz={loss_logz_val:.4f} "
                f"beta={beta:.4f} "
                f"log_z={log_z.detach().item():.4f} "
                f"weight_temp={current_weight_temp:.4f} "
                f"ga_mix_ratio={ga_mix_ratio_curr:.4f} "
                f"replay_frac={replay_fraction_used:.4f} "
                f"replay_frac_target={current_replay_blend:.4f} "
                f"best_reward={best_reward_seen:.4f} "
                f"policy_best_ema={policy_best_ema if policy_best_ema is not None else 0.0:.4f} "
                f"mean_train_reward={mean_train_reward:.4f} "
                f"n_policy_candidates={n_policy_candidates} "
                f"n_ga_candidates={n_ga_candidates} "
                f"n_train_batch={n_train_batch} "
                f"n_replay_batch={n_replay_batch}"
            )
            try:
                wandb.log({
                    'loss_total': loss_total_val,
                    'loss_flow': loss_flow_val,
                    'loss_reg': loss_reg_val,
                    'loss_logz': loss_logz_val,
                    'beta': beta,
                    'log_z': log_z.detach().item(),
                    'weight_temp': current_weight_temp,
                    'ga_mix_ratio': ga_mix_ratio_curr,
                    'replay_frac': replay_fraction_used,
                    'replay_frac_target': current_replay_blend,
                    'policy_best_ema': policy_best_ema if policy_best_ema is not None else 0.0,
                    'best_reward': best_reward_seen,
                    'mean_train_reward': mean_train_reward,
                    'n_policy_candidates': n_policy_candidates,
                    'n_ga_candidates': n_ga_candidates,
                    'n_train_batch': n_train_batch,
                    'n_replay_batch': n_replay_batch,
                })
            except Exception:
                pass
            step += 1
        
        best_scores = np.array(best_scores, dtype=float)
        if best_scores.size == 0:
            best_scores = np.empty((0, 3), dtype=float)
        elif best_scores.ndim == 1:
            best_scores = best_scores.reshape(-1, 3)

        results = {'exp_policy_smiles_novelty': policy_smiles_novelty, 
                    'exp_policy_smiles_norm_novelty': policy_smiles_norm_novelty, 
                    'exp_policy_mol_novelty': policy_mol_novelty, 
                    'exp_ga_smiles_norm_novelty': ga_smiles_norm_novelty, 
                    'exp_ga_smiles_novelty': ga_smiles_novelty, 
                    'exp_ga_mol_novelty': ga_mol_novelty,
                    'policy_ga_smiles_norm_novelty': policy_ga_smiles_norm_novelty, 
                    'policy_ga_smiles_novelty': policy_ga_smiles_novelty, 
                    'policy_ga_mol_novelty': policy_ga_mol_novelty,
                    'policy_smiles_norm_diversity': policy_smiles_norm_diversity,
                    'policy_smiles_diversity': policy_smiles_diversity,
                    'policy_mol_diversity': policy_mol_diversity,
                    'policy_best_scores': best_scores[:, 0],
                    'ga_best_scores': best_scores[:, 1],
                    'tot_best_scores': best_scores[:, 2],
                    'ga_runtime': np.sum(ga_times),
                    'eval_runtime': np.sum(eval_times),
                    'training_runtime': np.sum(training_times)
                    }
        
        print('Total runtime:', perf_counter() - total_start)
        print('GA runtime:', np.sum(ga_times))
        print('Eval runtime:', np.sum(eval_times))
        print('Training runtime:', np.sum(training_times))
        
        try:
            wandb.log({'Total runtime': perf_counter() - total_start,
                      'ga_runtime': np.sum(ga_times),
                    'eval_runtime': np.sum(eval_times),
                    'training_runtime': np.sum(training_times)})
        except:
            pass
        
        if config['rank_coefficient'] < 0.1:
            results_dir = os.path.join(path_here, "ga_results")
            os.makedirs(results_dir, exist_ok=True)
            with open(os.path.join(results_dir, f'run_{oracle.name}_results_seed{self.seed}.pkl'), 'wb') as f:
                pickle.dump(results, f)
            # results.to_pickle('./main/genetic_gfn/ga_results/run_' + oracle.name + '_results.pkl')
            tot_ga_results.to_pickle(f'./main/genetic_gfn/ga_results/run_{oracle.name}_ga_results_seed{self.seed}.pkl')
            tot_ga_results.to_csv(f'./main/genetic_gfn/ga_results/run_{oracle.name}_ga_results_seed{self.seed}.csv', index=False)
