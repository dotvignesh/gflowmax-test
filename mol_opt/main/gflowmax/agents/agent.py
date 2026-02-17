from collections import defaultdict, deque
import os
import random
import sys

import numpy as np
from rdkit import Chem

try:
    from main.moldqn.environments import environments as envs
except ModuleNotFoundError:
    # Fallback for benchmark entrypoints where moldqn is not on sys.path.
    moldqn_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "moldqn"))
    if moldqn_root not in sys.path:
        sys.path.append(moldqn_root)
    from environments import environments as envs


class GFlowMaxAgent:
    """DBReplay-style molecular optimizer with flow replay and explicit reverse edges."""

    def __init__(self, oracle, args, n_max_oracle_call=10000):
        self.oracle = oracle
        self.args = args
        self.n_max_oracle_call = n_max_oracle_call

        self.num_episodes = self._get("num_episodes", 200000)
        self.max_steps_per_episode = self._get("max_steps_per_episode", 40)
        self.init_mol = self._get("init_mol", "C")
        self.discount_factor = self._get("discount_factor", 0.9)
        self.atom_types = self._get("atom_types", ["C", "O", "N"])
        self.allow_removal = self._get("allow_removal", True)
        self.allow_no_modification = self._get("allow_no_modification", True)
        self.allow_bonds_between_rings = self._get("allow_bonds_between_rings", False)
        self.allowed_ring_sizes = self._get("allowed_ring_sizes", [5, 6])

        self.lr = self._get("learning_rate", 0.05)
        self.epsilon = self._get("epsilon", 0.5)
        self.epsilon_min = self._get("epsilon_min", 0.05)
        self.epsilon_decay = self._get("epsilon_decay", 0.9995)
        self.batch_size = self._get("batch_size", 128)
        self.tau = self._get("tau", 0.01)
        self.r_min = self._get("r_min", 1e-4)
        self.priority_eps = self._get("priority_epsilon", 1e-6)
        self.min_steps_before_noop = self._get("min_steps_before_noop", 2)
        self.treat_noop_as_terminal = self._get("treat_noop_as_terminal", True)
        self.score_nonterminal_steps = self._get("score_nonterminal_steps", False)

        self.beta = self._get("beta_init", 1.0)
        self.beta_rate = self._get("beta_rate", 0.01)
        self.beta_max = self._get("beta_max", 25.0)

        seed = self._get("seed", 0)
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

        self.env = envs.Molecule(
            atom_types=set(self.atom_types),
            discount_factor=self.discount_factor,
            oracle=self.oracle,
            init_mol=self.init_mol,
            allow_removal=self.allow_removal,
            allow_no_modification=self.allow_no_modification,
            allow_bonds_between_rings=self.allow_bonds_between_rings,
            allowed_ring_sizes=set(self.allowed_ring_sizes),
            max_steps=self.max_steps_per_episode,
        )

        self.log_flow = defaultdict(float)
        self.log_flow_target = defaultdict(float)
        self.pf_logits = defaultdict(float)
        self.pb_logits = defaultdict(float)

        self.base_valid_action_cache = {}
        self.reverse_candidate_cache = {}
        self.parent_smiles_cache = {}
        self.valid_action_cache = {}
        self.expanded_states = set()
        self.parent_cache = {}
        self.reverse_parents = defaultdict(set)

        self.buffer = deque(maxlen=self._get("replay_buffer_size", 20000))
        self.priorities = deque(maxlen=self._get("replay_buffer_size", 20000))
        self.best_reward = 1e-10

    def _get(self, name, default):
        if isinstance(self.args, dict):
            return self.args.get(name, default)
        return getattr(self.args, name, default)

    def _state_key(self, smiles, step):
        return (smiles, int(step))

    def _terminal_key(self, smiles, step):
        return ("__terminal__", smiles, int(step))

    def _is_terminal_key(self, state):
        return isinstance(state, tuple) and len(state) == 3 and state[0] == "__terminal__"

    def _state_smiles(self, state):
        if self._is_terminal_key(state):
            return state[1]
        if isinstance(state, tuple):
            return state[0]
        return state

    def _state_step(self, state):
        if self._is_terminal_key(state):
            return int(state[2])
        if isinstance(state, tuple) and len(state) >= 2:
            return int(state[1])
        return 0

    def _next_state_key(self, state, next_smiles):
        if isinstance(state, tuple) and len(state) >= 2:
            return self._state_key(next_smiles, state[1] + 1)
        return self._state_key(next_smiles, 1)

    def _get_base_valid_actions(self, smiles):
        if smiles not in self.base_valid_action_cache:
            self.base_valid_action_cache[smiles] = tuple(
                sorted(self.env.get_valid_actions(state=smiles))
            )
        return list(self.base_valid_action_cache[smiles])

    def _mask_stop_action(self, state_smiles, actions, step):
        if step < self.min_steps_before_noop:
            filtered = [a for a in actions if a != state_smiles]
            if filtered:
                return filtered
        return actions

    def _get_reverse_candidates(self, next_smiles):
        """
        Candidate predecessors for `next_smiles`.
        Use allow_removal=True so atom/bond-addition parents remain recoverable
        even when the forward policy forbids removal.
        """
        if next_smiles not in self.reverse_candidate_cache:
            candidates = set(
                envs.get_valid_actions(
                    next_smiles,
                    atom_types=self.env.atom_types,
                    allow_removal=True,
                    allow_no_modification=True,
                    allowed_ring_sizes=self.env.allowed_ring_sizes,
                    allow_bonds_between_rings=self.env.allow_bonds_between_rings,
                )
            )
            # The moldqn bond-removal helper keeps only one disconnected fragment.
            # Add all disconnected pieces from single-bond cuts so parent recovery
            # can still find the true predecessor in atom-addition transitions.
            mol = Chem.MolFromSmiles(next_smiles)
            if mol is not None:
                for bond in mol.GetBonds():
                    atom1 = bond.GetBeginAtomIdx()
                    atom2 = bond.GetEndAtomIdx()
                    cut = Chem.RWMol(mol)
                    cut.RemoveBond(atom1, atom2)
                    if Chem.SanitizeMol(cut, catchErrors=True):
                        continue
                    fragmented = Chem.MolToSmiles(cut)
                    if "." not in fragmented:
                        continue
                    for part in fragmented.split("."):
                        p = Chem.MolFromSmiles(part)
                        if p is not None:
                            candidates.add(Chem.MolToSmiles(p))
            candidates.add(next_smiles)
            self.reverse_candidate_cache[next_smiles] = tuple(sorted(candidates))
        return list(self.reverse_candidate_cache[next_smiles])

    def _get_parent_smiles(self, next_smiles):
        """
        Exact parent smiles set under the configured forward transition rules.
        """
        if next_smiles not in self.parent_smiles_cache:
            parents = []
            for parent_smiles in self._get_reverse_candidates(next_smiles):
                if next_smiles in self._get_base_valid_actions(parent_smiles):
                    parents.append(parent_smiles)
            self.parent_smiles_cache[next_smiles] = tuple(sorted(set(parents)))
        return list(self.parent_smiles_cache[next_smiles])

    def _log_softmax(self, values):
        if not values:
            return {}
        max_value = max(values.values())
        exp_values = {k: np.exp(v - max_value) for k, v in values.items()}
        norm = sum(exp_values.values()) + 1e-12
        return {k: np.log(v / norm + 1e-12) for k, v in exp_values.items()}

    def _softmax(self, values):
        if not values:
            return {}
        max_value = max(values.values())
        exp_values = {k: np.exp(v - max_value) for k, v in values.items()}
        norm = sum(exp_values.values()) + 1e-12
        return {k: v / norm for k, v in exp_values.items()}

    def _expand_state(self, state):
        """Materialize forward actions and reverse parent index for this state."""
        if state in self.expanded_states:
            return
        if self._is_terminal_key(state):
            self.valid_action_cache[state] = []
            self.expanded_states.add(state)
            return

        state_smiles = self._state_smiles(state)
        state_step = self._state_step(state)
        # Keep action order fixed across runs for reproducibility.
        actions = self._get_base_valid_actions(state_smiles)
        actions = self._mask_stop_action(state_smiles, actions, state_step)
        self.valid_action_cache[state] = actions
        for next_smiles in actions:
            _ = self.pf_logits[(state, next_smiles)]
            next_state = self._next_state_key(state, next_smiles)
            self.reverse_parents[next_state].add(state)
            _ = self.pb_logits[(next_state, state)]
        self.expanded_states.add(state)

    def _get_valid_actions(self, state):
        self._expand_state(state)
        return self.valid_action_cache[state]

    def _get_parents(self, next_state):
        if next_state in self.parent_cache:
            parents = list(self.parent_cache[next_state])
        else:
            next_smiles = self._state_smiles(next_state)
            next_step = self._state_step(next_state)
            parent_step = next_step - 1

            if parent_step < 0:
                parents = []
            else:
                parents = []
                for parent_smiles in self._get_parent_smiles(next_smiles):
                    # Step-based masking only affects no-op availability.
                    if (
                        parent_smiles == next_smiles
                        and parent_step < self.min_steps_before_noop
                    ):
                        continue
                    parent_state = self._state_key(parent_smiles, parent_step)
                    parents.append(parent_state)
                self.parent_cache[next_state] = tuple(parents)

        # Keep known trajectory parents as a fallback for edge cases where
        # chemistry reversibility does not recover all predecessors.
        if self.reverse_parents[next_state]:
            merged = set(parents)
            merged.update(self.reverse_parents[next_state])
            parents = sorted(merged)

        for parent in parents:
            _ = self.pb_logits[(next_state, parent)]
        return parents

    def _forward_log_probs(self, state):
        actions = self._get_valid_actions(state)
        if not actions:
            return {}
        logits = {action: self.pf_logits[(state, action)] for action in actions}
        return self._log_softmax(logits)

    def sample_action(self, state, step_idx):
        valid_actions = self._get_valid_actions(state)
        if not valid_actions:
            return None

        if self.rng.random() < self.epsilon:
            return self.rng.choice(valid_actions)

        probs = self._softmax({a: self.pf_logits[(state, a)] for a in valid_actions})
        actions = list(probs.keys())
        values = np.array([probs[a] for a in actions], dtype=np.float64)
        values = values / values.sum()
        idx = np.searchsorted(np.cumsum(values), self.rng.random(), side="right")
        idx = min(idx, len(actions) - 1)
        return actions[idx]

    def update_transition(self, state, action, next_state, is_terminal, terminal_reward):
        self._expand_state(state)
        self.log_flow[state] += 0.0
        self.log_flow[next_state] += 0.0
        self.log_flow_target[state] += 0.0
        self.log_flow_target[next_state] += 0.0

        valid_actions = self.valid_action_cache[state]
        parents = self._get_parents(next_state)
        if state not in self.reverse_parents[next_state]:
            self.reverse_parents[next_state].add(state)
            _ = self.pb_logits[(next_state, state)]
            parents = parents + [state]

        forward_lp = self._forward_log_probs(state)
        backward_lp = self._log_softmax({parent: self.pb_logits[(next_state, parent)] for parent in parents})

        log_f_s = self.log_flow[state]
        log_pf = forward_lp.get(action, -10.0)
        log_f_sp = self.beta * np.log(max(terminal_reward, self.r_min)) if is_terminal else self.log_flow_target[next_state]
        log_pb = backward_lp.get(state, -10.0)
        delta = (log_f_s + log_pf) - (log_f_sp + log_pb)
        grad = 2.0 * delta * self.lr

        self.log_flow[state] -= np.clip(grad, -5.0, 5.0)

        for candidate in valid_actions:
            prob = np.exp(forward_lp.get(candidate, -10.0))
            weight = (1.0 - prob) if candidate == action else -prob
            self.pf_logits[(state, candidate)] -= np.clip(grad * weight, -5.0, 5.0)

        for parent in parents:
            prob = np.exp(backward_lp.get(parent, -10.0))
            weight = (1.0 - prob) if parent == state else -prob
            self.pb_logits[(next_state, parent)] += np.clip(grad * weight, -5.0, 5.0)

        return float(delta)

    def train_episode(self):
        state_smiles, state_step = self.env.reset()
        state = self._state_key(state_smiles, state_step)
        final_reward = 0.0

        for step_idx in range(self.max_steps_per_episode):
            if self.oracle.finish:
                break

            action = self.sample_action(state, step_idx)
            if action is None:
                break

            stop_selected = action == self._state_smiles(state)
            if self.score_nonterminal_steps:
                next_smiles, next_step, reward, done = self.env.step(action)
                is_terminal = bool(done or (self.treat_noop_as_terminal and stop_selected))
                terminal_reward = max(self.r_min, float(reward))
            else:
                next_smiles = action
                next_step = state[1] + 1
                reached_horizon = next_step >= self.max_steps_per_episode
                is_terminal = bool(reached_horizon or (self.treat_noop_as_terminal and stop_selected))
                terminal_reward = self.r_min
                if is_terminal:
                    terminal_reward = max(self.r_min, float(self.oracle(next_smiles)))

            if is_terminal:
                next_state = self._terminal_key(next_smiles, next_step)
            else:
                next_state = self._state_key(next_smiles, next_step)

            self.reverse_parents[next_state].add(state)
            _ = self.pb_logits[(next_state, state)]

            self.buffer.append((state, action, next_state, is_terminal, terminal_reward))
            self.priorities.append(1.0)
            if is_terminal:
                self.best_reward = max(self.best_reward, terminal_reward)
                final_reward = terminal_reward

            if is_terminal or self.oracle.finish:
                break

            self._expand_state(next_state)
            state = next_state

        if len(self.buffer) >= self.batch_size:
            prob = np.array(self.priorities, dtype=np.float64)
            if np.sum(prob) <= 0:
                prob = np.ones_like(prob) / len(prob)
            else:
                prob = prob / np.sum(prob)
            indices = self.np_rng.choice(len(self.buffer), self.batch_size, p=prob, replace=False)

            for idx in indices:
                batch_transition = self.buffer[idx]
                delta = self.update_transition(*batch_transition)
                self.priorities[idx] = abs(delta) + self.priority_eps

            # Match DBReplay update semantics: soft-update target for all known states.
            for known_state in list(self.log_flow.keys()):
                self.log_flow_target[known_state] = (
                    (1.0 - self.tau) * self.log_flow_target[known_state]
                    + self.tau * self.log_flow[known_state]
                )

        self.beta = min(self.beta + self.beta_rate, self.beta_max)
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        return final_reward
