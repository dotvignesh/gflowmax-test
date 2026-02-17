# REINVENT_SELFIES x GFlowMax Hybrid: Replacement Scope

This document defines what to replace, what to keep, and how to stage a hybrid training objective that combines REINVENT-style sequence policy optimization with GFlowMax-style flow consistency.

## 1) Goal and Non-Goals

Prerequisite (must do first):
- Clone `reinvent` into a new method folder `reinvent-gfm` before any hybrid modifications.
- Do not implement hybrid logic directly inside `reinvent` or `reinvent_selfies`.
- Treat `reinvent-gfm` as the working branch of code for all objective/replay/model changes in this scope.

Goal:
- Keep SELFIES token generation and prior pretraining from `reinvent_selfies`.
- Inject a GFlowMax-like flow objective and replay behavior into training.
- Preserve `mol_opt` benchmark integration (oracle budget, logging, result saving).

Non-goals:
- Do not rewrite the whole method into the graph-edit environment used by `gflowmax`.
- Do not change dataset preparation or prior pretraining unless needed for compatibility.

## 2) Current System Boundaries

### 2.1 REINVENT_SELFIES side

Core files:
- `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/run.py`
- `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/model.py`
- `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/data_structs.py`
- `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/data.py`
- `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/pretrain.py`

Current training signal in `run.py`:
- Samples sequences from Agent.
- Scores terminal decoded molecules with oracle.
- Uses REINVENT augmented-likelihood loss:
  - `L_reinvent = (logP_prior(x) + sigma * R(x) - logP_agent(x))^2`
- Uses simple sequence-level prioritized replay from `Experience`.

Current representation:
- State is implicit in autoregressive token prefix.
- Action is next SELFIES token.
- Terminal is `[EOS]` or max length.

### 2.2 GFlowMax side

Core files:
- `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/gflowmax_mdp/run.py`

Current GFlowMax-style signal:
- Transition-level flow consistency using `(logF + logPf) - (target flow + logPb)`.
- Prioritized replay with priorities from temporal flow residual (`|delta|`).
- Beta schedule on terminal reward shaping.

Current representation:
- SMILES graph-edit MDP using moldqn action generator:
  - `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/moldqn/environments/environments.py`

## 3) Key Mismatch You Must Resolve

Mismatch:
- `reinvent_selfies` is token-autoregressive sequence generation.
- `gflowmax` is graph-edit action space over valid molecule edits.

Implication:
- You should hybridize objective and replay mechanics, not directly transplant graph-edit transition logic.
- A clean hybrid should stay in the SELFIES token MDP to reuse prior/vocab/model.

## 4) Replacement Scope by Priority

## 4.1 Must Replace (for a real hybrid objective)

1. Training loop objective composition  
File: `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/run.py`
- Replace single-objective update with combined objective:
  - REINVENT term (keep).
  - Flow term (add).
  - Optional entropy/regularization term (keep or tune down).

2. Replay buffer schema  
File: `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/data_structs.py`
- Replace sequence-only replay tuples `(selfies, score, prior_ll)` with transition-aware entries:
  - `state_prefix`
  - `action_token`
  - `next_prefix`
  - `done`
  - `terminal_reward`
  - optional cached `prior_ll_seq`, `seq_id`
- Add PER priorities based on flow residual instead of only score.

3. Agent model API for stepwise objective terms  
File: `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/model.py`
- Add methods to get step logits and hidden states for prefix transitions.
- Add optional flow head `logF(s)` if using state-flow local consistency.

## 4.2 Should Replace (strongly recommended)

4. Config surface  
File: `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/hparams_default.yaml`
- Keep: `learning_rate`, `batch_size`, `sigma`.
- Add:
  - `objective_mix_lambda`
  - `beta_init`, `beta_rate`, `beta_max`
  - `r_min`
  - `flow_replay_size`
  - `priority_alpha`, `priority_eps`
  - `flow_target_tau`
  - `max_seq_len`
  - `terminal_only_oracle`
  - `warmup_reinvent_steps`

5. Logging/diagnostics  
File: `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/run.py`
- Add logs:
  - `reinvent_loss`, `flow_loss`, `total_loss`
  - `beta`, replay effective sample size, mean `|delta|`
  - terminal reward stats

## 4.3 Keep Unchanged (or minimally touched)

6. Data and vocab build  
File: `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/data.py`

7. Prior pretraining  
File: `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent_selfies/pretrain.py`

8. Benchmark integration and oracle accounting  
Files:
- `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/optimizer.py`
- `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/run.py`

## 5) Hybrid Objective Options (from easiest to strongest)

## Option A: Minimal Hybrid (lowest surgery)

Use sequence-level GFlow objective plus REINVENT objective.

Definition:
- Keep current sequence log-likelihood `logP_agent(x)` and `logP_prior(x)`.
- Add a trajectory-balance-like terminal objective:
  - `L_flow_seq = (logZ + logP_agent(x) - beta * log(max(R(x), r_min)))^2`
- Total:
  - `L_total = lambda * L_reinvent + (1 - lambda) * L_flow_seq + reg`

Why this is attractive:
- No parent enumeration needed.
- No graph-edit migration.
- Uses existing `sample()` and `likelihood()` with minor model changes (`logZ` parameter).

What to replace:
- `run.py` objective section.
- Add scalar `logZ` parameter (can be in model or trainer state).
- Replay can remain sequence-level initially.

## Option B: Transition Flow Hybrid (recommended target)

Use local flow consistency on token-prefix transitions, with optional REINVENT anchor.

For each transition `(s, a, s')`:
- Non-terminal:
  - `delta = logF(s) + logPf(a|s) - logF_target(s')`
- Terminal:
  - `delta = logF(s) + logPf(a|s) - beta * log(max(R(x), r_min))`
- `L_flow = E[delta^2]`

Hybrid:
- `L_total = lambda * L_reinvent + (1 - lambda) * L_flow + reg`

Why this is better:
- Closer to GFlowMax/DBReplay mechanics.
- Supports PER by `|delta|`.
- More stable long horizon credit assignment than REINVENT-only.

Required replacements:
- `model.py`: flow head + prefix-step API.
- `data_structs.py`: transition replay + PER updates.
- `run.py`: transition extraction, flow loss, target update.

## 6) Concrete File-Level Change Plan

## 6.1 `/.../reinvent_selfies/model.py`

Add:
- `flow_head = nn.Linear(hidden_dim, 1)` in `MultiGRU`.
- `step(x_t, h)` method returning `(logits_t, h_next, logF_t)`.
- `forward_prefix(prefix_batch)` utility for batched prefixes.
- Optional `sample_with_trace(...)` returning token-level traces:
  - prefixes
  - sampled tokens
  - logPf per step
  - terminal flags

Keep:
- Existing `sample()` and `likelihood()` for backward compatibility.

## 6.2 `/.../reinvent_selfies/data_structs.py`

Replace `Experience` with a transition replay class:
- `add_trajectory(trace, terminal_reward, prior_ll=None)`
- `sample_transitions(n)` returning tensors for `(s, a, s', done, reward)`
- `update_priorities(indices, deltas, alpha, eps)`

Important:
- Use fixed max size and dedupe by hashed transition key.
- Avoid storing huge Python strings for every prefix; store token tuples or compact arrays.

## 6.3 `/.../reinvent_selfies/run.py`

Refactor `_optimize` loop into phases:
- Sample trajectory traces from Agent.
- Decode only terminal SELFIES to SMILES and query oracle.
- Push transitions into PER.
- If replay warm enough:
  - compute `L_flow` on replay transitions.
  - optionally compute `L_reinvent` on fresh batch.
  - combine and step optimizer.
- Update beta schedule and target flow parameters.
- Keep existing early-stop/budget logic via `self.finish`.

Also:
- Add config defaults and guards for missing keys.
- Keep `self.oracle.assign_evaluator(oracle)`.

## 6.4 `/.../reinvent_selfies/hparams_default.yaml`

Suggested starting config:
- keep existing:
  - `learning_rate`
  - `batch_size`
  - `sigma`
- add:
  - `objective_mix_lambda: 0.6`
  - `beta_init: 1.0`
  - `beta_rate: 0.008`
  - `beta_max: 25.0`
  - `r_min: 0.0001`
  - `flow_replay_size: 20000`
  - `priority_alpha: 1.0`
  - `priority_eps: 0.000001`
  - `flow_target_tau: 0.01`
  - `terminal_only_oracle: true`
  - `warmup_reinvent_steps: 200`

## 7) Mapping REINVENT Pieces to GFlowMax Counterparts

REINVENT current:
- `logP_prior + sigma * R` target
- sequence replay by score
- no explicit flow state value

Hybrid/GFlowMax counterpart:
- terminal flow target `beta * log(R)`
- PER by flow residual
- explicit `logF(s)` and target flow update

Clean conceptual mapping:
- REINVENT gives a high-quality terminal preference anchor.
- GFlowMax gives transition credit assignment and replay prioritization.
- Combined objective is naturally complementary.

## 8) Practical Risk List and Mitigation

Risk: reward sparsity and unstable early flow updates  
Mitigation:
- warm start with REINVENT-only for N steps.
- use `objective_mix_lambda` anneal from high to medium.

Risk: oracle budget waste on non-terminal prefixes  
Mitigation:
- terminal-only oracle scoring (same spirit as `gflowmax` config).

Risk: prefix state explosion in replay  
Mitigation:
- cap replay.
- store compact token IDs.
- evict by age + priority.

Risk: invalid decoded molecules  
Mitigation:
- assign reward 0 with `r_min` clamp in flow term.

Risk: objective scale mismatch (`sigma` vs `beta`)  
Mitigation:
- start with normalized rewards.
- tune `sigma`, `beta_rate`, and objective mix jointly.

## 9) Recommended Implementation Phases

Phase 0 (mandatory bootstrap):
- Copy/clone `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent` to `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/main/reinvent-gfm`.
- Register `reinvent-gfm` in `/Users/vigneshraja/Documents/NEU Notes/Thesis/GFlowMax-Molecules/mol_opt/run.py` method dispatch before training changes.
- Keep `reinvent` unchanged as baseline reference.

Phase 1 (fast validation):
- Implement Option A minimal hybrid.
- Confirm no regression in oracle-budget behavior.
- Compare against baseline REINVENT_SELFIES on QED.

Phase 2 (target hybrid):
- Add flow head and transition PER (Option B).
- Keep REINVENT anchor term with annealed mixing.
- Add diagnostics for `flow_loss` and residual distribution.

Phase 3 (ablation/tuning):
- `lambda` sweep.
- `beta` schedule sweep.
- replay size and PER alpha sweep.
- terminal-only vs mixed scoring ablation.

## 10) What You Should Not Replace First

Do not first replace:
- `data.py` vocabulary generation.
- `pretrain.py` prior training routine.
- `BaseOptimizer` and `Oracle` contracts.

Reason:
- These are stable and already wired into benchmark logging and budgeting.
- Most hybrid value comes from objective/replay, not data path rewrite.

## 11) Acceptance Criteria for “Hybrid Complete”

A hybrid implementation should satisfy:
- Trains and finishes under existing `max_oracle_calls`.
- Reports both REINVENT and flow losses.
- Uses prioritized replay keyed by flow residual.
- Preserves SELFIES generation pipeline and prior checkpoint usage.
- Produces at least parity with REINVENT_SELFIES on one standard oracle before further tuning.

---

If you want, the next step can be a precise patch plan with function signatures and pseudocode blocks for each modified file before coding.
