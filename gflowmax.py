"""
DB-β vs DQN-PER: 32×32 Super Hard Deceptive Grid
===================================================
The ultimate test. 32×32 DAG, ~62 step optimal path.

Design principles:
  - Progressive deception: traps go R=0.1 → R=0.92 with depth
  - Multiple trap corridors that look increasingly optimal
  - Dense holes creating maze-like structure
  - Goal R=1.0 requires threading a specific narrow corridor
  - Bellman needs ~62 successful backward propagations from a single goal obs

5 seeds. 200k trajectory budget. Sample efficiency only.
"""

import numpy as np
import random
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


# ============================================================================
# 32×32 SUPER HARD DECEPTIVE GRID
# ============================================================================

class SuperHardGrid:
    def __init__(self, seed=0):
        self.rows = 32; self.cols = 32
        self.start = (0,0); self.goal = (31,31)
        self.n_actions = 2

        rng = np.random.RandomState(seed)

        # --- HOLES: dense maze structure ---
        self.holes = set()

        # Diagonal barriers with narrow gaps
        for i in range(2, 30):
            if i % 5 not in [0, 1]:  # gaps every 5 rows
                self.holes.add((i, i))
            if i % 7 not in [0]:
                if i+2 < 32 and i-2 >= 0:
                    self.holes.add((i, i+2))

        # Systematic hole placement creating corridors
        for r in range(2, 30):
            for c in range(2, 30):
                # Skip terminals we'll place
                if (r,c) == self.goal: continue
                # Place holes in a pattern that creates maze corridors
                if (r*7 + c*13 + seed) % 11 == 0:
                    self.holes.add((r,c))
                if (r*3 + c*17 + seed) % 13 == 0:
                    self.holes.add((r,c))

        # Clear a path to goal (ensure solvability) — clear right-then-down-then-right corridor
        # Path 1: along row 0 to col 15, down to row 31, right to col 31
        for c in range(32): self.holes.discard((0, c))
        for r in range(32): self.holes.discard((r, 15))
        for c in range(15, 32): self.holes.discard((31, c))
        # Path 2: down to row 15, right to col 31, down to row 31
        for r in range(32): self.holes.discard((r, 0))
        for c in range(32): self.holes.discard((15, c))
        for r in range(15, 32): self.holes.discard((r, 31))
        # Clear some additional paths
        for r in range(32): self.holes.discard((r, 8)); self.holes.discard((r, 24))
        for c in range(32): self.holes.discard((8, c)); self.holes.discard((24, c))

        # --- TRAPS: progressive deception ---
        self.traps = {}

        # Layer 1 (rows 1-7): weak traps R=0.05-0.10
        trap_specs_1 = [
            (1,3), (2,5), (3,1), (4,6), (5,3), (6,1), (7,5),
            (1,8), (3,10), (5,12), (7,8),
        ]
        for i, pos in enumerate(trap_specs_1):
            if pos not in self.holes and pos != self.goal:
                self.traps[pos] = 0.05 + 0.005 * i

        # Layer 2 (rows 8-14): moderate traps R=0.10-0.20
        trap_specs_2 = [
            (8,2), (9,5), (10,3), (11,6), (12,1), (13,4), (14,2),
            (8,10), (10,12), (12,8), (14,11),
            (9,16), (11,18), (13,14),
        ]
        for i, pos in enumerate(trap_specs_2):
            if pos not in self.holes and pos != self.goal:
                self.traps[pos] = 0.10 + 0.008 * i

        # Layer 3 (rows 15-22): strong traps R=0.20-0.35
        trap_specs_3 = [
            (15,3), (16,5), (17,2), (18,6), (19,4), (20,1), (21,5), (22,3),
            (15,10), (17,12), (19,9), (21,11),
            (16,18), (18,16), (20,14), (22,17),
            (15,22), (18,20), (21,24),
        ]
        for i, pos in enumerate(trap_specs_3):
            if pos not in self.holes and pos != self.goal:
                self.traps[pos] = 0.20 + 0.008 * i

        # Layer 4 (rows 23-30): deep traps R=0.35-0.50
        trap_specs_4 = [
            (23,2), (24,5), (25,3), (26,6), (27,1), (28,4), (29,2), (30,5),
            (23,10), (25,12), (27,9), (29,11),
            (24,18), (26,16), (28,14), (30,17),
            (23,22), (25,20), (27,24), (29,22),
            (24,28), (26,26), (28,25), (30,28),
            (31,10), (31,20), (31,25),
        ]
        for i, pos in enumerate(trap_specs_4):
            if pos not in self.holes and pos != self.goal:
                self.traps[pos] = 0.35 + 0.006 * i

        # Edge terminals
        self.edge_terminals = {
            (0, 31): 0.08, (31, 0): 0.05,
            (15, 31): 0.25, (31, 15): 0.30,
            (7, 31): 0.15, (31, 7): 0.12,
            (23, 31): 0.35, (31, 23): 0.38,
        }

        # Remove any traps/edges that overlap with holes or goal
        self.traps = {k:v for k,v in self.traps.items() if k not in self.holes and k != self.goal}
        self.edge_terminals = {k:v for k,v in self.edge_terminals.items()
                               if k not in self.holes and k not in self.traps and k != self.goal}

        # Build state space
        self.states = [(r,c) for r in range(self.rows) for c in range(self.cols)]
        self.n_states = len(self.states)

        self.terminal_states = set(); self.terminal_rewards = {}
        self.terminal_states.add(self.goal); self.terminal_rewards[self.goal] = 1.0
        for h in self.holes: self.terminal_states.add(h); self.terminal_rewards[h] = 0.01
        for t,r in self.traps.items(): self.terminal_states.add(t); self.terminal_rewards[t] = r
        for t,r in self.edge_terminals.items(): self.terminal_states.add(t); self.terminal_rewards[t] = r

        # Build graph
        self.valid_actions = {}; self.children = defaultdict(list); self.parents = defaultdict(list)
        for r in range(self.rows):
            for c in range(self.cols):
                s = (r,c)
                if s in self.terminal_states: self.valid_actions[s]=[]; continue
                actions=[]
                if c+1<self.cols:
                    actions.append(0); sn=(r,c+1)
                    self.children[s].append((0,sn)); self.parents[sn].append((0,s))
                if r+1<self.rows:
                    actions.append(1); sn=(r+1,c)
                    self.children[s].append((1,sn)); self.parents[sn].append((1,s))
                self.valid_actions[s] = actions

        # Verify goal is reachable
        self._verify()

    def _verify(self):
        """BFS to verify goal is reachable."""
        visited = set(); queue = [self.start]; visited.add(self.start)
        goal_reachable = False
        while queue:
            s = queue.pop(0)
            if s == self.goal: goal_reachable = True
            for _, sn in self.children[s]:
                if sn not in visited:
                    visited.add(sn)
                    if not self.is_terminal(sn) or sn == self.goal:
                        queue.append(sn)

        n_traps = len(self.traps)
        n_holes = len(self.holes)
        trap_rewards = sorted(self.traps.values())
        print(f"  Grid: {self.rows}×{self.cols}")
        print(f"  Holes: {n_holes}, Traps: {n_traps}")
        if trap_rewards:
            print(f"  Trap rewards: {trap_rewards[0]:.3f} - {trap_rewards[-1]:.3f}")
        print(f"  Goal reachable: {goal_reachable}")
        print(f"  Min path length: ~{self.rows + self.cols - 2} steps")

    def step(self, s, a):
        r,c=s; ns=(r,c+1) if a==0 else (r+1,c)
        return ns, self.terminal_rewards.get(ns,0.0), ns in self.terminal_states

    def is_terminal(self, s): return s in self.terminal_states
    def get_terminal_reward(self, s): return self.terminal_rewards.get(s,0.0)

    def sample_trajectory(self, policy_fn, epsilon=0.1, rng=None):
        if rng is None: rng=random
        s=self.start; traj=[s]; actions=[]
        for _ in range(80):  # max 80 steps for 32x32
            if self.is_terminal(s): break
            valid=self.valid_actions.get(s,[])
            if not valid: break
            a=rng.choice(valid) if rng.random()<epsilon else policy_fn(s,valid)
            ns,_,done=self.step(s,a); actions.append(a); traj.append(ns); s=ns
            if done: break
        return traj, actions, self.get_terminal_reward(traj[-1])


# ============================================================================
# DB-β REPLAY
# ============================================================================

class DBReplayAgent:
    def __init__(self, env, lr=0.05, epsilon=0.5,
                 buffer_size=10000, batch_size=128, tau=0.005,
                 beta_init=1.0, beta_rate=0.008, beta_max=25.0,
                 seed=42, name='DB-Replay'):
        self.env=env; self.lr=lr; self.epsilon=epsilon
        self.tau=tau; self.name=name; self.batch_size=batch_size
        self.beta=beta_init; self.beta_rate=beta_rate; self.beta_max=beta_max
        self.best_reward=1e-10
        self.rng=random.Random(seed); self.np_rng=np.random.RandomState(seed)

        self.log_flow = {s:0.0 for s in env.states}
        self.log_flow_target = {s:0.0 for s in env.states}
        self.pf_logits = {}
        for s in env.states:
            for a in env.valid_actions.get(s,[]): self.pf_logits[(s,a)]=0.0
        self.pb_logits = {}
        for s in env.states:
            for _,sp in env.parents[s]: self.pb_logits[(s,sp)]=0.0

        self.buffer=deque(maxlen=buffer_size); self.priorities=deque(maxlen=buffer_size)

    def _log_softmax(self, d):
        if not d: return {}
        mx=max(d.values()); ls=mx+np.log(sum(np.exp(v-mx) for v in d.values()))
        return {k:v-ls for k,v in d.items()}

    def _softmax(self, d):
        if not d: return {}
        mx=max(d.values()); e={k:np.exp(v-mx) for k,v in d.items()}; t=sum(e.values())
        return {k:v/t for k,v in e.items()}

    def get_pf_lp(self, s):
        valid=self.env.valid_actions.get(s,[])
        if not valid: return {}
        return self._log_softmax({a:self.pf_logits.get((s,a),0.0) for a in valid})

    def get_pb_lp(self, sp):
        parents=self.env.parents[sp]
        if not parents: return {}
        return self._log_softmax({s_par:self.pb_logits.get((sp,s_par),0.0) for _,s_par in parents})

    def policy(self, s, va):
        lp=self.get_pf_lp(s)
        if not lp: return va[0] if va else None
        return max(lp,key=lp.get)

    def sample_action(self, s):
        valid=self.env.valid_actions.get(s,[])
        if not valid: return None
        if self.rng.random()<self.epsilon: return self.rng.choice(valid)
        probs=self._softmax({a:self.pf_logits.get((s,a),0.0) for a in valid})
        acts=list(probs.keys()); ps=[probs[a] for a in acts]
        return acts[np.searchsorted(np.cumsum(ps), self.rng.random())]

    def update_transition(self, s, a, sp, is_term, term_r):
        pf_lp=self.get_pf_lp(s); pb_lp=self.get_pb_lp(sp)
        log_f_s=self.log_flow[s]; log_pf=pf_lp.get(a,-10.0)
        if is_term:
            log_f_sp=self.beta*np.log(max(term_r,1e-10))
        else:
            log_f_sp=self.log_flow_target[sp]
        log_pb=pb_lp.get(s,-10.0)
        delta=(log_f_s+log_pf)-(log_f_sp+log_pb)
        g=2*delta*self.lr

        self.log_flow[s]-=np.clip(g,-5,5)
        for av in self.env.valid_actions.get(s,[]):
            pf_av=np.exp(pf_lp.get(av,-10.0))
            gv=(1.0-pf_av) if av==a else (-pf_av)
            self.pf_logits[(s,av)]-=np.clip(g*gv,-5,5)
        for _,sp2 in self.env.parents[sp]:
            pb_sp2=np.exp(pb_lp.get(sp2,-10.0))
            gv=(1.0-pb_sp2) if sp2==s else (-pb_sp2)
            self.pb_logits[(sp,sp2)]+=np.clip(g*gv,-5,5)
        return delta

    def train_episode(self):
        s=self.env.start
        for _ in range(80):
            if self.env.is_terminal(s): break
            valid=self.env.valid_actions.get(s,[])
            if not valid: break
            a=self.sample_action(s); ns,_,done=self.env.step(s,a)
            is_term=self.env.is_terminal(ns); term_r=self.env.get_terminal_reward(ns)
            self.buffer.append((s,a,ns,is_term,term_r)); self.priorities.append(1.0)
            if is_term: self.best_reward=max(self.best_reward,term_r)
            s=ns; 
            if done: break

        if len(self.buffer)>=self.batch_size:
            p=np.array(self.priorities); p=p/p.sum()
            idx=self.np_rng.choice(len(self.buffer),self.batch_size,p=p,replace=False)
            for i in idx:
                s2,a2,sp2,it2,tr2=self.buffer[i]
                delta=self.update_transition(s2,a2,sp2,it2,tr2)
                self.priorities[i]=abs(delta)+1e-6
            for s3 in self.env.states:
                self.log_flow_target[s3]=(1-self.tau)*self.log_flow_target[s3]+self.tau*self.log_flow[s3]

        self.beta=min(self.beta+self.beta_rate,self.beta_max)
        return self.env.get_terminal_reward(s)

    def evaluate(self, n=100):
        return np.mean([self.env.sample_trajectory(self.policy,0.0)[2] for _ in range(n)])


# ============================================================================
# DQN-PER
# ============================================================================

class DQNPERAgent:
    def __init__(self, env, lr=0.1, gamma=0.99, epsilon=0.5,
                 buffer_size=10000, batch_size=128, tau=0.005,
                 seed=42, name='DQN-PER'):
        self.env=env; self.lr=lr; self.gamma=gamma; self.epsilon=epsilon
        self.tau=tau; self.batch_size=batch_size; self.name=name
        self.Q=defaultdict(lambda:defaultdict(float))
        self.Q_target=defaultdict(lambda:defaultdict(float))
        self.buffer=deque(maxlen=buffer_size); self.priorities=deque(maxlen=buffer_size)
        self.rng=random.Random(seed); self.np_rng=np.random.RandomState(seed)

    def policy(self, s, va):
        if not va: return None
        return va[np.argmax([self.Q[s][a] for a in va])]

    def train_episode(self):
        s=self.env.start
        for _ in range(80):
            if self.env.is_terminal(s): break
            valid=self.env.valid_actions.get(s,[])
            if not valid: break
            a=self.rng.choice(valid) if self.rng.random()<self.epsilon else self.policy(s,valid)
            ns,r,done=self.env.step(s,a)
            self.buffer.append((s,a,r,ns,done)); self.priorities.append(1.0); s=ns
        if len(self.buffer)<self.batch_size: return self.env.get_terminal_reward(s)
        p=np.array(self.priorities); p=p/p.sum()
        idx=self.np_rng.choice(len(self.buffer),self.batch_size,p=p,replace=False)
        for i in idx:
            s2,a2,r2,ns2,done2=self.buffer[i]
            if done2 or not self.env.valid_actions.get(ns2): target=r2
            else:
                nv=self.env.valid_actions[ns2]
                best=max(nv,key=lambda x:self.Q[ns2][x])
                target=r2+self.gamma*self.Q_target[ns2][best]
            td=target-self.Q[s2][a2]; self.Q[s2][a2]+=self.lr*td
            self.priorities[i]=abs(td)+1e-6
        for s3 in self.env.states:
            for a3 in self.env.valid_actions.get(s3,[]):
                self.Q_target[s3][a3]=(1-self.tau)*self.Q_target[s3][a3]+self.tau*self.Q[s3][a3]
        return self.env.get_terminal_reward(s)

    def evaluate(self, n=100):
        return np.mean([self.env.sample_trajectory(self.policy,0.0)[2] for _ in range(n)])


# ============================================================================
# EXPERIMENT
# ============================================================================

def run(n_seeds=5):
    TOTAL_TRAJS = 150_000
    EVAL_EVERY = 1000

    print("="*75)
    print("32×32 SUPER HARD DECEPTIVE GRID")
    print("="*75)

    # Build env once to show stats
    test_env = SuperHardGrid(seed=0)
    print(f"\n  Budget: {TOTAL_TRAJS:,} trajectories per agent")
    print(f"  Seeds: {n_seeds}")
    print(f"  Agents: DB-Replay, DQN-PER\n")

    all_results = {
        'DB-Replay': {'eval_all':[], 'trajs':None},
        'DQN-PER':   {'eval_all':[], 'trajs':None},
    }

    for seed in range(n_seeds):
        print(f"--- Seed {seed+1}/{n_seeds} ---")
        env = SuperHardGrid(seed=0)  # same env layout, different agent RNG

        agents = {
            'DB-Replay': DBReplayAgent(env, lr=0.05, epsilon=0.5,
                                        buffer_size=10000, batch_size=128, tau=0.005,
                                        beta_init=1.0, beta_rate=0.008, beta_max=25.0,
                                        seed=seed*100+42),
            'DQN-PER':   DQNPERAgent(env, lr=0.1, gamma=0.99, epsilon=0.5,
                                      buffer_size=10000, batch_size=128, tau=0.005,
                                      seed=seed*100+42),
        }

        results = {n: {'trajs':[], 'evals':[]} for n in agents}

        for ep in range(1, TOTAL_TRAJS+1):
            # Epsilon decay
            if ep % (TOTAL_TRAJS//8) == 0:
                for agent in agents.values():
                    agent.epsilon = max(0.05, agent.epsilon * 0.75)

            for name, agent in agents.items():
                agent.train_episode()

            if ep % EVAL_EVERY == 0:
                for name, agent in agents.items():
                    er = agent.evaluate(200)
                    results[name]['trajs'].append(ep)
                    results[name]['evals'].append(er)

                if ep % 40000 == 0:
                    line = f"  trajs={ep:>7,} | "
                    for name in agents:
                        line += f"{name}={results[name]['evals'][-1]:.3f} "
                    print(line)

        for name in agents:
            all_results[name]['eval_all'].append(results[name]['evals'])
            all_results[name]['trajs'] = results[name]['trajs']

    # Summary
    print(f"\n{'='*75}")
    print("RESULTS — 32×32 Super Hard Deceptive Grid")
    print(f"{'='*75}")

    print(f"\n{'Agent':<14} | {'Final':>12} | {'Trajs→Goal':>14} | {'Seeds OK':>10}")
    print("-"*60)

    for name in all_results:
        d = all_results[name]
        arr = np.array(d['eval_all'])
        fm = arr[:,-1].mean(); fs = arr[:,-1].std()
        trajs = d['trajs']

        first_goals = []
        for se in d['eval_all']:
            found = None
            for i,r in enumerate(se):
                if r >= 0.9: found = trajs[i]; break
            first_goals.append(found if found else float('inf'))

        n_ok = sum(1 for e in first_goals if e < float('inf'))
        avg = np.mean([e for e in first_goals if e<float('inf')]) if n_ok else float('inf')
        avg_str = f"{avg:,.0f}" if n_ok else "never"

        print(f"{name:<14} | {fm:.3f}±{fs:.3f} | {avg_str:>14} | {n_ok}/{n_seeds}")

    # Per-seed detail
    print(f"\nPer-seed first goal (trajectories):")
    for name in all_results:
        d = all_results[name]
        trajs = d['trajs']
        fgs = []
        for se in d['eval_all']:
            found = "never"
            for i,r in enumerate(se):
                if r >= 0.9: found = f"{trajs[i]:,}"; break
            fgs.append(found)
        print(f"  {name}: {' | '.join(fgs)}")

    return all_results


def plot(all_results):
    colors = {'DB-Replay':'#4CAF50', 'DQN-PER':'#FF5722'}

    fig, axes = plt.subplots(1, 3, figsize=(17, 5.5))
    fig.suptitle('DB-β vs DQN-PER — 32×32 Super Hard Deceptive Grid\n'
                 'Traps: R=0.05→0.50 (easy to reach)  |  Goal: R=1.0 (hard to reach)  |  200k budget',
                 fontsize=13, fontweight='bold')

    # 1: Convergence with std bands
    ax = axes[0]
    for name, d in all_results.items():
        arr = np.array(d['eval_all'])
        trajs = d['trajs']
        m = arr.mean(0); s = arr.std(0)
        ax.plot(trajs, m, label=name, color=colors[name], linewidth=2.5)
        ax.fill_between(trajs, m-s, np.minimum(m+s, 1.05), color=colors[name], alpha=0.15)
    ax.axhline(1.0, color='gray', ls='--', alpha=0.3)
    ax.axhline(0.50, color='red', ls=':', alpha=0.4, label='Best trap (R≈0.50)')
    ax.set_xlabel('Total Trajectories', fontsize=11)
    ax.set_ylabel('Eval Reward', fontsize=11)
    ax.set_title('Convergence (mean ± std)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.15)

    # 2: Reliability
    ax = axes[1]
    for name, d in all_results.items():
        arr = np.array(d['eval_all'])
        sr = (arr >= 0.9).mean(0)
        ax.plot(d['trajs'], sr, label=name, color=colors[name], linewidth=2.5)
    ax.set_xlabel('Total Trajectories', fontsize=11)
    ax.set_ylabel('Fraction of Seeds at Goal', fontsize=11)
    ax.set_title('Reliability (R≥0.9)')
    ax.legend(fontsize=9); ax.grid(True, alpha=0.3); ax.set_ylim(-0.05, 1.15)

    # 3: Speed boxplot
    ax = axes[2]
    all_fg = {}
    for name, d in all_results.items():
        trajs = d['trajs']
        fgs = []
        for se in d['eval_all']:
            found = None
            for i,r in enumerate(se):
                if r>=0.9: found=trajs[i]; break
            fgs.append(found if found else 250000)
        all_fg[name] = fgs

    bp = ax.boxplot([all_fg[n] for n in all_results], patch_artist=True, widths=0.5)
    for patch, name in zip(bp['boxes'], all_results):
        patch.set_facecolor(colors[name]); patch.set_alpha(0.7)
    ax.set_xticklabels(list(all_results.keys()), fontsize=11)
    ax.set_ylabel('Trajectories to Goal', fontsize=11)
    ax.set_title('Sample Efficiency')
    ax.grid(True, alpha=0.3, axis='y')

    # Annotate medians
    for i, name in enumerate(all_results):
        fgs = [f for f in all_fg[name] if f < 250000]
        if fgs:
            med = np.median(fgs)
            ax.text(i+1, med-8000, f'{med:,.0f}', ha='center', fontsize=10, fontweight='bold')

    plt.tight_layout()
    plt.savefig('./db_vs_dqn_32x32.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved db_vs_dqn_32x32.png")


if __name__ == '__main__':
    t0 = time.time()
    all_results = run(n_seeds=5)
    plot(all_results)
    print(f"\nTotal runtime: {time.time()-t0:.1f}s")