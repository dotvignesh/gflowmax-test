"""
GFlowMax v3 + DQN comparison
==============================
All GFlowMax variants: 3000 steps
Q-Learning + DQN: 6000 steps (2x advantage)

DQN: tabular but with experience replay + target network (soft update).
This isolates whether replay alone (without SubTB) can escape the trap.
"""

import numpy as np
import random
from collections import defaultdict, deque
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import time


# ============================================================================
# ENVIRONMENT
# ============================================================================

class HardDAGGridWorld:
    def __init__(self):
        self.rows = 8
        self.cols = 8
        self.start = (0, 0)
        self.goal = (7, 7)
        self.holes = {(1,1),(1,6),(2,2),(3,4),(4,6),(5,3),(6,5)}
        self.traps = {(1,3):0.4, (2,5):0.6, (4,1):0.3}
        self.edge_terminals = {(0,7):0.2, (7,0):0.15, (3,7):0.35, (7,3):0.25}
        self.n_actions = 2
        self.states = [(r,c) for r in range(self.rows) for c in range(self.cols)]
        self.n_states = len(self.states)

        self.terminal_states = set()
        self.terminal_rewards = {}
        self.terminal_states.add(self.goal)
        self.terminal_rewards[self.goal] = 1.0
        for h in self.holes:
            self.terminal_states.add(h); self.terminal_rewards[h] = 0.01
        for t,r in self.traps.items():
            self.terminal_states.add(t); self.terminal_rewards[t] = r
        for t,r in self.edge_terminals.items():
            self.terminal_states.add(t); self.terminal_rewards[t] = r

        self.valid_actions = {}
        self.children = defaultdict(list)
        self.parents = defaultdict(list)
        for r in range(self.rows):
            for c in range(self.cols):
                s = (r,c)
                if s in self.terminal_states:
                    self.valid_actions[s] = []; continue
                actions = []
                if c+1 < self.cols:
                    actions.append(0); sn=(r,c+1)
                    self.children[s].append((0,sn)); self.parents[sn].append((0,s))
                if r+1 < self.rows:
                    actions.append(1); sn=(r+1,c)
                    self.children[s].append((1,sn)); self.parents[sn].append((1,s))
                self.valid_actions[s] = actions

    def step(self, state, action):
        r,c = state
        ns = (r,c+1) if action==0 else (r+1,c)
        return ns, self.terminal_rewards.get(ns,0.0), ns in self.terminal_states

    def is_terminal(self, s): return s in self.terminal_states
    def get_terminal_reward(self, s): return self.terminal_rewards.get(s, 0.0)

    def sample_trajectory(self, policy_fn, epsilon=0.1, rng=None):
        if rng is None: rng = random
        s = self.start; traj=[s]; actions=[]
        for _ in range(20):
            if self.is_terminal(s): break
            valid = self.valid_actions.get(s,[])
            if not valid: break
            if rng.random() < epsilon:
                a = rng.choice(valid)
            else:
                a = policy_fn(s, valid)
            ns,_,done = self.step(s,a)
            actions.append(a); traj.append(ns); s=ns
            if done: break
        return traj, actions, self.get_terminal_reward(traj[-1])


# ============================================================================
# TABULAR PARAMS (for GFlowNet agents)
# ============================================================================

class TabularLogParams:
    def __init__(self, env):
        self.env = env
        self.log_pf = {}
        for s in env.states:
            valid = env.valid_actions.get(s,[])
            if valid:
                lp = -np.log(len(valid))
                for a in valid: self.log_pf[(s,a)] = lp
        self.log_pb = {}
        for sp in env.states:
            parents = env.parents[sp]
            if parents:
                lp = -np.log(len(parents))
                for a,s_par in parents: self.log_pb[(sp,s_par)] = lp
        self.log_flow = {s:0.0 for s in env.states}
        self.log_Z = 0.0

    def _normalize(self, raw):
        mx = max(raw.values())
        ls = mx + np.log(sum(np.exp(v-mx) for v in raw.values()))
        return {k: v-ls for k,v in raw.items()}

    def get_forward_log_probs(self, state):
        valid = self.env.valid_actions.get(state,[])
        if not valid: return {}
        return self._normalize({a: self.log_pf.get((state,a),-10.0) for a in valid})

    def get_backward_log_probs(self, state):
        parents = self.env.parents[state]
        if not parents: return {}
        return self._normalize({sp: self.log_pb.get((state,sp),-10.0) for _,sp in parents})

    def sample_forward(self, state, epsilon=0.1, rng=None):
        if rng is None: rng = random
        valid = self.env.valid_actions.get(state,[])
        if not valid: return None
        if rng.random() < epsilon: return rng.choice(valid)
        lp = self.get_forward_log_probs(state)
        probs = {a: np.exp(v) for a,v in lp.items()}
        t = sum(probs.values()); probs = {a:p/t for a,p in probs.items()}
        acts = list(probs.keys()); ps = [probs[a] for a in acts]
        return acts[np.searchsorted(np.cumsum(ps), rng.random())]


# ============================================================================
# Q-LEARNING (no replay)
# ============================================================================

class QLearningAgent:
    def __init__(self, env, lr=0.15, gamma=0.99, epsilon=0.4, seed=42):
        self.env = env; self.lr = lr; self.gamma = gamma; self.epsilon = epsilon
        self.Q = defaultdict(lambda: defaultdict(float))
        self.rng = random.Random(seed)
        self.name = "Q-Learning"

    def policy(self, state, valid_actions):
        if not valid_actions: return None
        return valid_actions[np.argmax([self.Q[state][a] for a in valid_actions])]

    def train_episode(self):
        state = self.env.start
        for _ in range(20):
            if self.env.is_terminal(state): break
            valid = self.env.valid_actions.get(state,[])
            if not valid: break
            a = self.rng.choice(valid) if self.rng.random()<self.epsilon else self.policy(state,valid)
            ns, reward, done = self.env.step(state, a)
            if done or not self.env.valid_actions.get(ns):
                target = reward
            else:
                nv = self.env.valid_actions[ns]
                target = reward + self.gamma * max(self.Q[ns][a2] for a2 in nv) if nv else reward
            self.Q[state][a] += self.lr * (target - self.Q[state][a])
            state = ns
        return self.env.get_terminal_reward(state)

    def evaluate(self, n=100):
        return np.mean([self.env.sample_trajectory(self.policy,0.0)[2] for _ in range(n)])


# ============================================================================
# DQN (tabular with replay buffer + target network)
# ============================================================================

class DQNAgent:
    """
    Tabular DQN with:
      - Experience replay buffer (uniform or prioritized)
      - Target Q-network (soft-updated)
      - Configurable batch size

    This is the strongest standard RL baseline for this setting:
    same replay mechanism that made GFlowMax-Replay strong,
    but with standard Bellman updates instead of SubTB.
    """

    def __init__(self, env, lr=0.1, gamma=0.99, epsilon=0.4,
                 buffer_size=500, batch_size=16,
                 tau=0.01,                      # soft target update rate
                 prioritized=False,             # prioritized experience replay
                 seed=42, name="DQN"):
        self.env = env
        self.lr = lr; self.gamma = gamma; self.epsilon = epsilon
        self.tau = tau
        self.prioritized = prioritized
        self.batch_size = batch_size
        self.name = name

        # Online Q
        self.Q = defaultdict(lambda: defaultdict(float))
        # Target Q
        self.Q_target = defaultdict(lambda: defaultdict(float))

        # Replay buffer: stores (s, a, r, s', done)
        self.buffer = deque(maxlen=buffer_size)

        # Per-transition priority (for PER)
        self.priorities = deque(maxlen=buffer_size)

        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def policy(self, state, valid_actions):
        if not valid_actions: return None
        return valid_actions[np.argmax([self.Q[state][a] for a in valid_actions])]

    def _soft_update_target(self):
        """Q_target = (1-τ)Q_target + τ Q"""
        for s in self.env.states:
            for a in self.env.valid_actions.get(s, []):
                self.Q_target[s][a] = (1 - self.tau) * self.Q_target[s][a] + self.tau * self.Q[s][a]

    def train_episode(self):
        state = self.env.start

        # Collect full trajectory, store each transition
        for _ in range(20):
            if self.env.is_terminal(state): break
            valid = self.env.valid_actions.get(state, [])
            if not valid: break

            a = self.rng.choice(valid) if self.rng.random() < self.epsilon else self.policy(state, valid)
            ns, reward, done = self.env.step(state, a)

            self.buffer.append((state, a, reward, ns, done))
            self.priorities.append(1.0)  # max priority for new transitions

            state = ns

        # Skip training if buffer too small
        if len(self.buffer) < self.batch_size:
            return self.env.get_terminal_reward(state)

        # Sample minibatch
        if self.prioritized:
            priors = np.array(self.priorities)
            probs = priors / priors.sum()
            indices = self.np_rng.choice(len(self.buffer), self.batch_size, p=probs, replace=False)
        else:
            indices = self.np_rng.choice(len(self.buffer), self.batch_size, replace=False)

        # Bellman updates from minibatch
        for idx in indices:
            s, a, r, ns, done = self.buffer[idx]

            if done or not self.env.valid_actions.get(ns):
                target = r
            else:
                nv = self.env.valid_actions[ns]
                # Double DQN: select action with online Q, evaluate with target Q
                best_a = max(nv, key=lambda a2: self.Q[ns][a2])
                target = r + self.gamma * self.Q_target[ns][best_a]

            td_error = target - self.Q[s][a]
            self.Q[s][a] += self.lr * td_error

            # Update priority
            if self.prioritized:
                self.priorities[idx] = abs(td_error) + 1e-6

        # Soft update target network
        self._soft_update_target()

        return self.env.get_terminal_reward(state)

    def evaluate(self, n=100):
        return np.mean([self.env.sample_trajectory(self.policy, 0.0)[2] for _ in range(n)])


# ============================================================================
# GFLOWNET-TB (β=1)
# ============================================================================

class GFlowNetTBAgent:
    def __init__(self, env, lr=0.01, epsilon=0.4, seed=42):
        self.env = env; self.lr = lr; self.epsilon = epsilon
        self.params = TabularLogParams(env)
        self.rng = random.Random(seed)
        self.name = "GFlowNet-TB"

    def policy(self, state, valid_actions):
        lp = self.params.get_forward_log_probs(state)
        if not lp: return valid_actions[0] if valid_actions else None
        return max(lp, key=lp.get)

    def train_episode(self):
        traj, actions, reward = self.env.sample_trajectory(
            lambda s,v: self.params.sample_forward(s, self.epsilon, self.rng),
            epsilon=0.0, rng=self.rng)
        if not actions: return reward
        reward = max(reward, 1e-10)
        log_pf_sum=0; log_pb_sum=0
        for t in range(len(actions)):
            s,sn,a = traj[t],traj[t+1],actions[t]
            log_pf_sum += self.params.get_forward_log_probs(s).get(a,-10.0)
            log_pb_sum += self.params.get_backward_log_probs(sn).get(s,-10.0)
        delta = self.params.log_Z + log_pf_sum - np.log(reward) - log_pb_sum
        grad = 2*delta*self.lr
        self.params.log_Z -= grad
        for t in range(len(actions)):
            s,sn,a = traj[t],traj[t+1],actions[t]
            fwd=self.params.get_forward_log_probs(s); bwd=self.params.get_backward_log_probs(sn)
            for av in self.env.valid_actions.get(s,[]):
                g=(1.0-np.exp(fwd[av])) if av==a else (-np.exp(fwd[av]))
                self.params.log_pf[(s,av)] -= grad*g
            for _,sp in self.env.parents[sn]:
                g=(1.0-np.exp(bwd[sp])) if sp==s else (-np.exp(bwd[sp]))
                self.params.log_pb[(sn,sp)] += grad*g
        return reward

    def evaluate(self, n=100):
        return np.mean([self.env.sample_trajectory(self.policy,0.0)[2] for _ in range(n)])


# ============================================================================
# GFLOWMAX ON-POLICY (softmax)
# ============================================================================

class GFlowMaxOnPolicy:
    def __init__(self, env, lr=0.005, epsilon=0.4,
                 beta_init=1.0, beta_rate=0.015, beta_max=20.0,
                 subtb_lambda=0.8, alpha_fb=0.1, gamma_z=0.05,
                 onpolicy_batch_N=16, weight_temp=5.0,
                 seed=42, name='GFlowMax-OnPolicy'):
        self.env = env; self.lr = lr; self.epsilon = epsilon
        self.params = TabularLogParams(env)
        self.beta = beta_init; self.beta_rate = beta_rate; self.beta_max = beta_max
        self.subtb_lambda = subtb_lambda
        self.alpha_fb = alpha_fb; self.gamma_z = gamma_z
        self.onpolicy_batch_N = onpolicy_batch_N
        self.weight_temp = weight_temp
        self.name = name; self.best_reward = 1e-10
        self.rng = random.Random(seed)

    def policy(self, state, valid_actions):
        lp = self.params.get_forward_log_probs(state)
        if not lp: return valid_actions[0] if valid_actions else None
        return max(lp, key=lp.get)

    def compute_subtb_grads(self, traj, actions, reward):
        reward = max(reward, 1e-10); n = len(actions)
        if n == 0: return 0.0, {}
        log_pf_list, log_pb_list = [], []
        for t in range(n):
            s,sn,a = traj[t],traj[t+1],actions[t]
            log_pf_list.append(self.params.get_forward_log_probs(s).get(a,-10.0))
            log_pb_list.append(self.params.get_backward_log_probs(sn).get(s,-10.0))
        cum_pf=[0.0]; cum_pb=[0.0]
        for t in range(n):
            cum_pf.append(cum_pf[-1]+log_pf_list[t])
            cum_pb.append(cum_pb[-1]+log_pb_list[t])
        lf_term = self.beta * np.log(reward)
        total_loss=0.0; grads=defaultdict(float); count=0
        for i in range(n+1):
            for j in range(i+1,n+1):
                w = self.subtb_lambda**(j-i)
                lfi = self.params.log_flow[traj[i]] if i<n else lf_term
                lfj = self.params.log_flow[traj[j]] if j<n else lf_term
                delta = lfi+(cum_pf[j]-cum_pf[i])-lfj-(cum_pb[j]-cum_pb[i])
                total_loss += w*delta**2; count+=1; g = 2*w*delta
                if i<n: grads[('flow',traj[i])] += g
                if j<n: grads[('flow',traj[j])] -= g
                for t in range(i,j):
                    s,sn,a = traj[t],traj[t+1],actions[t]
                    fwd=self.params.get_forward_log_probs(s)
                    bwd=self.params.get_backward_log_probs(sn)
                    for av in self.env.valid_actions.get(s,[]):
                        gv=(1.0-np.exp(fwd[av])) if av==a else (-np.exp(fwd[av]))
                        grads[('pf',s,av)] += g*gv
                    for _,sp in self.env.parents[sn]:
                        gv=(1.0-np.exp(bwd[sp])) if sp==s else (-np.exp(bwd[sp]))
                        grads[('pb',sn,sp)] -= g*gv
        if count>0:
            total_loss/=count
            for k in grads: grads[k]/=count
        return total_loss, grads

    def compute_fb_grads(self, traj, actions):
        n=len(actions); loss=0.0; grads=defaultdict(float)
        for t in range(n):
            s,sn,a = traj[t],traj[t+1],actions[t]
            fwd=self.params.get_forward_log_probs(s); bwd=self.params.get_backward_log_probs(sn)
            delta = fwd.get(a,-10.0)+bwd.get(s,-10.0); loss += delta**2; g=2*delta
            for av in self.env.valid_actions.get(s,[]):
                gv=(1.0-np.exp(fwd[av])) if av==a else (-np.exp(fwd[av]))
                grads[('pf',s,av)] += g*gv
            for _,sp in self.env.parents[sn]:
                gv=(1.0-np.exp(bwd[sp])) if sp==s else (-np.exp(bwd[sp]))
                grads[('pb',sn,sp)] += g*gv
        if n>0: loss/=n
        for k in grads: grads[k]/=n
        return loss, grads

    def apply_grads(self, grads, lr):
        for key,grad in grads.items():
            g = np.clip(grad,-5.0,5.0)
            if key[0]=='flow': self.params.log_flow[key[1]] -= lr*g
            elif key[0]=='pf': self.params.log_pf[(key[1],key[2])] -= lr*g
            elif key[0]=='pb': self.params.log_pb[(key[1],key[2])] -= lr*g
            elif key==('log_Z',): self.params.log_Z -= lr*g

    def update_beta(self):
        if self.best_reward <= 0: return
        target = self.beta*np.log(max(self.best_reward,1e-10))
        gap = abs(self.params.log_Z - target)
        sig = 1.0/(1.0+np.exp(-(-gap/max(abs(target),1.0)+1.0)*3))
        self.beta = min(self.beta + self.beta_rate*sig, self.beta_max)

    def train_episode(self):
        batch = []
        for _ in range(self.onpolicy_batch_N):
            traj, actions, reward = self.env.sample_trajectory(
                lambda s,v: self.params.sample_forward(s, self.epsilon, self.rng),
                epsilon=0.0, rng=self.rng)
            if actions:
                batch.append((traj, actions, reward))
                self.best_reward = max(self.best_reward, reward)
        if not batch: return 0.0

        rewards = [max(r,1e-10) for _,_,r in batch]
        log_r = [self.weight_temp * np.log(r) for r in rewards]
        mx = max(log_r); exp_r = [np.exp(l-mx) for l in log_r]
        total = sum(exp_r); weights = [e/total for e in exp_r]

        all_grads = defaultdict(float)
        for (bt,ba,br), w in zip(batch, weights):
            if w < 1e-12: continue
            _, sg = self.compute_subtb_grads(bt, ba, br)
            for k,v in sg.items(): all_grads[k] += w*v
            if self.alpha_fb > 0:
                _, fg = self.compute_fb_grads(bt, ba)
                for k,v in fg.items(): all_grads[k] += self.alpha_fb*w*v

        if self.best_reward > 0:
            target = self.beta*np.log(max(self.best_reward,1e-10))
            all_grads[('log_Z',)] += self.gamma_z*2*(self.params.log_Z - target)

        self.apply_grads(all_grads, self.lr)
        self.update_beta()
        return np.mean([r for _,_,r in batch])

    def evaluate(self, n=100):
        return np.mean([self.env.sample_trajectory(self.policy,0.0)[2] for _ in range(n)])


# ============================================================================
# GFLOWMAX REPLAY (original v2)
# ============================================================================

class GFlowMaxReplay:
    def __init__(self, env, lr=0.005, epsilon=0.4,
                 beta_init=1.0, beta_rate=0.015, beta_max=20.0,
                 subtb_lambda=0.8, alpha_fb=0.1, gamma_z=0.05,
                 seed=42, name='GFlowMax-Replay'):
        self.env = env; self.lr = lr; self.epsilon = epsilon
        self.params = TabularLogParams(env)
        self.beta = beta_init; self.beta_rate = beta_rate; self.beta_max = beta_max
        self.subtb_lambda = subtb_lambda
        self.alpha_fb = alpha_fb; self.gamma_z = gamma_z
        self.name = name; self.best_reward = 1e-10
        self.buffer = []; self.buffer_size = 300
        self.rng = random.Random(seed)
        self.np_rng = np.random.RandomState(seed)

    def policy(self, state, valid_actions):
        lp = self.params.get_forward_log_probs(state)
        if not lp: return valid_actions[0] if valid_actions else None
        return max(lp, key=lp.get)

    def compute_subtb_grads(self, traj, actions, reward):
        reward = max(reward,1e-10); n=len(actions)
        if n==0: return 0.0,{}
        log_pf_list,log_pb_list=[],[]
        for t in range(n):
            s,sn,a=traj[t],traj[t+1],actions[t]
            log_pf_list.append(self.params.get_forward_log_probs(s).get(a,-10.0))
            log_pb_list.append(self.params.get_backward_log_probs(sn).get(s,-10.0))
        cum_pf=[0.0];cum_pb=[0.0]
        for t in range(n):cum_pf.append(cum_pf[-1]+log_pf_list[t]);cum_pb.append(cum_pb[-1]+log_pb_list[t])
        lf_term=self.beta*np.log(reward)
        total_loss=0.0;grads=defaultdict(float);count=0
        for i in range(n+1):
            for j in range(i+1,n+1):
                w=self.subtb_lambda**(j-i)
                lfi=self.params.log_flow[traj[i]] if i<n else lf_term
                lfj=self.params.log_flow[traj[j]] if j<n else lf_term
                delta=lfi+(cum_pf[j]-cum_pf[i])-lfj-(cum_pb[j]-cum_pb[i])
                total_loss+=w*delta**2;count+=1;g=2*w*delta
                if i<n:grads[('flow',traj[i])]+=g
                if j<n:grads[('flow',traj[j])]-=g
                for t in range(i,j):
                    s,sn,a=traj[t],traj[t+1],actions[t]
                    fwd=self.params.get_forward_log_probs(s);bwd=self.params.get_backward_log_probs(sn)
                    for av in self.env.valid_actions.get(s,[]):
                        gv=(1.0-np.exp(fwd[av])) if av==a else (-np.exp(fwd[av]))
                        grads[('pf',s,av)]+=g*gv
                    for _,sp in self.env.parents[sn]:
                        gv=(1.0-np.exp(bwd[sp])) if sp==s else (-np.exp(bwd[sp]))
                        grads[('pb',sn,sp)]-=g*gv
        if count>0:
            total_loss/=count
            for k in grads:grads[k]/=count
        return total_loss,grads

    def compute_fb_grads(self, traj, actions):
        n=len(actions);loss=0.0;grads=defaultdict(float)
        for t in range(n):
            s,sn,a=traj[t],traj[t+1],actions[t]
            fwd=self.params.get_forward_log_probs(s);bwd=self.params.get_backward_log_probs(sn)
            delta=fwd.get(a,-10.0)+bwd.get(s,-10.0);loss+=delta**2;g=2*delta
            for av in self.env.valid_actions.get(s,[]):
                gv=(1.0-np.exp(fwd[av])) if av==a else (-np.exp(fwd[av]))
                grads[('pf',s,av)]+=g*gv
            for _,sp in self.env.parents[sn]:
                gv=(1.0-np.exp(bwd[sp])) if sp==s else (-np.exp(bwd[sp]))
                grads[('pb',sn,sp)]+=g*gv
        if n>0:loss/=n
        for k in grads:grads[k]/=n
        return loss,grads

    def apply_grads(self, grads, lr):
        for key,grad in grads.items():
            g=np.clip(grad,-5.0,5.0)
            if key[0]=='flow':self.params.log_flow[key[1]]-=lr*g
            elif key[0]=='pf':self.params.log_pf[(key[1],key[2])]-=lr*g
            elif key[0]=='pb':self.params.log_pb[(key[1],key[2])]-=lr*g
            elif key==('log_Z',):self.params.log_Z-=lr*g

    def update_beta(self):
        if self.best_reward<=0:return
        target=self.beta*np.log(max(self.best_reward,1e-10))
        gap=abs(self.params.log_Z-target)
        sig=1.0/(1.0+np.exp(-(-gap/max(abs(target),1.0)+1.0)*3))
        self.beta=min(self.beta+self.beta_rate*sig,self.beta_max)

    def train_episode(self):
        traj,actions,reward=self.env.sample_trajectory(
            lambda s,v:self.params.sample_forward(s,self.epsilon,self.rng),
            epsilon=0.0,rng=self.rng)
        if not actions:return reward
        self.best_reward=max(self.best_reward,reward)
        self.buffer.append((traj,actions,reward))
        if len(self.buffer)>self.buffer_size:self.buffer.pop(0)
        batch_size=min(12,len(self.buffer))
        weights=np.array([r**2+0.01 for _,_,r in self.buffer]);weights/=weights.sum()
        indices=self.np_rng.choice(len(self.buffer),batch_size,p=weights,replace=False)
        batch=[self.buffer[i] for i in indices]
        all_grads=defaultdict(float)
        for bt,ba,br in batch:
            _,sg=self.compute_subtb_grads(bt,ba,br)
            for k,v in sg.items():all_grads[k]+=v/batch_size
            if self.alpha_fb>0:
                _,fg=self.compute_fb_grads(bt,ba)
                for k,v in fg.items():all_grads[k]+=self.alpha_fb*v/batch_size
        if self.best_reward>0:
            target=self.beta*np.log(max(self.best_reward,1e-10))
            all_grads[('log_Z',)]+=self.gamma_z*2*(self.params.log_Z-target)
        self.apply_grads(all_grads,self.lr);self.update_beta()
        return reward

    def evaluate(self, n=100):
        return np.mean([self.env.sample_trajectory(self.policy,0.0)[2] for _ in range(n)])


# ============================================================================
# EXPERIMENT
# ============================================================================

def run_experiment():
    seed = 42
    env = HardDAGGridWorld()

    STEPS_GFLOWMAX = 3000
    STEPS_RL = 6000  # 2x advantage for RL baselines
    EVAL_INTERVAL = 100
    N_EVAL = 200

    # All agents
    agents = {
        'Q-Learning (6k)':       QLearningAgent(env, seed=seed),
        'DQN (6k)':              DQNAgent(env, lr=0.1, gamma=0.99, epsilon=0.4,
                                          buffer_size=500, batch_size=16,
                                          tau=0.01, prioritized=False, seed=seed,
                                          name='DQN'),
        'DQN-PER (6k)':          DQNAgent(env, lr=0.1, gamma=0.99, epsilon=0.4,
                                          buffer_size=500, batch_size=16,
                                          tau=0.01, prioritized=True, seed=seed,
                                          name='DQN-PER'),
        'GFlowNet-TB (3k)':     GFlowNetTBAgent(env, seed=seed),
        'GFlowMax-Replay (3k)': GFlowMaxReplay(env, seed=seed),
        'GFlowMax-OnPol (3k)':  GFlowMaxOnPolicy(env, seed=seed),
    }

    max_steps = {
        'Q-Learning (6k)': STEPS_RL,
        'DQN (6k)': STEPS_RL,
        'DQN-PER (6k)': STEPS_RL,
        'GFlowNet-TB (3k)': STEPS_GFLOWMAX,
        'GFlowMax-Replay (3k)': STEPS_GFLOWMAX,
        'GFlowMax-OnPol (3k)': STEPS_GFLOWMAX,
    }

    results = {name: {'episodes':[], 'evals':[]} for name in agents}

    print("="*75)
    print("GFlowMax vs RL Baselines (RL gets 2x steps)")
    print("="*75)
    print(f"RL baselines (Q-Learning, DQN, DQN-PER): {STEPS_RL} steps")
    print(f"GFlowMax variants + GFlowNet-TB: {STEPS_GFLOWMAX} steps")
    print(f"DQN batch_size=16, buffer=500, soft target update τ=0.01")
    print(f"DQN-PER: same + prioritized experience replay")
    print()

    all_max = max(max_steps.values())

    for ep in range(1, all_max + 1):
        # Epsilon decay
        if ep % 500 == 0:
            for agent in agents.values():
                if hasattr(agent, 'epsilon'):
                    agent.epsilon = max(0.05, agent.epsilon * 0.7)

        for name, agent in agents.items():
            if ep <= max_steps[name]:
                agent.train_episode()

        if ep % EVAL_INTERVAL == 0:
            line = f"  Step {ep:5d} | "
            for name, agent in agents.items():
                if ep <= max_steps[name] or ep == all_max:
                    er = agent.evaluate(N_EVAL)
                    results[name]['episodes'].append(ep)
                    results[name]['evals'].append(er)
                    short = name.split(' ')[0]
                    line += f"{short}:{er:.2f} "
            if ep % 1000 == 0:
                print(line)

    # Summary
    print(f"\n{'='*75}")
    print("FINAL RESULTS")
    print(f"{'='*75}")
    print(f"\n{'Agent':<25} | {'Steps':>6} | {'Final Eval':>11} | {'Steps→Goal':>11}")
    print("-"*65)

    for name in agents:
        d = results[name]
        if not d['evals']:
            continue
        final = d['evals'][-1]
        fg = "never"
        for i, r in enumerate(d['evals']):
            if r >= 0.9:
                fg = str(d['episodes'][i])
                break
        budget = max_steps[name]
        print(f"{name:<25} | {budget:>6} | {final:>11.3f} | {fg:>11}")

    return results, agents, max_steps


def plot_results(results, max_steps):
    colors = {
        'Q-Learning (6k)':       '#F44336',
        'DQN (6k)':              '#E91E63',
        'DQN-PER (6k)':          '#FF5722',
        'GFlowNet-TB (3k)':     '#FF9800',
        'GFlowMax-Replay (3k)': '#4CAF50',
        'GFlowMax-OnPol (3k)':  '#2196F3',
    }
    linestyles = {
        'Q-Learning (6k)':       '--',
        'DQN (6k)':              '--',
        'DQN-PER (6k)':          '--',
        'GFlowNet-TB (3k)':     ':',
        'GFlowMax-Replay (3k)': '-',
        'GFlowMax-OnPol (3k)':  '-',
    }

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('GFlowMax vs RL Baselines\n'
                 'RL agents get 6000 steps, GFlowMax agents get 3000 steps',
                 fontsize=14, fontweight='bold')

    # Convergence
    ax = axes[0]
    for name, d in results.items():
        if not d['evals']: continue
        ax.plot(d['episodes'], d['evals'], label=name,
                color=colors.get(name,'gray'), linewidth=2.5,
                linestyle=linestyles.get(name,'-'))
    ax.axhline(1.0, color='gray', ls='--', alpha=0.3, label='Optimal')
    ax.axhline(0.6, color='orange', ls=':', alpha=0.3, label='Best trap')
    ax.axvline(3000, color='black', ls=':', alpha=0.2)
    ax.text(3050, 0.05, '← GFlowMax stops here', fontsize=8, alpha=0.5)
    ax.set_xlabel('Update Step', fontsize=12)
    ax.set_ylabel('Eval Reward', fontsize=12)
    ax.set_title('Convergence (RL gets 2x steps)')
    ax.legend(fontsize=8, loc='center right')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-0.05, 1.15)

    # Bar chart: steps to goal
    ax = axes[1]
    names = []
    steps_to_goal = []
    bar_colors = []
    hatches = []

    for name, d in results.items():
        if not d['evals']: continue
        fg = None
        for i, r in enumerate(d['evals']):
            if r >= 0.9:
                fg = d['episodes'][i]; break
        names.append(name.replace(' (6k)','*\n(6k steps)').replace(' (3k)','\n(3k steps)'))
        steps_to_goal.append(fg if fg else max_steps[name] + 500)
        bar_colors.append(colors.get(name, 'gray'))
        hatches.append('//' if fg is None else '')

    bars = ax.bar(range(len(names)), steps_to_goal, color=bar_colors, alpha=0.8,
                  edgecolor='black', linewidth=0.5)
    for bar, h, stg, ms in zip(bars, hatches, steps_to_goal, 
                                 [max_steps[n] for n in results if results[n]['evals']]):
        if h:
            bar.set_hatch(h)
            bar.set_alpha(0.4)

    ax.set_xticks(range(len(names)))
    ax.set_xticklabels(names, fontsize=7.5)
    ax.set_ylabel('Steps to Reach Goal (R≥0.9)', fontsize=11)
    ax.set_title('Convergence Speed\n(hatched = never reached goal)')
    ax.grid(True, alpha=0.3, axis='y')

    # Add value labels
    for i, (bar, stg) in enumerate(zip(bars, steps_to_goal)):
        label = str(stg) if stg <= 6500 else "NEVER"
        y = bar.get_height() + 50
        ax.text(bar.get_x() + bar.get_width()/2, y, label,
                ha='center', va='bottom', fontsize=9, fontweight='bold')

    plt.tight_layout()
    plt.savefig('./gflowmax_vs_dqn.png', dpi=150, bbox_inches='tight')
    plt.close()
    print("\nSaved gflowmax_vs_dqn.png")


if __name__ == '__main__':
    t0 = time.time()
    results, agents, max_steps = run_experiment()
    plot_results(results, max_steps)
    print(f"\nRuntime: {time.time()-t0:.1f}s")
