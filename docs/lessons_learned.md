# Lessons Learned: From MCTS-DIME to DIME-Planning

This document captures the key findings from the MCTS-DIME exploration project, which preceded
DIME-Planning. The goal is to prevent repeating the same bugs and architectural mistakes.

**Read this before implementing anything.**

---

## What MCTS-DIME Was

A companion project applying DIME (Gadgil et al., ICLR 2024) to Monte Carlo Tree Search.
The key design: replace the MCTS value network with two heads:
- `f(h_s)` — predictor: binary classifier, "is s on the optimal path?"
- `v(h_s)` — value network: estimates CMI gain from expanding s

where `h_s` is a **tree-conditioned memory** built by a learned backup network
(MCTSnets-style). Acquisition function: `a(s) = log f(h_s) + λ v(h_s) + c_ucb sqrt(log N_parent / N_child)`.

Experiments were run in Python on two benchmarks: a binary tree (Exp1) and DeepSea (Exp2).

---

## Experiment 1: CMI Diagnostic (SUCCESS)

**Setup**: Fixed binary tree of depth 6, one rewarded leaf. Train MCTS-DIME on the same tree
for 200 episodes. Check if `v(h_s)` tracks `Δ(s)` (the actual CMI proxy).

**Result**: After fixing a key bug (see below), Spearman ρ rose to **+0.93**.
100% success rate on the tree. The CMI signal was clearly learned.

**Key finding**: DIME Lemma 1 holds empirically — the value network converges to the CMI
proxy in a tree-search setting.

**Bug that had to be fixed**: The tree environment was re-seeded each episode
(`BinaryTreeEnv(seed=ep)`), changing the reward location every episode. The predictor
never got signal to accumulate. Fix: create ONE fixed environment before the loop.

**Takeaway for DIME-Planning**: The CMI diagnostic should be reproduced before running
full experiments. Compute Spearman ρ(v(s), Δ(s)) on a small domain (Blocksworld easy)
to confirm the value network is learning before scaling up.

---

## Experiment 2: DeepSea (FAILURE — fundamental, not fixable)

**Setup**: DeepSea grid (N×N), reward only at bottom-right corner. MCTS-DIME vs UCT.

**Result**: MCTS-DIME achieved **0% solve rate** across all configurations:
- Grid sizes: 4×4, 6×6, 8×8
- Simulations: 32, 64, 256 per step
- Many hyperparameter configurations
- The `f` values for root children converged to ~0.587/0.581 (essentially random)

**Root cause (architectural, not a bug)**:

MCTS-DIME uses tree-conditioned memories `h_s` built by a backup network.
These memories are **rebuilt from scratch at the start of every episode**.
The GNN/backup network weights are shared across episodes, but the memories themselves
are ephemeral — they carry no information between episodes.

This means:
- Episode 1: `f(h_s)` starts from noise for all nodes in the new tree
- Episode 1000: still starts from noise for all nodes in the new tree
- The network learns general "backup" computation, but the predictor output for any specific
  node is re-initialized every episode

In a sparse-reward environment (reward only at one corner of 8×8 grid), the predictor
needs to identify WHICH specific states are on the path to the reward. But since the
memories reset, it can only accumulate knowledge via the network weights — not via
the memory state. This creates an impossible task: the network must learn `f(h_s) ≈ P(s on path)`
purely from weight updates, with no persistent state memory.

**Why DIME-Planning does not have this problem**:

In DIME-Planning, `f(s)` takes the **raw PDDL state observation** via a GNN.
The GNN weights ARE the persistent signal — they are shared across all planning problems
and all episodes. A state s that appeared in 1000 training problems will have consistent
GNN embeddings reflecting what was learned about it.

There is no tree memory that resets. The GNN IS the memory.

**Comparison**:
```
MCTS-DIME:    f(h_s)  — h_s rebuilt each episode, weights shared
DIME-Planning: f(s)   — s observed directly via GNN, weights persistent and accumulate
```

This is the fundamental architectural advantage of the planning setting.

---

## Critical Bugs Found and Fixed

### Bug 1: Label Threshold (`best_reward = -1.0`)

**Location**: `mctsdime/mctsdime/train.py`, function `compute_labels()`

**Bug**:
```python
best_reward, best_leaf_id = -1.0, None  # WRONG
```

**Effect**: Any leaf with `reward = 0` (the default for non-goal leaves) was treated as
"better than best", triggering `y=1` labels for ALL ancestors of that leaf.
In sparse reward environments (where 98%+ of episodes find NO reward), this poisons
essentially all training data — the predictor learned to predict "on path" for
states leading to dead ends.

**Fix**:
```python
best_reward, best_leaf_id = 0.0, None   # CORRECT: require reward > 0
```

**For DIME-Planning**: In Julia pseudo-code:
```julia
best_reward = 0.0   # NOT -Inf or -1.0
for node in search_tree
    if node.is_goal && node.reward > best_reward
        best_reward = node.reward
        best_node = node
    end
end
# Only assign y=1 if best_reward > 0
```

**Rule**: `y_s = 1` ONLY for ancestors of states with positive reward (goal reached).
If no goal is found in the episode, ALL states get `y_s = 0`. Never assign positive
labels from a non-goal node.

### Bug 2: Variable Tree Environment

**Location**: `mctsdime/exp1_cmi_diagnostic.py`

**Bug**: `env = BinaryTreeEnv(seed=ep)` inside the episode loop.

**Effect**: Every episode had a different tree with different reward location.
The predictor saw "positive label on left subtree" in episode 1, "positive label on right
subtree" in episode 2, etc. No consistent signal.

**Fix**: Create ONE environment outside the loop with a fixed seed.

**For DIME-Planning**: Not directly applicable (IPC problems are fixed), but the general
principle holds: during debugging, ensure the training distribution is consistent and
that positive labels come from actual solved problems.

### Bug 3: Evaluation Loop (DeepSea)

**Location**: `mctsdime/exp2_deepsea.py`

**Bug**: Run MCTS from root once, then follow tree greedily to evaluate.

**Effect**: For deep nodes (e.g., step 7 in 8×8 grid), the greedy path from root
often led to nodes that were never expanded — `best_action()` returned random actions.

**Fix**: Re-run MCTS search at each step from the current state.

**For DIME-Planning**: Less relevant (GBFS evaluations don't have this structure),
but when writing evaluation code, ensure each test problem is solved fresh — do not
cache tree structure across independent test problems.

### Bug 4: DFS Bias from Untrained Network

**Location**: MCTS-DIME acquisition function

**Bug**: Random network weights cause `log f(h_s)` to be consistently higher for
one branch → acquisition function always selects the same child → depth-first search
(never backtracks).

**Fix**: Added UCB exploration term: `c_ucb * sqrt(log N_parent / N_child)`

**For DIME-Planning**: At initialisation, ensure the GBFS expansion policy is not
degenerate. Consider initialising with small random noise on `log f(s)` to break ties.
More importantly: during early training, verify that the planner is actually exploring
(not just following one path).

---

## What λ=0 (UCT) Comparison Showed

The baseline UCT (λ=0 in MCTS-DIME, equivalent to standard MCTS with visit count bonus)
performed the same as MCTS-DIME on DeepSea. This confirmed that the failure was not
about the CMI signal being wrong — the entire approach (MCTS-DIME) was architecturally
unsuited for cross-episode generalisation.

**For DIME-Planning**: The comparison between `lgbfs` (λ=0) and `DIME` (λ>0) is more
meaningful than MCTS comparisons because:
1. Both use the same GNN backbone (persistent, cross-problem)
2. The only difference is whether `v(s)` guides expansion
3. The GNN weights accumulate knowledge across problems — the predictor quality improves
   with training data, unlike MCTS-DIME where it resets

---

## The λ Annealing Question

**From MCTS-DIME experiments**: At λ=1.0 fixed, the CMI exploration can compete with
the converged predictor — once `f` is confident, `v(s)→0` for familiar states,
so the CMI term effectively drives search AWAY from the known optimal path.

**For DIME-Planning**: Two options:
1. Fixed λ — simplest, start here
2. Annealed λ: `λ(t) = λ_0 · decay^t` — reduce exploration as predictor converges

Start with fixed λ ∈ {0.1, 0.5, 1.0, 2.0} (sweep). If results show best λ decreasing
with training epochs, switch to annealing.

**Expected behaviour**: λ=0.1 is "near-lgbfs", λ=2.0 is "heavy exploration". The optimal
λ probably depends on domain difficulty — easy domains prefer low λ, hard domains with
dead ends prefer high λ.

---

## The Δ(s) Computation Challenge

**What Δ(s) is**: The reduction in predictor loss from expanding state s:
```
Δ(s) = L_pred(T_before) - L_pred(T_after)
```
where `T_before` is the search tree before expanding s, `T_after` includes s's children.

**Why it's hard to compute exactly**: Requires two forward passes of the full model
on the problem's state set, before and after each expansion. Expensive.

**Practical approximations**:
1. **Change in logit**: `Δ(s) ≈ log f(s_{child}) - log f(s)` — proxy without full recomputation
2. **Batch approximation**: compute Δ over a mini-batch of recent expansions post-hoc
3. **Offline from trace**: After running GBFS to get the full search trace, compute Δ
   for each expansion step retrospectively

**Recommended approach**: Start with offline computation from the search trace (option 3).
It's simpler to implement and correct. Only optimise if training throughput is too slow.

---

## Summary: What to Preserve vs Avoid

**Preserve from MCTS-DIME:**
- The CMI diagnostic (Exp1) validated DIME Lemma 1 → reproduce in planning context
- Two-head architecture (shared backbone, predictor head + value head) — use same design
- Label rule: y=1 only for positive-reward ancestors

**Avoid from MCTS-DIME:**
- Tree-conditioned memories (h_s) — use raw state observations (GNN) instead
- Resetting predictor state each episode — GNN weights persist across all problems
- DeepSea as a benchmark for learned planners — it requires cross-episode memory

**DIME-Planning's fundamental advantage:**
The GNN backbone provides exactly the persistent, cross-episode representation that
MCTS-DIME lacked. Training on 1000 problems with 5 seeds accumulates thousands of
(s, y_s, Δ(s)) tuples that jointly train the GNN to recognise "what makes a state
likely on the optimal path" across the entire domain — not just within one episode.
This is why the planning setting is the right setting for DIME.
