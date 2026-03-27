# DIME-Planning: Method Description

## 1. Problem Setting

We consider classical AI planning: given a domain `D` and a problem `P = (s_0, G)` with
initial state `s_0` and goal `G`, find a sequence of actions `π = (a_1, ..., a_T)` such that
executing `π` from `s_0` reaches a state satisfying `G`.

We use **Greedy Best-First Search (GBFS)**: maintain an open list (priority queue), expand
the node with the best priority `f(s)`, add successors to the open list, repeat until the
goal is reached. The priority function `f(s)` is what we learn.

**Standard approach**: train `f(s) ≈ h*(s)` (cost-to-go regression) or as a ranking function
over open-list nodes. At test time, GBFS with the learned `f` solves new problems in the
same domain.

---

## 2. TreeMDP Framework (Chrestien et al., NeurIPS 2023)

GBFS is recast as an MDP over **partially-expanded search trees** `T`:

- **State**: the search tree `T` (open list + expanded nodes)
- **Action**: node `s ∈ open(T)` to expand next
- **Transition**: expand `s`, add successors to open list → `T'`
- **Reward**: `1` if goal reached, `0` otherwise (episode terminates at goal or timeout)

This formulation enables defining proper learning objectives:

### L_gbfs (ranking loss)
Let `π*(T)` be the set of open-list nodes on the optimal continuation of the plan from `T`.
The ranking loss penalises any non-optimal open-list node ranking above an optimal one:

```
L_gbfs(f, T) = E_{s+ ∈ π*(T), s- ∉ π*(T)} [max(0, f(s-) - f(s+))]
```

Equivalently: `f(s+) > f(s-)` for all optimal vs non-optimal node pairs.

This is a **discriminative objective**: `f` learns to classify optimal-path nodes vs others.
At optimality, `f(s) ≈ P(s ∈ optimal path | T)` — exactly DIME's predictor.

### L* (regression loss)
```
L*(f, T) = E_s [( f(s) - h*(s) )^2]
```
Requires access to oracle `h*` (from training plans). Less discriminative than L_gbfs.

### L_po (on-policy variant)
Computed from the actual GBFS expansion trace with the current policy, avoiding distribution
shift between train and test. More expensive to compute but more stable for self-play training.

---

## 3. DIME Framework (Gadgil et al., ICLR 2024)

DIME trains two networks for **adaptive feature acquisition**:

**Setup**: latent variable `Y`, observable features `X = (X_1, ..., X_d)`, subset `S ⊆ [d]`
already observed.

**Predictor** `f(x_S)`: binary classifier distinguishing `p(y|x_S)` from `p(y)`. Trained via:
```
L_pred = E [BCE(f(x_S), y)]
```

**Value network** `v(x_S)`: estimates expected CMI from acquiring one more feature:
```
v(x_S) ≈ I(Y; X_new | X_S)  [at optimality, by DIME Lemma 1]
```
Trained to predict the CMI proxy:
```
Δ(S) = E [L_pred(x_S) - L_pred(x_{S ∪ {new}})]
```

**Acquisition function**: at each step, acquire the feature maximising:
```
a(S, new) = log f(x_S) + λ · v(x_S)
```

**DIME Lemma 1**: At joint optimality of `f` and `v`, `v(x_S) → I(Y; X_new | X_S)`.
This gives DIME a formal information-theoretic grounding.

---

## 4. DIME-Planning (this work)

### Correspondence

| DIME concept | Planning interpretation |
|---|---|
| Latent `Y` | Is `s` on the optimal plan? (binary) |
| Features `X_S` | States expanded so far in search |
| Predictor `f(x_S)` | `P(s ∈ optimal path \| current search state)` |
| CMI `I(Y; X_new \| X_S)` | Information gained about optimal plan by expanding `s` |
| Value `v(x_S)` | Expected reduction in plan uncertainty from expanding `s` |
| Acquisition `a(S, new)` | GBFS node expansion priority |

### Method

**Predictor** `f_θ(s)`: GNN applied to the PDDL state graph of `s`, outputting a scalar
in `[0, 1]`. Trained via binary cross-entropy:

```
y_s = 1  if s appears on the optimal plan
y_s = 0  if s was expanded but not on the optimal plan

L_pred = -E_s [y_s log f(s) + (1 - y_s) log(1 - f(s))]
```

Note: this is structurally identical to `lgbfs` training. The lgbfs loss can be seen as
a margin-based approximation to this BCE objective. Training with BCE directly is
cleaner and enables the CMI value estimation.

**Value network** `v_φ(s)`: second head on the same GNN backbone, outputting a
non-negative scalar via Softplus activation. Trained to predict CMI proxy:

```
Δ(s) = L_pred(before expanding s) - L_pred(after expanding s)
```

Loss:
```
L_value = E_s [( v(s) - max(0, Δ(s)) )^2]
```

**Joint training**: for each solved planning problem, collect `(s, y_s, Δ(s))` tuples
from the search trace, then:
```
L_total = L_pred + α · L_value
```

**GBFS expansion priority**:
```
a(s) = log f(s) + λ · v(s)
```
where `λ ≥ 0` is a hyperparameter (default: 1.0). Higher priority = expanded first.

### Architecture

```
State s (PDDL atom graph)
       ↓
GNN extractor (shared backbone, e.g. ObjectAtom or AtomBinaryFE)
       ↓
Shared representation h_s ∈ R^d
       ↙           ↘
Predictor head    Value head
MLP → Sigmoid     MLP → Softplus
f(s) ∈ [0,1]     v(s) ≥ 0
```

This follows the DIME architecture exactly, with the GNN as the shared feature extractor.

---

## 5. Connection to Existing Losses

### lgbfs ≈ f at λ=0

The lgbfs loss ranks `f(s+) > f(s-)` for optimal vs non-optimal nodes. This is a relaxed
version of training `f` as a binary classifier. At convergence both give `f(s) ∝ P(s on path)`.
Therefore: **lgbfs with any planner = DIME with λ=0** (up to the loss approximation).

This means:
- The existing lgbfs baseline in NeuroPlannerExperiments.jl is the ablation `λ=0`
- DIME adds only `v(s)` — a second head and the CMI proxy computation
- The experimental comparison directly tests whether CMI guidance helps

### lstar vs DIME

`lstar` trains a regression to `h*(s)` (cost-to-go). This is fundamentally different from DIME:
- `h*(s)` is a continuous scalar; `f(s)` is a probability
- `lstar` doesn't distinguish "on optimal path" from "not on optimal path"
- `lstar` is harder to train (requires knowing `h*`); `f` only needs binary labels from the plan

DIME-Planning is expected to outperform `lstar` on problems where the optimal path is highly
non-uniform (exponential search space with one narrow corridor to the goal).

---

## 6. Training Protocol

### Phase 1: Imitation (offline, from expert plans)

For each training problem with a known solution:
1. Run GBFS with current `f` (or uniform priority initially)
2. Collect all states seen during search: `expanded ∪ open_list`
3. Assign labels: `y_s = 1` if `s` on optimal plan, else `0`
4. Compute `Δ(s)` from search trace (see Section 4)
5. Update `θ, φ` via one gradient step

This mirrors the imitation pipeline in NeuroPlannerExperiments.jl.

### Phase 2: Bootstrapped (self-play)

Run GBFS with current `f, v` on new problems. If a solution is found, use it as training data
for the next round. This mirrors the self-learning pipeline in NeuroPlannerExperiments.jl.

### Label rule (critical)

`y_s = 1` ONLY for ancestors of states that reach the goal (`reward > 0`).
If no solution is found in an episode, ALL states get `y_s = 0`.
See `CLAUDE.md` for the motivation (the MCTS-DIME label bug).

---

## 7. Theoretical Properties

### CMI monotonicity
By DIME Lemma 1: at optimality, expanding nodes in order of decreasing `v(s)` maximises
the information gained per expansion — a greedy approximation to the optimal adaptive
plan of queries.

### Submodularity
Mutual information `I(Y; X_S)` is submodular in `S`. The greedy CMI maximisation therefore
achieves a `(1 - 1/e)` approximation ratio to the optimal subset selection (Golovin & Krause,
2011, adaptive submodularity).

### Connection to UCT
The acquisition function `a(s) = log f(s) + λ v(s)` parallels the UCT formula:
```
UCT(s) = Q(s) + c · sqrt(log N(parent) / N(s))
```
where `log f(s)` plays the role of `Q(s)` (exploitation) and `λ v(s)` plays the role of
the exploration bonus. The key difference: UCT's bonus is visit-count-based (frequentist),
while DIME's bonus is information-theoretic (principled).

---

## 8. Expected Results

On easy problems: all methods perform similarly (search is trivial).

On medium/hard problems with sparse rewards (long optimal plans, large branching factor):
- `lstar`: may fail when `h*` is hard to approximate
- `lgbfs`: good ranking, but explores uniformly once the known path is found
- **DIME (λ>0)**: `v(s)` directs expansion toward nodes that would most reduce plan
  uncertainty — expected to explore more efficiently and solve harder problems

The key advantage of DIME over lgbfs is expected on domains like **Sokoban** and
**Floortile** where the search space has many dead ends and the optimal path is narrow.
