# Related Work and Positioning

## Overview

DIME-Planning sits at the intersection of three research streams:
learned planning heuristics, information-theoretic exploration, and MCTS with uncertainty.

---

## Comparison Table

| Method | Search | Predictor type | Exploration signal | Cross-problem? | Self-supervised? |
| --- | --- | --- | --- | --- | --- |
| A\* + h\* | GBFS/A\* | Regression (h\*) | None | Yes (GNN) | No (needs h\*) |
| LEAH / lgbfs | GBFS | Ranking (P on path) | None | Yes (GNN) | No (needs plans) |
| L\* | GBFS | Regression (h\*) | None | Yes (GNN) | No (needs h\*) |
| LEAH bootstrapped | GBFS | Ranking | None | Yes (GNN) | Yes (self-play) |
| MCTS-UCT | MCTS | Q-value (visit count) | UCB visit count | No | N/A |
| AlphaZero | MCTS | Policy + value | PUCT (visit count) | No | Yes (self-play) |
| Epistemic MCTS | MCTS | Q-value + UBE | Epistemic variance | No | Yes |
| MCTS-DIME | MCTS | f(h\_s) tree memory | CMI via v(h\_s) | No (per-episode memory) | Yes |
| **DIME-Planning** | **GBFS** | **f(s) discriminative** | **CMI via v(s)** | **Yes (GNN)** | **Partially** |

**Key differentiators of DIME-Planning:**

- Only method combining CMI-based exploration with cross-problem GNN generalisation
- Formal information-theoretic grounding (DIME Lemma 1, Gadgil et al., ICLR 2024)
- Directly extends lgbfs (Chrestien et al., NeurIPS 2023) — the current best learned GBFS approach

---

## LEAH / Chrestien et al. (NeurIPS 2023)

**Method**: Learn a heuristic to *rank* open-list nodes. Defines L_gbfs (ranking loss) and
L* (cost-to-go regression). Uses GNN on PDDL state graphs (ObjectAtom, AtomBinary architectures).

**Relation to DIME-Planning**: lgbfs is the λ=0 special case of DIME-Planning. DIME adds
the value network v(s) estimating CMI and the joint acquisition function.

**Code**: `NeuroPlannerExperiments.jl` implements all LEAH losses and architectures.

**Paper**: Chrestien, Pevný, Šír, Schewe (NeurIPS 2023). Official repo:
`github.com/aicenter/Optimize-Planning-Heuristics-to-Rank`

---

## NeuroPlanner / Šír et al

**Method**: GNN-based neural heuristics for PDDL planning, supporting multiple loss functions
and architectures. The underlying library this project uses.

**Relation**: DIME-Planning is implemented as an extension of NeuroPlanner + NeuroPlannerExperiments.

---

## Epistemic MCTS / E-AlphaZero (Oren et al., ICLR 2025)

**Method**: AlphaZero extended with epistemic uncertainty estimation via:

- UBE head (Uncertainty Bellman Equation): propagates uncertainty backward via `σ²(V(s)) ← σ²(r) + γ² σ²(V(s'))`
- SimHash: detects novel states, assigns high reward uncertainty to unseen states
- Two policy heads: exploitation policy (standard) and exploration policy (driven by σ(V))

**Relation to DIME-Planning**:

- Both use a learned uncertainty/information signal to guide exploration
- Epistemic MCTS propagates uncertainty via Bellman (value-based); DIME uses CMI (information-theoretic)
- Epistemic MCTS needs a world model + replay buffer; DIME works from solved plans
- DIME generalises across problems (GNN); epistemic MCTS is per-instance

**Key difference**: epistemic MCTS measures *model uncertainty* (variance across ensemble);
DIME measures *plan uncertainty* (CMI about which path is optimal). These are complementary.

**Code**: JAX/Haiku, `github.com/adaptive-agents-lab/epistemic-mcts`

---

## MCTS-DIME (companion project)

**Method**: Applies DIME to MCTS instead of GBFS. Uses MCTSnets-style learned backup network
to build tree-conditioned memories `h_s`. Predictor `f(h_s)` and value network `v(h_s)` guide
simulation policy via acquisition function `a(s) = log f(h_s) + λ v(h_s)`.

**Key finding**: Validates DIME Lemma 1 in tree-search context (Experiment 1: Spearman ρ → +0.93).
But fails to generalise across episodes in navigation (DeepSea) because memories are rebuilt
from scratch each episode — no persistent cross-episode signal.

**Why DIME-Planning avoids this**: `f(s)` takes raw state observations via GNN, which
generalises across episodes by design. This is the fundamental architectural advantage of
the planning setting.

**Code/writeup**: [`overleaf/mctsdime.tex`](../overleaf/mctsdime.tex) (submodule). Implementation lives outside this repo.

---

## TreeDIME (companion project, different from this)

**Method**: Applies DIME to tree-structured heuristic search (closest predecessor of this project).
Framing is closer to TreeMDP + DIME. Theory-focused, less experimental.

**Relation**: DIME-Planning is the experimental, IPC-benchmarked instantiation of this idea.

**Writeup**: [`overleaf/main.tex`](../overleaf/main.tex) (submodule)

---

## MCTSnets (Guez et al., ICML 2018)

**Method**: Replaces all four MCTS phases (embedding, backup, simulation, readout) with learned
neural networks. Node memories `h_s ∈ R^d` are tree-conditioned via a learned backup network.

**Superseded by**: MuZero (Schrittwieser et al., Nature 2020), which jointly learns a latent
world model + MCTS.

**Relation**: MCTS-DIME builds on MCTSnets. Not directly relevant to DIME-Planning (different
search algorithm).

---

## AlphaZero / MuZero

**Method**: MCTS guided by policy + value networks trained by self-play. PUCT selection:
`UCT(s) = Q(s) + c·P(s)·sqrt(N_parent)/(1+N(s))`.

**Relation**: DIME-Planning's acquisition function `a(s) = log f(s) + λ v(s)` parallels PUCT
with `f` as policy and `v` as exploration bonus. Key difference: PUCT's bonus is visit-count-based
(frequentist), DIME's is information-theoretic (principled).

---

## Levin Tree Search (LTS)

**Method**: Combines policy (probability of correct action at each node) with search cost.
Implemented in `NeuroPlannerExperiments.jl/src/levin_tree_search.jl`.

**Relation**: LTS uses a policy network; DIME uses both a predictor and an information signal.
LTS optimises for anytime performance; DIME optimises for efficient goal finding.

---

## Adaptive Submodularity (Golovin & Krause, 2011)

**Theory**: For adaptive submodular objective functions, greedy adaptive maximisation achieves
`(1 - 1/e)` approximation to the optimal adaptive policy.

**Relation**: `I(Y; X_S)` is submodular in `S`. Greedy CMI maximisation (which is what DIME's
acquisition function approximates) therefore has a formal `(1 - 1/e)` approximation guarantee
for the information gathered within a budget of `k` expansions.

This is a key theoretical selling point for DIME-Planning in the paper.
