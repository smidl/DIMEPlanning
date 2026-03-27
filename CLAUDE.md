# CLAUDE.md — AI Assistant Context for DIMEPlanning

This file provides full context for AI assistants (Claude Code and others) working on this project.
Read it at the start of every session before making any changes.

---

## Repository Structure

```text
DIMEPlanning/
├── CLAUDE.md                        ← this file (AI context)
├── README.md                        ← project overview
├── papers/                          ← PDFs + theory summaries for all key papers
│   ├── README.md                    ← convention for adding new papers
│   ├── leah2022.pdf + leah2022_theory.md   ← LEAH / Chrestien et al. NeurIPS 2023
│   ├── treemdp.pdf + treemdp_theory.md     ← TreeMDP / IJCAI 2025 submission
│   ├── dime.pdf + dime_theory.md           ← DIME / Gadgil et al. ICLR 2024
│   ├── mctsnets.pdf + mctsnets_theory.md   ← MCTSnets / Guez et al. ICML 2018
│   └── jin20a.pdf + jin20a_theory.md       ← Linear MDP / Jin et al. COLT 2020
├── docs/
│   ├── method.md                    ← full theoretical description of DIME-Planning
│   ├── related_work.md              ← positioning vs other methods
│   ├── experimental_plan.md         ← what to run, domains, metrics, SLURM
│   ├── infrastructure.md            ← NeuroPlannerExperiments.jl guide
│   └── lessons_learned.md           ← MCTS-DIME failures and pitfalls
├── src/                             ← (to be created) Julia implementation
│   ├── DIMEPlanning.jl
│   ├── dime_loss.jl
│   ├── dime_planner.jl
│   └── dime_train.jl
├── experiments/                     ← (to be created) experiment scripts
│   └── imitation/
│       ├── run_experiment.jl        ← SLURM entry point
│       └── run_and_show.jl          ← config generation + analysis
├── overleaf/                        ← git submodule: Overleaf paper repository
│   ├── main.tex                     ← TreeDIME paper (planning-focused)
│   └── mctsdime.tex                 ← MCTS-DIME companion paper
├── NeuroPlannerExperiments.jl/      ← git submodule: public experiment pipeline
└── PrivateNeuroPlanner.jl/          ← git submodule: private planning library (TreeMDP)
```

All path references in this repo are relative to the DIMEPlanning root (no `../` paths).

---

## What this project is

A research project applying the **DIME** (Discriminative Mutual Information Estimation)
framework to classical AI planning via the **TreeMDP** formulation. The goal is a paper
showing that CMI-guided node expansion improves over state-of-the-art learned planners
(LEAH, NeuroPlanner) on IPC 2023 benchmarks.

The project is **Julia-based**, building on two existing packages:

- `PrivateNeuroPlanner.jl` — PDDL parsing, GNN-based state extractors, planning algorithms,
  TreeMDP formulation (see `PrivateNeuroPlanner.jl/` submodule)
- `NeuroPlannerExperiments.jl` — experiment pipeline (configs, SLURM, results collection)
  (see `NeuroPlannerExperiments.jl/` submodule)

Do not modify these packages — extend them from `DIMEPlanning` via Julia's multiple dispatch.

---

## CRITICAL: Referencing Convention

**Always verify claims against the theory files before writing or coding.**

When citing a paper in docs, code comments, or the LaTeX paper:

1. Check `papers/{key}_theory.md` for the formal statement
2. Use the exact theorem/lemma number: "DIME Lemma 1" not "the DIME paper shows"
3. Use the full citation: "Chrestien et al. (NeurIPS 2023)" not "the LEAH paper"
4. Never paraphrase key theoretical results from memory

When a new paper becomes relevant:

1. Add the PDF to `papers/`
2. Write a `_theory.md` summary following the convention in `papers/README.md`
3. Add it to the `papers/README.md` index

---

## CRITICAL: LEAH vs TreeMDP — They Are Different Papers

These are frequently conflated but are **distinct works**:

### LEAH — Chrestien et al., NeurIPS 2023

**Paper**: `papers/leah2022.pdf` / **Summary**: `papers/leah2022_theory.md`

**What it contributes**:

- Proposes training GBFS heuristics via a **ranking loss** (L_gbfs) rather than regression to h\*
- Defines the "perfect ranking heuristic" (Definition 1, Theorem 1 in the paper)
- Shows ranking loss converges faster than regression (VC theory argument)
- L_gbfs penalises any non-optimal open-list node ranking above an optimal one:
  `L_gbfs(h, T) = Σ_{s+ ∈ optimal, s- ∉ optimal} log(1 + exp(h(s+) - h(s-)))`
- Provides GNN architectures (ObjectAtom, AtomBinaryFE, etc.)

**Code**: Public — `github.com/aicenter/Optimize-Planning-Heuristics-to-Rank`
(same code is in `NeuroPlannerExperiments.jl/` submodule)

### TreeMDP — IJCAI 2025 submission (unpublished)

**Paper**: `papers/treemdp.pdf` / **Summary**: `papers/treemdp_theory.md`

**What it contributes**:

- Reframes GBFS as an **MDP over partially-expanded search trees** (Tree-MDP)
- Shows L_gbfs with softmax = policy gradient in Tree-MDP (new theoretical insight)
- Introduces **L_po** (on-policy loss): uses the actual GBFS expansion trace, not the
  optimal trace — avoids off-policy distribution shift
- L_po has O(n) complexity vs O(n²) for L_gbfs

**Code**: Private — `PrivateNeuroPlanner.jl/` submodule

### Summary table

| Aspect | LEAH | TreeMDP |
| --- | --- | --- |
| Venue | NeurIPS 2023 | IJCAI 2025 (submitted) |
| Key loss | L_gbfs | L_po |
| Key insight | Ranking > regression | GBFS = MDP, L_po avoids off-policy |
| Code | Public NeuroPlannerExperiments.jl | Private PrivateNeuroPlanner.jl |
| Cite when | Referring to L_gbfs, ranking, architectures | Referring to Tree-MDP formulation, L_po |

---

## Key Theoretical Concepts

### DIME (Gadgil et al., ICLR 2024)

**Full paper + summary**: `papers/dime.pdf` / `papers/dime_theory.md`

DIME trains two networks jointly on a binary classification task:

- **Predictor** `f(x_S)` — discriminates p(y|x_S) from p(y); trained via BCE
- **Value network** `v(x_S)` — estimates expected CMI from acquiring one more feature;
  trained to predict `Δ(s) = E[L_pred(x_S) − L_pred(x_{S∪{new}})]`

**DIME Lemma 1** (Gadgil et al., ICLR 2024, Lemma 1): At joint optimality,
`v(x_S) → I(Y; X_new | X_S)` (the value network recovers the conditional mutual information).

Acquisition function: `a(S, new) = log f(x_S) + λ·v(x_S)`

### TreeMDP (IJCAI 2025 submission)

**Full paper + summary**: `papers/treemdp.pdf` / `papers/treemdp_theory.md`

GBFS recast as an MDP over partially-expanded search trees:

- **State**: the current search tree T (open list + expanded nodes)
- **Action**: which node s ∈ open(T) to expand next
- **Reward**: reaching the goal

Key losses (from LEAH and TreeMDP together):

- `L_gbfs`: ranking loss — optimal-path nodes should rank above non-optimal expanded nodes
- `L*`: regression loss — predict optimal cost-to-go h\*(s)
- `L_po`: on-policy loss — uses actual GBFS expansion trace (TreeMDP paper only)

### DIME-Planning (this project)

The core proposal: replace the GBFS priority h(s) with an acquisition function:

```text
a(s) = log f(s) + λ · v(s)
```

where:

- `f(s) = P(s is on the optimal plan | current search state)` — the DIME predictor
- `v(s) = E[CMI gained by expanding s]` — the DIME value network
- `λ ≥ 0` — trades exploitation vs exploration

**Critical insight**: The existing `lgbfs` loss (from LEAH, Chrestien et al.) already
trains something structurally equivalent to `f` — it learns to rank optimal-path nodes
above expanded-but-suboptimal nodes, which is equivalent to learning
`P(s on optimal path)`. So **lgbfs IS the λ=0 ablation**.
DIME's novel contribution is `v(s)` and the joint acquisition function.

---

## Relationship to MCTS-DIME (companion project)

A parallel exploration applied DIME to MCTS (code in `mctsdime/` sibling directory
outside this repo). Key findings — see `docs/lessons_learned.md` for details.

**What worked**: CMI diagnostic (Exp1) — Spearman ρ → +0.93 on fixed binary tree.
Validates DIME Lemma 1 in tree-search context.

**What did NOT work**: Cross-episode generalisation — MCTS-DIME's tree memories `h_s`
reset each episode, no persistent predictor signal. DeepSea solve rate: 0%.

**Why DIME-Planning avoids this**: `f(s)` takes raw PDDL state via GNN. GNN weights
persist and accumulate knowledge across ALL planning problems and episodes. This is the
fundamental architectural advantage that MCTS-DIME lacked.

---

## The Predictor Quality Issue

The central challenge: `f(s)` must be good enough that `log f(s)` is a useful
exploitation signal, and `v(s)` must be trained on meaningful `Δ(s)` targets.

**Δ(s) computation in planning:**

```text
Δ(s) = L_pred(before expanding s) − L_pred(after expanding s)
```

This requires tracking predictor loss before/after each expansion. Natural in the
TreeMDP training loop where we have access to the full search trace.

**Critical label rule (learned from MCTS-DIME bugs):**
Labels `y=1` must ONLY be assigned to ancestors of states with **positive reward**.
If no plan is found, all labels must be `y=0`. Initialising `best_reward = -1.0` (wrong)
instead of `best_reward = 0.0` (correct) causes dead-end ancestors to get positive labels.

In Julia code (pseudo):
```julia
best_reward = 0.0   # NOT -Inf or -1.0 — this threshold matters
for node in search_tree
    if node.is_goal && node.reward > best_reward
        best_reward = node.reward
        best_node = node
    end
end
# Only if best_reward > 0 do we assign y=1 to ancestors
```

---

## Key Files in Submodules

- [NeuroPlannerExperiments.jl/src/losses.jl](NeuroPlannerExperiments.jl/src/losses.jl) — lgbfs, lstar loss implementations
- [NeuroPlannerExperiments.jl/src/forward.jl](NeuroPlannerExperiments.jl/src/forward.jl) — ForwardPlannerX (A\*/GBFS base)
- [NeuroPlannerExperiments.jl/src/planners.jl](NeuroPlannerExperiments.jl/src/planners.jl) — planner config structs
- [NeuroPlannerExperiments.jl/src/train.jl](NeuroPlannerExperiments.jl/src/train.jl) — training loop
- [NeuroPlannerExperiments.jl/imitation/run_experiment.jl](NeuroPlannerExperiments.jl/imitation/run_experiment.jl) — SLURM entry point (imitation)
- [PrivateNeuroPlanner.jl/src/](PrivateNeuroPlanner.jl/src/) — TreeMDP-based planning library (L_po, etc.)
- [overleaf/main.tex](overleaf/main.tex) — TreeDIME planning paper (this project's paper)
- [overleaf/mctsdime.tex](overleaf/mctsdime.tex) — MCTS-DIME companion paper

---

## Development Conventions

- The Julia package should be named `DIMEPlanning` and extend NeuroPlannerExperiments.jl
- New loss: `DimeLoss` — wraps lgbfs predictor + value network, computable from search trace
- New planner: `DimePlanner` — `ForwardPlannerX` with priority `a(s) = log f(s) + λ·v(s)`
- All configs JSON-compatible, following NeuroPlannerExperiments.jl conventions
- Results stored in same directory structure as NeuroPlannerExperiments.jl
- Do not modify submodules — extend via Julia's multiple dispatch
- All internal links use paths relative to DIMEPlanning root (never `../`)

---

## Open Research Questions

1. **Should v(s) share the GNN backbone with f(s)?** Probably yes — both need state features.
   A shared extractor with two heads mirrors the DIME architecture.

2. **How to compute Δ(s) efficiently?** Approximations: use the change in f(s)'s logit
   as a proxy, or compute Δ over a mini-batch of recent expansions.

3. **Online vs offline training of v(s)?** Option (a) offline from solved plans
   (Δ computed post-hoc from the full trace) is simpler to start with.

4. **Does λ need annealing?** From MCTS-DIME: at λ=1.0 fixed, CMI exploration can fight
   a converged predictor. A schedule λ(t) = λ_0 · decay^t may be needed.

5. **Connection to epistemic uncertainty**: DIME's v(s) is a form of epistemic uncertainty
   estimate. This connects to Epistemic MCTS (Oren et al., ICLR 2025) which propagates
   model uncertainty via a UBE (Uncertainty Bellman Equation). Worth discussing in related work.
