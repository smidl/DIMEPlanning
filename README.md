# DIMEPlanning

## Discriminative Mutual Information Estimation for Heuristic Planning

A research project applying DIME-style information-theoretic exploration to classical AI planning via the TreeMDP framework. The goal is a publication-quality paper showing that CMI-guided node expansion outperforms purely heuristic-driven planners on IPC benchmarks.

---

## Project Overview

Standard learned planners (LEAH, NeuroPlanner) train a heuristic `h(s)` to estimate cost-to-go and use it to guide GBFS/A*. This project replaces or augments the heuristic with a **discriminative predictor** `f(s) = P(s is on the optimal plan)` and a **value network** `v(s) = E[CMI gained by expanding s]`, grounded in DIME (Gadgil et al., ICLR 2024).

The acquisition function for node expansion becomes:

```math
a(s) = log f(s) + λ · v(s)
```

where `λ` trades exploitation (expand likely-optimal nodes) against exploration (expand informative nodes). At `λ=0` this recovers the standard lgbfs predictor. At `λ>0` the CMI term actively seeks nodes that would most reduce uncertainty about the optimal plan.

**Key claim**: The existing `lgbfs` loss in NeuroPlannerExperiments.jl already trains `f` implicitly. DIME's novel contribution is `v(s)` and the joint acquisition function — a targeted, low-overhead extension of the existing pipeline.

---

## Repository Structure

```text
DIMEPlanning/
├── README.md                  ← this file
├── CLAUDE.md                  ← context for AI assistants
├── docs/
│   ├── method.md              ← full theoretical description of DIME-Planning
│   ├── related_work.md        ← positioning vs LEAH, MCTSnets, Epistemic MCTS
│   ├── experimental_plan.md   ← what to run, on what domains, with what metrics
│   ├── infrastructure.md      ← NeuroPlannerExperiments.jl guide and extension points
│   └── lessons_learned.md     ← prior exploration results and pitfalls to avoid
└── src/                       ← (to be created) Julia implementation
    ├── DIMEPlanning.jl
    ├── dime_loss.jl
    ├── dime_planner.jl
    └── dime_train.jl
```

---

## Prerequisites

- Julia 1.11+ with access to `NeuroPlanner.jl` (ask repo owner for access if private)
- `NeuroPlannerExperiments.jl` (sibling directory or registered package)
- IPC 2023 planning domains (download script in `NeuroPlannerExperiments.jl/src/datasets.jl`)
- SLURM cluster for large-scale experiments (see `docs/experimental_plan.md`)

---

## Quick Start (once implementation exists)

```julia
# Install dependencies
using Pkg; Pkg.activate("."); Pkg.instantiate()

# Run a single DIME experiment on Blocksworld
julia src/run_experiment.jl --domain blocksworld --loss dime --lambda 1.0 --seed 1
```

---

## Key References

- **DIME**: Gadgil et al., "Discriminative Estimation of Total Variation Distance" (ICLR 2024)
- **TreeMDP / L_gbfs**: Chrestien et al., "Optimizing Planning Heuristics to Rank" (NeurIPS 2023)
- **LEAH**: Same as above; learning heuristics via ranking objectives
- **L***: Cost-to-go regression loss (used as baseline in NeuroPlannerExperiments.jl)
- **NeuroPlanner.jl**: The underlying planning + GNN library this project builds on
- **MCTS-DIME**: Companion exploration applying DIME to MCTS (see `docs/lessons_learned.md`)

---

## Status

- [ ] Method design (see `docs/method.md`)
- [ ] Julia implementation (`src/`)
- [ ] Single-domain validation (Blocksworld, easy)
- [ ] Full IPC 2023 benchmark sweep
- [ ] Paper writing

---

## Contact / Collaborators

See git log for contributors. For questions about the theoretical background, see `docs/method.md` and `CLAUDE.md`.
