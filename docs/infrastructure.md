# Infrastructure Guide: NeuroPlannerExperiments.jl

This document describes the existing experimental pipeline that DIME-Planning builds on.
Do not modify NeuroPlannerExperiments.jl — extend it from this package via Julia's
multiple dispatch.

Location: `NeuroPlannerExperiments.jl/` (git submodule in this repo)

---

## Package Structure

```text
NeuroPlannerExperiments.jl/
├── Project.toml                   # Julia dependencies
├── src/
│   ├── NeuroPlannerExperiments.jl # Main module, exports
│   ├── configs.jl                 # @confdef config structs (Parameters.jl-based)
│   ├── datasets.jl                # IPC 2023 domain loading, Google Drive download
│   ├── models.jl                  # Neural network construction from config
│   ├── extractor.jl               # GNN state extractor selection
│   ├── losses.jl                  # Loss function dispatch (lstar, lgbfs, etc.)
│   ├── optimisers.jl              # Optimizer construction
│   ├── train.jl                   # Training loop (supervised + bootstrapped)
│   ├── forward.jl                 # ForwardPlannerX (A*/GBFS base implementation)
│   ├── levin_tree_search.jl       # Levin Tree Search
│   ├── planners.jl                # Planner config structs
│   ├── logging.jl                 # TensorBoard + file logging
│   └── searchtree_utils.jl        # Search tree post-processing utilities
├── imitation/                     # Imitation learning experiments
│   ├── run_experiment.jl          # SLURM entry point
│   ├── run_and_show.jl            # Config generation + results analysis
│   ├── experiment.json            # Example config
│   └── files.jl                  # Path utilities
└── selflearning.jl/               # Self-learning experiments
    ├── experiment.jl              # SLURM entry point
    └── run_and_show.jl            # Config generation + results analysis
```

---

## Configuration System

Uses `Parameters.jl` with the `@confdef` macro. All configs are JSON-serialisable.

**Key config structs** (from `configs.jl`):
```julia
@confdef struct Loss
    name::String = "lstar"   # "lstar" | "lgbfs" | "lgbfssoftmax" | ...
    γ::Float32 = 0.99
end

@confdef struct Model
    type::String = "ObjectAtom"  # GNN architecture
    # ... layer sizes, pooling type, etc.
end

@confdef struct Planner
    type::String = "GBFS"    # "GBFS" | "Astar" | "LevinTS"
    max_nodes::Int = 10_000
end
```

**Adding a new loss** (DIME): add `"dime"` to the dispatch in `losses.jl`, OR override
from this package:
```julia
# In DIMEPlanning/src/dime_loss.jl
function NeuroPlannerExperiments.materialize(l::Loss)
    if l.name == "dime"
        return DimeLoss(l.λ)
    end
    invoke(NeuroPlannerExperiments.materialize, Tuple{Loss}, l)
end
```

---

## Experiment Workflow

### 1. Generate configs

`run_and_show.jl` creates JSON config files in `results/confs/{id}.json`.
Each config specifies: domain, architecture, loss, seed, planner, hyperparameters.

```julia
# Typical config generation loop:
for domain in domains, arch in architectures, loss in losses, seed in 1:5
    config = merge(base_config, (domain=domain, arch=arch, loss=loss, seed=seed))
    write("results/confs/$(next_id).json", json(config))
end
```

### 2. Submit SLURM array job

```bash
sbatch --array=1-N run_experiment.jl
```

Each task reads its config, trains the model, evaluates on test problems, saves results.

### 3. Analyse results

`run_and_show.jl` contains `list_files()` which aggregates `.jls` result files into a
DataFrame for plotting.

---

## Planning Domains (IPC 2023)

10 domains, each with training and test splits (easy/medium/hard):

| Domain | Difficulty | Notes |
|---|---|---|
| Blocksworld | Easy-Medium | Good first test |
| Ferry | Easy | Fast to run |
| Rovers | Medium | Moderate branching |
| Satellite | Medium | Moderate |
| Spanner | Medium | Linear structure |
| Miconic | Medium | Elevator-like |
| Childsnack | Hard | Complex dependencies |
| Floortile | Hard | Narrow optimal paths |
| Sokoban | Hard | Dead ends, CMI expected to help most |
| Transport | Hard | Large state space |

**Download**: triggered automatically by `datasets.jl` from Google Drive.
Run once on the cluster before submitting experiments.

---

## GNN Architectures

| Name | Description | Recommended for |
|---|---|---|
| `ObjectAtom` | Object + atom bipartite graph | General purpose |
| `AtomBinaryFE` | Atom-to-atom binary features | Fast, good baseline |
| `AtomBinaryME` | Multiple edge types | Better for relational domains |
| `ObjectBinaryFE` | Object-centric | Simple domains |

Start with `AtomBinaryFE` for fast iteration, use `ObjectAtom` for final experiments.

---

## How to Add DIME to the Pipeline

### Step 1: New loss module

Create `DIMEPlanning/src/dime_loss.jl`:
- `DimeLoss` struct holding `λ::Float32` and two network heads
- `dime_minibatch_constructor(pddld, domain, problem, plan)` — builds training batch
  with labels `y_s` and CMI proxy `Δ(s)` from the search trace
- `(loss::DimeLoss)(model, batch)` — computes `L_pred + α * L_value`

### Step 2: New planner priority

Create `DIMEPlanning/src/dime_planner.jl`:
- Override the heuristic evaluation in `ForwardPlannerX` to use `a(s) = log f(s) + λ v(s)`
- `DimePlanner` config struct extending the existing `Planner` struct
- Requires both `f` (predictor head) and `v` (value head) from the shared GNN

### Step 3: Experiment configs

Copy `NeuroPlannerExperiments.jl/imitation/run_and_show.jl` to
`DIMEPlanning/experiments/imitation/run_and_show.jl` and add DIME configs:
```julia
losses = ["lstar", "lgbfs", "dime"]  # dime is new
lambdas = [0.1, 0.5, 1.0, 2.0]      # sweep λ
```

### Step 4: SLURM entry point

Copy `NeuroPlannerExperiments.jl/imitation/run_experiment.jl` and update:
- Header paths (replace `/home/pevnytom/` with your cluster paths)
- `using DIMEPlanning` instead of (or in addition to) `NeuroPlannerExperiments`

---

## SLURM Configuration

Current scripts have hardcoded paths for `pevnytom`'s cluster. Update these before use:

In `run_experiment.jl`:
```
#SBATCH --error=/home/pevnytom/logs/pddl.%j.err   ← CHANGE to your path
#SBATCH -p cpuextralong                             ← CHANGE to your partition
julia-1.12.1                                        ← CHANGE to available Julia version
```

Typical resource requirements:
- Imitation learning: 32GB RAM, cpulong (8-24h)
- Self-learning: 64GB RAM, cpulong (24-48h)
- Per-domain, per-seed jobs are independent — array jobs with N = domains × architectures × losses × seeds

---

## Results Format

Results stored as Julia serialised objects (`.jls`):

```text
results/{domain}/{arch}_{loss}_{seed}_{hash}/
├── model.jls                   # Trained Flux model
├── GBFS_{hash}.jls             # Evaluation results (DataFrame)
└── configuration_{hash}.jls   # Config metadata
```

DataFrame columns: `solution_time`, `sol_length`, `expanded`, `generated`, `solved`,
`used_in_train`, `problem_file`.

Primary metric: `solved` (Boolean). Secondary: `expanded` (efficiency).
