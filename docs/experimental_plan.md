# Experimental Plan: DIME-Planning

## Goal

Demonstrate that CMI-guided node expansion (`λ > 0`) outperforms purely heuristic-driven
planners on IPC 2023 benchmarks, with GBFS as the search algorithm.

**Primary claim**: `lgbfs + v(s)` (DIME with λ > 0) solves more problems, especially on
hard domains with narrow optimal paths, compared to `lgbfs` (λ=0) and `lstar`.

---

## Domains

All 10 IPC 2023 Learning Track domains:

| Domain | Difficulty | Why included |
|---|---|---|
| Ferry | Easy | Sanity check — all methods should solve |
| Blocksworld | Easy-Medium | Standard benchmark, good first validation |
| Rovers | Medium | Moderate branching factor |
| Satellite | Medium | Moderate, different from Rovers |
| Spanner | Medium | Linear structure — predictor should converge fast |
| Miconic | Medium | Elevator-like, well-structured |
| Childsnack | Hard | Complex object dependencies |
| Floortile | Hard | Narrow optimal paths — DIME advantage expected |
| Sokoban | Hard | Dead ends — CMI exploration most valuable here |
| Transport | Hard | Large state space, diverse solutions |

**Start with**: Ferry + Blocksworld for fast iteration and debugging.
**Key domains for paper story**: Sokoban and Floortile (hard, narrow paths, most benefit from CMI).

---

## Baselines

All baselines are already implemented in `NeuroPlannerExperiments.jl`:

| Method | Description | Status |
|---|---|---|
| `lstar` | Regression to h* (cost-to-go) | Available |
| `lgbfs` | Ranking loss; equivalent to DIME λ=0 | Available |
| `lgbfssoftmax` | Softmax variant of lgbfs | Available |
| `DIME (λ=0.1)` | CMI exploration, low weight | **New** |
| `DIME (λ=0.5)` | CMI exploration, medium weight | **New** |
| `DIME (λ=1.0)` | CMI exploration, default | **New** |
| `DIME (λ=2.0)` | CMI exploration, high weight | **New** |

Note: `lgbfs` IS `DIME (λ=0)` — this is not an approximation, it is an exact correspondence
(up to BCE vs margin loss). This makes the comparison extremely clean.

---

## Hyperparameter Grid

### Phase 1: Sweep (to find working configuration)

```
domains      = ["ferry", "blocksworld", "rovers"]
architectures = ["AtomBinaryFE", "ObjectAtom"]
losses       = ["lstar", "lgbfs", "dime"]
lambdas      = [0.1, 0.5, 1.0, 2.0]   # only for "dime"
alpha        = [0.1, 1.0]              # weight of L_value vs L_pred
seeds        = [1, 2, 3]
```

Total jobs (phase 1): 3 × 2 × (2 + 4) × 2 × 3 = 216

### Phase 2: Full sweep (after phase 1 finds working λ)

```
domains      = all 10 IPC domains
architectures = ["AtomBinaryFE", "ObjectAtom"]
losses       = ["lstar", "lgbfs", "dime"]
lambdas      = [best_λ, ±1 neighbor]   # 3 values
alpha        = [best_α]                # 1 fixed value
seeds        = [1, 2, 3, 4, 5]
```

Total jobs (phase 2): 10 × 2 × (2 + 3) × 5 = 500

---

## Metrics

### Primary
- **Solve rate**: fraction of test problems solved within `max_nodes` expansions
  - Reported per domain × difficulty (easy/medium/hard split)
  - Aggregated: macro-average across domains

### Secondary
- **Nodes expanded**: number of GBFS expansions until solution (conditional on solved)
  - Lower is better — measures search efficiency
- **Solution length**: length of returned plan
  - Lower is better (optimality proxy)

### DataFrame columns (from NeuroPlannerExperiments.jl results)
```
solution_time, sol_length, expanded, generated, solved, used_in_train, problem_file
```

### Reporting
- Table: solve rate per domain, per method (main result table)
- Plot: solve rate vs λ (ablation plot showing λ=0 → DIME trade-off)
- Plot: nodes expanded vs domain difficulty (efficiency plot)
- Plot: CMI proxy Δ(s) vs v(s) Spearman ρ (diagnostic — validates DIME Lemma 1)

---

## SLURM Configuration

### Resource requirements

| Phase | RAM | Partition | Wall time |
|---|---|---|---|
| Imitation (phase 1) | 32 GB | cpulong | 8–24h |
| Imitation (phase 2) | 32 GB | cpulong | 8–24h |
| Self-learning | 64 GB | cpulong | 24–48h |

### Array job structure

Each SLURM task = one (domain, architecture, loss, λ, seed) combination.
Tasks are independent — no communication needed.

```bash
# Submit phase 1 (example for 216 jobs):
sbatch --array=1-216 experiments/imitation/run_experiment.jl
```

### Paths to update in run_experiment.jl

The template from `NeuroPlannerExperiments.jl/imitation/run_experiment.jl` has
hardcoded paths for `pevnytom`'s cluster. Update:

```
#SBATCH --error=/home/YOUR_USERNAME/logs/dime.%j.err
#SBATCH --output=/home/YOUR_USERNAME/logs/dime.%j.out
#SBATCH -p YOUR_PARTITION          # e.g., cpulong, gpu, amdlong
julia-1.12.1                       # or whichever Julia is available
```

---

## Validation Protocol

### Step 1: Single-domain sanity check (before full sweep)

Run on Ferry (easiest) with 1 seed:
```julia
# Expected: all methods solve >90% of easy problems
# If DIME fails here, something is wrong with implementation
```

### Step 2: CMI diagnostic (validates DIME Lemma 1 in planning)

For a fixed trained model on Blocksworld, compute:
```
Spearman ρ( v(s), Δ(s) )  over all expanded states in the test set
```
Target: ρ > 0.5 (from MCTS-DIME Exp1, ρ = +0.93 was achievable)

If ρ is low, the value network is not learning the CMI signal — debug before continuing.

### Step 3: Easy vs hard domain comparison

On Sokoban and Floortile (hard), confirm:
- `lgbfs` (λ=0) struggles with dead ends
- `DIME` (λ > 0) shows higher solve rate

If DIME does NOT help on hard domains, investigate:
1. Is v(s) trained correctly? Check ρ diagnostic
2. Is λ too high/low? Run finer grid
3. Is the backbone GNN too weak? Try ObjectAtom instead of AtomBinaryFE
4. Consider λ annealing (see `docs/lessons_learned.md`)

---

## Self-Learning Phase (after imitation works)

Once imitation learning shows DIME advantage, extend to bootstrapped self-play:

1. Train on initial dataset of solved problems (imitation)
2. Run DIME planner on harder problems (above imitation training difficulty)
3. Add newly solved problems to training set
4. Repeat

This mirrors the self-learning pipeline in `NeuroPlannerExperiments.jl/selflearning.jl/`.
Only attempt this after imitation results are solid.

---

## Expected Timeline

1. **Week 1–2**: Implement DimeLoss and DimePlanner (see `docs/infrastructure.md`)
2. **Week 3**: Single-domain validation (Ferry, Blocksworld), CMI diagnostic
3. **Week 4**: Phase 1 sweep on 3 domains, identify best λ/α
4. **Week 5–6**: Phase 2 full sweep, collect results
5. **Week 7**: Analysis, plotting, paper writing begins

---

## Open Questions Before Starting

1. **α (L_value weight)**: Try α ∈ {0.1, 1.0}. If α too large, value loss dominates and
   the predictor underfits. Start with α=0.1.

2. **Δ(s) approximation**: Computing exact per-expansion loss reduction is expensive.
   Approximation: use change in predictor logit as a proxy. See `CLAUDE.md` Section on
   Δ(s) computation.

3. **Training problem difficulty**: Train on easy problems first (like LEAH does), then
   evaluate on easy/medium/hard. Do NOT train on hard problems initially.

4. **Max nodes at test**: Use same `max_nodes=10000` as NeuroPlannerExperiments.jl
   baseline experiments for fair comparison.
