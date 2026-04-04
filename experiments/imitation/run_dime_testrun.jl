"""
DIME end-to-end testrun on the ferry domain.

Trains DIMEModel for a few epochs then evaluates DIMEHeuristic (λ=0.5)
against the lgbfs baseline on the same test set.  Prints a side-by-side
solve-rate summary and per-problem timing.

Run locally:
    julia --project=. experiments/imitation/run_dime_testrun.jl

Takes ~10-20 min on a laptop (training dominates; eval is fast on easy split).
"""

using DIMEPlanning
using NeuroPlannerExperiments
using NeuroPlannerExperiments.Optimisers
using NeuroPlanner
using NeuroPlanner.PDDL
using DataFrames
using Serialization
using Random
using Statistics

Random.seed!(42)

# ---- paths ------------------------------------------------------------------
DATA      = joinpath(@__DIR__, "..", "..", "NeuroPlannerExperiments.jl",
                     "data", "ipc23_learning", "ferry")
DOMAIN_FILE = joinpath(DATA, "domain.pddl")

domain  = load_domain(DOMAIN_FILE)
dataset = materialize(ISPCDataset("ipc23_ferry"))

# Use easy test split only to keep runtime short
test_files = filter(contains("testing/easy"), dataset.problem_files)
println("Training problems : $(length(dataset.train_files))")
println("Test problems     : $(length(test_files))")

# ---- extractor + minibatches ------------------------------------------------
pddld = materialize(Extractor(; architecture="atombinaryfe", graph_layers=2), domain)

println("\nBuilding DIME minibatches …")
dime_samples = construct_minibatches(pddld, DIMEMiniBatch, dataset)
println("  $(length(dime_samples)) minibatches ready")

# ---- model config -----------------------------------------------------------
conf = Model(
    message_pass_model = FFNN(hidden_dim=32, output_dim=32, layers=1, layernorm=true),
    pooling = "SegmentedSumMax",
    pooled_model = FFNN(hidden_dim=32, output_dim=1, layers=2, layernorm=true)
)

dime_model  = construct_dime_model(pddld, dataset, conf)
lgbfs_model = construct_model(pddld, dataset, conf)

# ---- build lgbfs minibatches for baseline training -------------------------
println("\nBuilding lgbfs minibatches …")
lgbfs_samples = construct_minibatches(pddld, materialize(Loss("lgbfs")), dataset)

# ---- training ---------------------------------------------------------------
train_conf = SupervisedTraining(; max_epoch=50)

println("\nTraining lgbfs baseline …")
opt = NeuroPlannerExperiments.materialize(NeuroPlannerExperiments.OptADAM())
st_lgbfs = Optimisers.setup(opt, lgbfs_model)
lgbfs_model, _ = train(lgbfs_model, lgbfs_samples, (opt, st_lgbfs), train_conf)

# Option 1: train DIME with empty context (same signal as lgbfs).
# This eliminates train/test context distribution shift.
# At inference the model still receives the growing context — testing whether
# a context-free predictor can exploit context it was never trained with.
println("\nTraining DIME model (Option 1: empty context during training) …")
dime_samples_noctx = strip_context.(dime_samples)
opt2 = NeuroPlannerExperiments.materialize(NeuroPlannerExperiments.OptADAM())
st_dime = Optimisers.setup(opt2, dime_model)
dime_model, _ = train(dime_model, dime_samples_noctx, (opt2, st_dime), train_conf)

# ---- evaluation -------------------------------------------------------------
astar_conf = AStar(max_time=30)

println("\nEvaluating lgbfs baseline on $(length(test_files)) test problems …")
lgbfs_stats = evaluate_heuristic(pddld, lgbfs_model, astar_conf, test_files)

# Evaluate DIME at three λ values to understand the tradeoff
dime_results = map([0.0f0, 0.1f0, 0.5f0]) do λ
    planner_conf = DIMEPlanner(max_time=30, λ=λ)
    println("\nEvaluating DIME (λ=$λ) on $(length(test_files)) test problems …")
    stats = evaluate_heuristic(pddld, dime_model, planner_conf, test_files)
    (λ=λ, freeze=false, stats=stats)
end

# Diagnostic: freeze_context=true — context-free inference with DIME f-head
# If this matches lgbfs, the growing context at test time is what hurts performance.
println("\nEvaluating DIME (λ=0, freeze_context=true) …")
dime_frozen = let
    planner_conf = DIMEPlanner(max_time=30, λ=0.0f0, freeze_context=true)
    stats = evaluate_heuristic(pddld, dime_model, planner_conf, test_files)
    (λ=0.0f0, freeze=true, stats=stats)
end

# ---- summary ----------------------------------------------------------------
function summarise(df, label)
    if isempty(df) || !hasproperty(df, :solved)
        println("  $label: no results")
        return
    end
    n      = nrow(df)
    solved = sum(df.solved)
    t_med  = median(filter(isfinite, df.solution_time))
    exp_med = median(filter(isfinite, df.expanded))
    println("  $label: solved $solved/$n  |  median time $(round(t_med, digits=2))s  |  median expanded $(round(Int, exp_med))")
end

println("\n========== Results ==========")
summarise(lgbfs_stats, "lgbfs (baseline)")
for r in dime_results
    summarise(r.stats, "DIME  λ=$(r.λ)")
end
summarise(dime_frozen.stats, "DIME  λ=0 freeze_ctx")

# Per-problem comparison (lgbfs vs all λ values)
if !isempty(lgbfs_stats) && hasproperty(lgbfs_stats, :problem_file)
    println("\nPer-problem expanded nodes (solved? / n_expanded):")
    all_results = [dime_results; [dime_frozen]]
    header = rpad("problem", 22) * rpad("lgbfs", 14) *
             join([rpad(r.freeze ? "DIME freeze" : "DIME λ=$(r.λ)", 14) for r in all_results])
    println(header)
    for pf in sort(unique(lgbfs_stats.problem_file))
        rb = only(filter(row -> row.problem_file == pf, eachrow(lgbfs_stats)))
        row = rpad(basename(pf), 22) * rpad("$(rb.solved)/$(rb.expanded)", 14)
        for r in all_results
            if hasproperty(r.stats, :problem_file)
                rd_rows = filter(rr -> rr.problem_file == pf, eachrow(r.stats))
                if isempty(rd_rows)
                    row *= rpad("—", 14)
                else
                    rd = only(rd_rows)
                    row *= rpad("$(rd.solved)/$(rd.expanded)", 14)
                end
            end
        end
        println(row)
    end
end
