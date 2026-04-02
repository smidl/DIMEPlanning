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
train_conf = SupervisedTraining(; max_epoch=5)

println("\nTraining lgbfs baseline …")
opt = NeuroPlannerExperiments.materialize(NeuroPlannerExperiments.OptADAM())
st_lgbfs = Optimisers.setup(opt, lgbfs_model)
lgbfs_model, _ = train(lgbfs_model, lgbfs_samples, (opt, st_lgbfs), train_conf)

println("\nTraining DIME model …")
opt2 = NeuroPlannerExperiments.materialize(NeuroPlannerExperiments.OptADAM())
st_dime = Optimisers.setup(opt2, dime_model)
dime_model, _ = train(dime_model, dime_samples, (opt2, st_dime), train_conf)

# ---- evaluation -------------------------------------------------------------
planner_conf = DIMEPlanner(max_time=30, λ=0.5f0)
astar_conf   = AStar(max_time=30)

println("\nEvaluating lgbfs baseline on $(length(test_files)) test problems …")
lgbfs_stats = evaluate_heuristic(pddld, lgbfs_model, astar_conf, test_files)

println("\nEvaluating DIME (λ=$(planner_conf.λ)) on $(length(test_files)) test problems …")
dime_stats  = evaluate_heuristic(pddld, dime_model,  planner_conf, test_files)

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
summarise(dime_stats,  "DIME  λ=$(planner_conf.λ)")

# Per-problem comparison
if !isempty(lgbfs_stats) && !isempty(dime_stats) &&
   hasproperty(lgbfs_stats, :problem_file) && hasproperty(dime_stats, :problem_file)
    println("\nPer-problem (solved / expanded):")
    println(rpad("problem", 30), rpad("lgbfs", 18), "DIME")
    for pf in sort(unique(lgbfs_stats.problem_file))
        rb = only(filter(r -> r.problem_file == pf, eachrow(lgbfs_stats)))
        rd_rows = filter(r -> r.problem_file == pf, eachrow(dime_stats))
        if isempty(rd_rows)
            println(rpad(basename(pf), 30), rpad("$(rb.solved)/$(rb.expanded)", 18), "—")
        else
            rd = only(rd_rows)
            println(rpad(basename(pf), 30),
                    rpad("$(rb.solved)/$(rb.expanded)", 18),
                    "$(rd.solved)/$(rd.expanded)")
        end
    end
end
