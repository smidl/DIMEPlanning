using NeuroPlannerExperiments
using DIMEPlanning

# Edit this file to change the DIME experiment sweep.
# Included by dime_generate_confs.jl and run_dime_experiment.jl.

prepath   = joinpath(ENV["HOME"], "DIMEPlanning", "results")
confs_dir = joinpath(prepath, "dime_confs")

datasets = [
    ISPCDataset("ipc23_ferry"),
    ISPCDataset("ipc23_blocksworld"),
    ISPCDataset("ipc23_sokoban"),
]

archs      = ["atombinaryfe"]
extractors = [Extractor(; architecture=a, graph_layers=l)
              for (a, l) in Iterators.product(archs, [2])]

message_blocks = [FFNN(hidden_dim=h, output_dim=h, layers=1, layernorm=ln)
                  for (h, ln) in Iterators.product([32], [true])]
pooled_blocks  = [FFNN(hidden_dim=h, output_dim=1, layers=l, layernorm=ln)
                  for (h, l, ln) in Iterators.product([32], [2], [true])]
models = [Model(message_pass_model=mp, pooled_model=o)
          for (mp, o) in Iterators.product(message_blocks, pooled_blocks)]

trains = [SupervisedTraining(; max_epoch=200)]
seeds  = [1, 2, 3]

# One planner per config — same structure as standard experiment sweep.
# Variants:
#   λ=0 freeze  — context-free (≡ lgbfs ranking), diagnostic baseline
#   λ=0.1       — mild CMI exploration (additive UCB-style)
#   λ=0.5       — stronger CMI exploration
#   product     — EI-style β-free: a = f / (1 + max(v,0))
planners = [
    DIMEPlanner(max_time=30, λ=0.0f0, acquisition="additive", freeze_context=true),
    DIMEPlanner(max_time=30, λ=0.1f0, acquisition="additive", freeze_context=false),
    DIMEPlanner(max_time=30, λ=0.5f0, acquisition="additive", freeze_context=false),
    DIMEPlanner(max_time=30, λ=0.0f0, acquisition="product",  freeze_context=false),
]
