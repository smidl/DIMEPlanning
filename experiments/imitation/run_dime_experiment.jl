#!/usr/bin/env sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p cpulong
#SBATCH --error=/home/%u/logs/dime.%j.err
#SBATCH --out=/home/%u/logs/dime.%j.out
#=
export DATADEPS_ALWAYS_ACCEPT=true
export PATH="${HOME}/.julia/juliaup/bin:${PATH}"

REPO="${HOME}/DIMEPlanning"
srun --unbuffered -D "${REPO}/experiments/imitation" \
    julia --project="${REPO}" \
    --color=no --startup-file=no --threads=auto \
    "${BASH_SOURCE[0]}" "$@"
exit
=#

id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
prepath   = joinpath(ENV["HOME"], "DIMEPlanning", "results")
confs_dir = joinpath(prepath, "dime_confs")
path = readdir(confs_dir; join=true, sort=true)[id]

s = rand(1:60)
@info """
DIME Configuration:
⋅ File: $(basename(path))
⋅ Sleep: $(s)s
"""
sleep(s)

using DIMEPlanning
using NeuroPlannerExperiments
using NeuroPlannerExperiments.Optimisers
using NeuroPlanner
using NeuroPlanner.PDDL
using DataFrames
using Serialization
using Random
using Statistics

include(joinpath(ENV["HOME"], "DIMEPlanning", "experiments", "imitation", "dime_files.jl"))

@info "Parsing conf"
conf = load_config(path)
conf = merge(conf, (; planner = parse_config(conf.planner)))
@info conf

seed = conf.seed
Random.seed!(seed)
dataset = materialize(conf.dataset)
domain  = load_domain(dataset.domain_file)
pddld   = materialize(conf.extractor, domain)

# ---- train (shared across planner variants with same dataset/model/seed) -----
modelfile = model_file(conf, prepath, true)

model = if isfile(modelfile)
    @info "Model exists, loading from $(modelfile)"
    deserialize(modelfile)[1]
else
    @info "Building DIMEModel and training"
    dime_model = construct_dime_model(pddld, dataset, conf.model)
    raw_samples = construct_minibatches(pddld, DIMEMiniBatch, dataset)
    samples = strip_context.(raw_samples)
    optimiser = NeuroPlannerExperiments.materialize(conf.training.optimiser)
    state_tree = Optimisers.setup(optimiser, dime_model)
    dime_model, _ = train(dime_model, samples, (optimiser, state_tree), conf.training)
    serialize(modelfile, (;model=dime_model, conf))
    dime_model
end

# ---- evaluate ----------------------------------------------------------------
rfile = result_file(conf, prepath, true)
@info "Evaluating $(conf.planner)"
stats = evaluate_heuristic(pddld, model, conf.planner, dataset, rfile)
stats = stats isa NamedTuple ? stats.stats : stats
serialize(rfile, (;stats, conf))
write_config(conf_file(conf, prepath), conf)
@info "Written to $(rfile)"
