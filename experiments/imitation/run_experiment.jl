#!/usr/bin/env sh
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH -p cpulong
#SBATCH --error=/home/%u/logs/pddl.%j.err
#SBATCH --out=/home/%u/logs/pddl.%j.out
#=
export DATADEPS_ALWAYS_ACCEPT=true
export PATH="${HOME}/.julia/juliaup/bin:${PATH}"

REPO="${HOME}/DIMEPlanning"
srun --unbuffered -D "${REPO}/experiments/imitation" \
    julia --project="${REPO}/NeuroPlannerExperiments.jl" \
    --color=no --startup-file=no --threads=auto \
    "${BASH_SOURCE[0]}" "$@"
exit
=#

# parse args
id = parse(Int, ENV["SLURM_ARRAY_TASK_ID"])
prepath = joinpath(ENV["HOME"], "DIMEPlanning", "results")
path = readdir(joinpath(prepath, "confs"); join=true, sort=true)[id]

s = rand(1:60)
@info """
Configuration:
⋅ File: $(basename(path))
⋅ Dir: $(dirname(path))
⋅ Sleep: $(s)s
⋅ Exec dir: $(pwd())
"""
sleep(s)

using NeuroPlannerExperiments
using NeuroPlannerExperiments.Optimisers
using NeuroPlanner
using DataFrames
using PDDL
using Serialization
using Random
using Statistics
include(joinpath(ENV["HOME"], "DIMEPlanning", "experiments", "imitation", "files.jl"))

@info "parsing conf"
conf = load_config(path)
@info conf

seed = conf.seed
Random.seed!(seed)
dataset = materialize(conf.dataset)
domain = load_domain(dataset.domain_file)
pddld = materialize(conf.extractor, domain)

modelfile = model_file(conf, prepath)
!isdir(result_dir(conf, prepath)) && mkpath(result_dir(conf, prepath))
@info "preparing model"
model = if isfile(modelfile)
    @info "Model exists, deserializing"
    deserialize(modelfile)[1]
else
    model = construct_model(pddld, dataset, conf.model)
    samples = construct_minibatches(pddld, materialize(conf.loss), dataset)
    optimiser = NeuroPlannerExperiments.materialize(conf.training.optimiser)
    state_tree = Optimisers.setup(optimiser, model)
    model, _ = train(model, samples, (optimiser, state_tree), conf.training)
    serialize(modelfile, (;model, conf))
    model
end

rfile = result_file(conf, prepath)
@info "Evaluating starting"
stats = evaluate_heuristic(pddld, model, conf.planner, dataset, rfile)
stats = stats isa NamedTuple ? stats.stats : stats
serialize(rfile, (;stats, conf))
write_config(conf_file(conf, prepath), conf)
@info "Writing file to $(rfile)"
