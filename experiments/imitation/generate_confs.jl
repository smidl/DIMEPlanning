using NeuroPlannerExperiments
include(joinpath(@__DIR__, "files.jl"))
include(joinpath(@__DIR__, "config.jl"))

mkpath(confs_dir)

confs = Iterators.product(datasets, models, extractors, losses, trains, planners, seeds)
n = 0
for (i, (dataset, model, extractor, loss, train, planner, seed)) in enumerate(confs)
    conf = create_config(dataset, model, extractor, loss, train, planner, seed)
    write_config(joinpath(confs_dir, "$(i).json"), conf)
    n += 1
end
println("Written $n configs to $confs_dir")
