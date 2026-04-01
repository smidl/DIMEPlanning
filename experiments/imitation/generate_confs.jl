using NeuroPlannerExperiments
include(joinpath(@__DIR__, "files.jl"))
include(joinpath(@__DIR__, "config.jl"))

mkpath(confs_dir)

confs = collect(Iterators.product(datasets, models, extractors, losses, trains, planners, seeds))
for (i, (dataset, model, extractor, loss, train, planner, seed)) in enumerate(confs)
    conf = create_config(dataset, model, extractor, loss, train, planner, seed)
    write_config(joinpath(confs_dir, "$(i).json"), conf)
end
println("Written $(length(confs)) configs to $confs_dir")
