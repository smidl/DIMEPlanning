using NeuroPlannerExperiments
using DIMEPlanning
include(joinpath(@__DIR__, "dime_config.jl"))
include(joinpath(@__DIR__, "dime_files.jl"))

mkpath(confs_dir)

confs = collect(Iterators.product(datasets, models, extractors, trains, planners, seeds))
for (i, (dataset, model, extractor, train, planner, seed)) in enumerate(confs)
    conf = (;
        dataset,
        model,
        extractor,
        training = train,
        planner,
        seed,
    )
    write_config(joinpath(confs_dir, "$(i).json"), conf)
end
println("Written $(length(confs)) configs to $confs_dir")
