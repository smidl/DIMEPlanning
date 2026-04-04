# Path helpers for DIME experiments — same interface as files.jl.
# Model is shared across planner variants with the same (dataset, extractor, model, seed).
# The model hash intentionally excludes the planner so different planners share one model.

function result_dir(conf::NamedTuple, prepath=nothing, create_dir=false)
    a = lowercase(conf.dataset.name)
    b = lowercase(conf.extractor.architecture)
    s = string(conf.seed)
    h = hash(map(k -> conf[k], (:dataset, :extractor, :model)))
    path = joinpath("dime", a, join(string.([b, "dime", s, h]), "_"))
    path = prepath === nothing ? path : joinpath(prepath, path)
    create_dir && !isdir(path) && mkpath(path)
    path
end

model_file(conf::NamedTuple, prepath=nothing, create_dir=false) =
    joinpath(result_dir(conf, prepath, create_dir), "model.jls")

result_file(conf::NamedTuple, prepath=nothing, create_dir=false) =
    joinpath(result_dir(conf, prepath, create_dir),
             string(string(conf.planner), "_", hash(conf.planner), ".jls"))

conf_file(conf::NamedTuple, prepath=nothing, create_dir=false) =
    joinpath(result_dir(conf, prepath, create_dir),
             string("configuration_", hash(conf.planner), ".jls"))
