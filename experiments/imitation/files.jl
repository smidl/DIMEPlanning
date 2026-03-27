function result_dir(conf::NamedTuple, prepath=nothing, create_dir = false)
    a = lowercase(conf.dataset.name)
    b = lowercase(conf.extractor.architecture)
    c = lowercase(conf.loss.name)
    s = lowercase(string(conf.seed))
    h = hash(map(k -> conf[k], (:dataset,:extractor,:loss, :model)))
    path = joinpath(a, join(string.([b,c,s,h]),"_"))
    path = prepath === nothing ? path : joinpath(prepath, path)
    if create_dir && !isdir(result_dir(conf, prepath))
        mkpath(path)
    end
    path
end

model_file(conf::NamedTuple, prepath = nothing, create_dir = false) = joinpath(result_dir(conf, prepath, create_dir), "model.jls")
result_file(conf::NamedTuple, prepath = nothing, create_dir = false) = joinpath(result_dir(conf, prepath, create_dir), string(string(conf.planner),"_",hash(conf.planner),".jls"))
conf_file(conf::NamedTuple, prepath = nothing, create_dir = false) = joinpath(result_dir(conf, prepath, create_dir), string("configuration_",hash(conf.planner),".jls"))
