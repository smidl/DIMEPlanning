using NeuroPlannerExperiments
using NeuroPlannerExperiments.NeuroPlanner.StatsBase
using DataFrames
using Serialization

include(joinpath(@__DIR__, "files.jl"))

prepath = joinpath(ENV["HOME"], "DIMEPlanning", "results")

function best_models(odir=prepath)
    problem_dirs = filter(s -> !contains(s, "confs"), readdir(odir, join = true))
    for pd in problem_dirs
        map(readdir(pd, join = true)) do ed
            try
                stat_files = filter(f -> contains(f, "AStar"), readdir(ed))
                isempty(stat_files) && return(missing)
                stat_file = only(stat_files)
                df = deserialize(joinpath(ed, stat_file))
            catch me
                @show me
            end
        end
    end
end

function list_files(odir=prepath)
    problem_dirs = filter(s -> !contains(s, "confs"), readdir(odir, join = true))
    isempty(problem_dirs) && return DataFrame()
    exp_dirs = mapreduce(pd -> readdir(pd, join = true), vcat, problem_dirs; init=String[])

    experiments = mapreduce(vcat, exp_dirs) do ed
        fs = readdir(ed)
        fs = filter(s -> (startswith(s, "AStar") && !endswith(s,".tmp")), fs)
        isempty(fs) && String[]
        return(map(s -> joinpath(ed,s), fs))
    end
    xs = map(NeuroPlannerExperiments.IPC_PROBLEMS) do d
        sub_files  = filter(f -> contains(f, d[7:end]), experiments)
        isempty(sub_files) && return(missing)
        sl = filter(!isnothing, map(sub_files) do ef
            df, conf = deserialize(ef)
            sub_df = filter(s -> contains(s.problem_file,"testing"),df)
            isempty(sub_df) || !hasproperty(sub_df, :solved) ? nothing : mean(sub_df.solved)
        end)
        isempty(sl) && return(missing)
        mx, mn, sd = maximum(sl), mean(sl), std(sl)
        println(d,"  ",(mx, mn, sd)," (",length(sub_files),")")
        (;dataset = d, best = mx, mean = mn, std = sd, finished = length(sub_files))
    end
    rows = collect(skipmissing(xs))
    isempty(rows) ? DataFrame() : DataFrame(rows)
end


datasets = [ISPCDataset("ipc23_ferry")]
archs = ["atombinaryfe"]
extractors = [Extractor(;architecture=a, graph_layers=l) for (a,l) in Iterators.product(archs, [2])]

message_blocks = [FFNN(hidden_dim = h, output_dim = h, layers = 1, layernorm = ln) for (h,ln) in Iterators.product([32], [true])]
pooled_blocks = [FFNN(hidden_dim = h, output_dim = 1, layers = l, layernorm = ln) for (h,l,ln) in Iterators.product([32], [2], [true])]

models = [Model(message_pass_model = mp, pooled_model = o) for (mp,o) in Iterators.product(message_blocks, pooled_blocks)]
losses = [Loss(s) for s in ["lstar", "lgbfs"]]
trains = [SupervisedTraining(;max_epoch = 10)]
planners = [AStar(max_time = 30)]
seeds = [1, 2, 3]

confs_dir = joinpath(prepath, "confs")
mkpath(confs_dir)

confs = Iterators.product(datasets, models, extractors, losses, trains, planners, seeds)
for (i, (dataset, model, extractor, loss, train, planner, seed)) in enumerate(confs)
    conf = create_config(dataset, model, extractor, loss, train, planner, seed)
    write_config(joinpath(confs_dir, "$(i).json"), conf)
end

# Check progress
files = map(readdir(confs_dir; join=true, sort=true)) do path
    c = load_config(path)
    ofile = result_file(c, prepath)
    (;path, stats = ofile, tmp = ofile*".tmp", model = model_file(c, prepath))
end

println("valid results")
for d in NeuroPlannerExperiments.IPC_PROBLEMS
    sub_files  = filter(f -> contains(f.stats, d[7:end]), files)
    status = map(sub_files) do c
        isfile(c.stats) && return(:done)
        isfile(c.tmp) && return(:evaluating)
        isfile(c.model) && return(:trained)
        return(:notdone)
    end |> countmap
    println(d)
    display(status)
    println()
end

k = length(readdir(confs_dir))
batch_start = 1
while true
    global batch_start
    if (length(readlines(`squeue -u $(ENV["USER"])`)) < 80) && batch_start < k
        batch_stop = min(k, batch_start+99)
        try
            cmd = `sbatch --array=$(batch_start)-$(batch_stop) run_experiment.jl`
            run(cmd)
            batch_start += 100
            println("issuing new jobs: ", batch_start)
        catch e
            println("sbatch failed: ", e)
        end
    end
    display(list_files())
    sleep(3600)
end
