using NeuroPlannerExperiments
using NeuroPlannerExperiments.NeuroPlanner.StatsBase
using DataFrames, Serialization, Statistics
include(joinpath(@__DIR__, "files.jl"))
include(joinpath(@__DIR__, "config.jl"))

function job_status()
    isdir(confs_dir) || return println("No confs dir yet — run generate_confs.jl first")
    files = map(readdir(confs_dir; join=true, sort=true)) do path
        c = load_config(path)
        ofile = result_file(c, prepath)
        (; path, stats=ofile, tmp=ofile*".tmp", model=model_file(c, prepath))
    end
    println("Job status:")
    for d in NeuroPlannerExperiments.IPC_PROBLEMS
        sub = filter(f -> contains(f.stats, d[7:end]), files)
        isempty(sub) && continue
        status = map(sub) do c
            isfile(c.stats) && return :done
            isfile(c.tmp)   && return :evaluating
            isfile(c.model) && return :trained
            return :notdone
        end |> countmap
        println("  $d  $status")
    end
end

function show_results(odir=prepath)
    problem_dirs = filter(s -> !contains(s, "confs"), readdir(odir, join=true))
    isempty(problem_dirs) && return println("No results yet.")
    exp_dirs = mapreduce(pd -> readdir(pd, join=true), vcat, problem_dirs; init=String[])

    experiments = mapreduce(vcat, exp_dirs; init=String[]) do ed
        fs = filter(s -> startswith(s, "AStar") && !endswith(s, ".tmp"), readdir(ed))
        map(s -> joinpath(ed, s), fs)
    end

    xs = map(NeuroPlannerExperiments.IPC_PROBLEMS) do d
        sub = filter(f -> contains(f, d[7:end]), experiments)
        isempty(sub) && return missing
        sl = filter(!isnothing, map(sub) do ef
            r = deserialize(ef)
            df = r isa NamedTuple ? r.stats : r[1]
            isempty(df) || !hasproperty(df, :problem_file) || !hasproperty(df, :solved) && return nothing
            sub_df = filter(s -> contains(s.problem_file, "testing"), df)
            isempty(sub_df) ? nothing : mean(sub_df.solved)
        end)
        isempty(sl) && return missing
        (; dataset=d, best=maximum(sl), mean=mean(sl), std=std(sl), n=length(sl))
    end
    rows = collect(skipmissing(xs))
    isempty(rows) ? println("No completed results yet.") : display(DataFrame(rows))
end

job_status()
println()
show_results()
