using NeuroPlannerExperiments
include(joinpath(@__DIR__, "files.jl"))
include(joinpath(@__DIR__, "config.jl"))

isdir(confs_dir) || error("No confs dir found — run generate_confs.jl first")

println("Starting submission loop. Ctrl+C to stop.")
while true
    files = map(readdir(confs_dir; join=true, sort=true)) do path
        c = load_config(path)
        ofile = result_file(c, prepath)
        (; path, stats=ofile, tmp=ofile*".tmp", model=model_file(c, prepath))
    end
    pending = findall(c -> !isfile(c.stats), files)

    if isempty(pending)
        println("All $(length(files)) jobs done.")
        break
    end

    queued = length(readlines(`squeue -u $(ENV["USER"])`)) - 1  # subtract header
    println("$(length(files) - length(pending))/$(length(files)) done, $(length(pending)) pending, $queued in queue")

    if queued < 80
        batch = pending[1:min(100, end)]
        indices = join(batch, ",")
        try
            run(`sbatch --array=$(indices) run_experiment.jl`)
            println("Submitted indices $(first(batch))–$(last(batch))")
        catch e
            println("sbatch failed: $e")
        end
    end

    sleep(3600)
end
