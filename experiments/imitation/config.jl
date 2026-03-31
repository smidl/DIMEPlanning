using NeuroPlannerExperiments

# Edit this file to change the experiment sweep.
# All other scripts include this file — do not put logic here.

prepath  = joinpath(ENV["HOME"], "DIMEPlanning", "results")
confs_dir = joinpath(prepath, "confs")

datasets  = [ISPCDataset("ipc23_ferry")]
archs     = ["atombinaryfe"]
extractors = [Extractor(; architecture=a, graph_layers=l)
              for (a, l) in Iterators.product(archs, [2])]

message_blocks = [FFNN(hidden_dim=h, output_dim=h, layers=1, layernorm=ln)
                  for (h, ln) in Iterators.product([32], [true])]
pooled_blocks  = [FFNN(hidden_dim=h, output_dim=1, layers=l, layernorm=ln)
                  for (h, l, ln) in Iterators.product([32], [2], [true])]
models   = [Model(message_pass_model=mp, pooled_model=o)
            for (mp, o) in Iterators.product(message_blocks, pooled_blocks)]

losses   = [Loss(s) for s in ["lstar", "lgbfs"]]
trains   = [SupervisedTraining(; max_epoch=10)]
planners = [AStar(max_time=30)]
seeds    = [1, 2, 3]
