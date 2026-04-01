####################################################################
# DIMEMiniBatch
#
# A training sample for DIME-Planning. Extends LₛMiniBatch with:
#   - A context representation of the search tree T (expanded nodes)
#   - Δ targets for the value network (loss improvement per expansion)
#
# Architecture (see docs/method.md):
#   f(s | T) = MLP([GNN(s) ; SegmentedSumMax({GNN(t) : t ∈ T})])
#   v(s | T) = MLP([GNN(s) ; SegmentedSumMax({GNN(t) : t ∈ T})])
#
# Both heads share a GNN backbone. The context bag uses the same GNN
# weights as the query state, so comparisons are in the same space.
####################################################################

"""
    DIMEMiniBatch

A minibatch for joint training of the DIME predictor f and value network v.

Fields:
- `x`          : batched state representations for all states (query + context)
                 Mill structure, same type as in LₛMiniBatch
- `H₊`         : one-hot matrix — open-set states (should rank higher = worse)
- `H₋`         : one-hot matrix — trajectory states (should rank lower = better)
- `path_cost`  : g-values for each state
- `context_bags`: Mill.AlignedBags indicating which states form the context T
                  for each query state in the open set
- `query_ids`  : linear indices into `x` for each query state (open-set states)
- `Δ_targets`  : Float32 vector — loss-improvement target for value network,
                  one per query state. Δ(s) = L_pred(T_before) - L_pred(T_after)
"""
struct DIMEMiniBatch{X,H,Y,B} <: NeuroPlanner.AbstractMinibatch
    x::X
    H₊::H
    H₋::H
    path_cost::Y
    context_bags::B        # AlignedBags: for each query, which states are context
    query_ids::Vector{Int} # index of each query state in x
    Δ_targets::Vector{Float32}
end

"""
    DIMEMiniBatch(pddld, domain, problem, plan; kwargs...)

Construct a DIMEMiniBatch from a plan (sequence of actions).
Simulates the plan to get a trajectory, then calls the trajectory constructor.
"""
function DIMEMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem,
                       plan::AbstractVector{<:Julog.Term}; kwargs...)
    state = initstate(domain, problem)
    trajectory = SymbolicPlanners.simulate(StateRecorder(), domain, state, plan)
    DIMEMiniBatch(pddld, domain, problem, trajectory, nothing; kwargs...)
end

"""
    DIMEMiniBatch(pddld, domain, problem, trajectory, st; kwargs...)

Core constructor. Replicates LₛMiniBatch state collection, then additionally:
1. For each open-set state s, records which states were in T at that point
   (i.e., all trajectory states expanded before s was encountered).
2. Computes Δ_targets offline using predictor loss before/after each expansion.
   At construction time, Δ is approximated as 0.0 (placeholder); real Δ is
   computed during training via `compute_delta_targets!`.
"""
function DIMEMiniBatch(pddld, domain::GenericDomain, problem::GenericProblem,
                       trajectory::AbstractVector{<:GenericState},
                       st::Union{Nothing,NeuroPlanner.RSearchTree} = nothing;
                       goal_aware = true,
                       goal_state = NeuroPlanner.goalstate(pddld.domain, problem),
                       max_branch = typemax(Int),
                       dedup = true,
                       kwargs...)

    !issubset(goal_state, trajectory[end]) && error("Last state on trajectory is not superset of goal state")
    pddle = goal_aware ? NeuroPlanner.add_goalstate(pddld, problem, goal_state) :
                         NeuroPlanner.specialize(pddld, problem)
    spec = SymbolicPlanners.Specification(problem)

    # ---- collect all states and their g-values (same as LₛMiniBatch) --------
    state = trajectory[1]
    stateids = Dict(hash(state) => 1)
    states = [(g = 0, state = state)]
    htrajectory = hash.(trajectory)

    I₊ = Vector{Int}()   # open-set state indices (for H₊/H₋)
    I₋ = Vector{Int}()

    # For each open-set query, track which trajectory states were expanded before it
    # context_ids[k] = sorted list of state indices that form T when query k was seen
    context_ids_per_query = Vector{Vector{Int}}()
    query_ids = Vector{Int}()

    expanded_so_far = Int[]   # trajectory state indices in expansion order

    for i in 1:(length(trajectory)-1)
        sᵢ, sⱼ = trajectory[i], trajectory[i+1]
        hsⱼ = hash(sⱼ)
        gᵢ = states[stateids[hash(sᵢ)]].g

        next_states = NeuroPlanner._next_states(domain, problem, sᵢ, st)
        if length(next_states) > max_branch
            ii = findall(s -> s.state == sⱼ, next_states)
            ii = union(ii, StatsBase.sample(1:length(next_states), max_branch, replace=false))
            next_states = next_states[ii]
        end
        isempty(next_states) && error("inner node not in search tree")

        for ns in next_states
            act = ns.parent_action
            act_cost = SymbolicPlanners.get_cost(spec, domain, sᵢ, act, ns.state)
            ns.id ∈ keys(stateids) && continue
            stateids[ns.id] = length(stateids) + 1
            push!(states, (;g = gᵢ + act_cost, state = ns.state))
        end
        @assert hsⱼ ∈ keys(stateids)

        # Record that sᵢ was just expanded
        push!(expanded_so_far, stateids[hash(sᵢ)])

        open_set = setdiff(keys(stateids), htrajectory)
        for s in open_set
            qid = stateids[s]
            push!(I₊, qid)
            push!(I₋, stateids[hsⱼ])
            push!(query_ids, qid)
            push!(context_ids_per_query, copy(expanded_so_far))
        end
    end

    # ---- build Mill structures -----------------------------------------------
    if isempty(I₊)
        H₊ = OneHotArrays.onehotbatch([], 1:length(stateids))
        H₋ = OneHotArrays.onehotbatch([], 1:length(stateids))
    else
        H₊ = OneHotArrays.onehotbatch(I₊, 1:length(stateids))
        H₋ = OneHotArrays.onehotbatch(I₋, 1:length(stateids))
    end
    path_cost = Float32[s.g for s in states]

    inner_type = typeof(pddle(first(states).state))
    x = Mill.batch(inner_type[pddle(s.state) for s in states])
    x = dedup ? NeuroPlanner.deduplicate(x) : x

    # ---- build context bags --------------------------------------------------
    # For each query q_k, context_ids_per_query[k] lists state indices in x
    # that form the context T. We represent this as an AlignedBags over a
    # re-indexed flat array (one segment per query).
    n_queries = length(query_ids)
    if n_queries == 0
        # empty — no open-set states
        context_bags = Mill.ScatteredBags(Vector{Int}[])
        Δ_targets = Float32[]
    else
        # ScatteredBags stores actual state indices into x (not sequential re-indexed)
        context_bags = Mill.ScatteredBags(context_ids_per_query)
        Δ_targets = zeros(Float32, n_queries)   # placeholder; filled by compute_delta_targets!
    end

    DIMEMiniBatch(x, H₊, H₋, path_cost, context_bags, query_ids, Δ_targets)
end
