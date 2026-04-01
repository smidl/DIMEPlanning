####################################################################
# DIMEHeuristic and DIMEPlanner
#
# At planning time, the search tree T grows incrementally.
# DIMEHeuristic wraps DIMEModel and maintains a running context:
#   - expanded_states: list of Mill nodes for states expanded so far
#   - At each call h(s), re-pools the context and evaluates:
#       a(s) = log f(s|T) + λ * v(s|T)
#
# λ=0 → pure predictor (equivalent to lgbfs)
# λ>0 → CMI-guided exploration
####################################################################

"""
    DIMEHeuristic <: SymbolicPlanners.Heuristic

Heuristic that uses DIMEModel with accumulated search tree context.

Fields:
- `pddle`    : problem-specialized extractor
- `model`    : DIMEModel
- `λ`        : CMI exploration weight (λ=0 → pure predictor)
- `context`  : Vector of Mill nodes for expanded states (grows during search)
- `t`        : timer (for profiling, compatible with NeuroHeuristic)
"""
mutable struct DIMEHeuristic <: SymbolicPlanners.Heuristic
    pddle::Any
    model::DIMEModel
    λ::Float32
    context::Vector{Any}   # accumulated expanded state embeddings
    t::Base.RefValue{Float64}
end

function DIMEHeuristic(pddld, problem::GenericProblem, model::DIMEModel;
                       λ::Real = 0.5f0,
                       goal_state = NeuroPlanner.goalstate(pddld.domain, problem))
    pddle = NeuroPlanner.add_goalstate(pddld, problem, goal_state)
    DIMEHeuristic(pddle, model, Float32(λ), Any[], Ref(0.0))
end

"""
    SymbolicPlanners.compute(h::DIMEHeuristic, domain, state, spec)

Evaluate the DIME acquisition function a(s) = log f(s|T) + λ·v(s|T).
After evaluation, the state is added to the context T.
"""
function SymbolicPlanners.compute(h::DIMEHeuristic, _domain::Domain, state::GenericState, _spec::Specification)
    t = @elapsed begin
        x_s = h.pddle(state)
        x_query = Mill.batch([x_s])

        if isempty(h.context)
            # No context yet — use a single empty bag
            context_bags = Mill.AlignedBags([1:0])
            x_all = x_query
            query_ids = [1]
        else
            # Stack context + query into one batch.
            # h.context stores single-state batched nodes (same type as x_query).
            # catobs them all together then append the query at the end.
            n_ctx = length(h.context)
            # reduce(catobs, Vector{<:KnowledgeBase}) is defined in NeuroPlanner.
            # h.context is Vector{Any}; convert to a concretely-typed vector first.
            ctx_vec = convert(Vector{typeof(x_query)}, h.context)
            x_all = reduce(Mill.catobs, [ctx_vec; [x_query]])
            # Context = first n_ctx observations, query = last one
            context_bags = Mill.AlignedBags([1:n_ctx])
            query_ids = [n_ctx + 1]
        end

        f_scores, v_scores = h.model(x_all, context_bags, query_ids)
        f_val = f_scores[1, 1]
        v_val = v_scores[1, 1]

        # Acquisition: lower = better (heuristic convention: expand min first)
        # log f is negative for low P(on optimal plan) → high priority when negative
        a = -log(σ(f_val)) - h.λ * max(v_val, 0f0)

        # Add current state to context for future evaluations (store as single-obs batch)
        push!(h.context, x_query)
    end
    h.t[] += t
    return Float32(a)
end

"""
    reset_context!(h::DIMEHeuristic)

Clear the accumulated search tree context. Call between planning episodes.
"""
function reset_context!(h::DIMEHeuristic)
    empty!(h.context)
    h
end

# ---- Planner config ---------------------------------------------------------

"""
    DIMEPlanner <: PlannerType

Config struct for a DIME-guided GBFS planner.

Fields:
- `max_nodes` : expansion budget
- `max_time`  : wall-clock time limit (seconds)
- `λ`         : CMI exploration weight
- `search_noise` : optional noise for tie-breaking
"""
@confdef struct DIMEPlanner <: NeuroPlannerExperiments.PlannerType
    max_nodes::Int64 = typemax(Int64)
    max_time::Int64 = 30
    λ::Float32 = 0.5f0
    search_noise = nothing
end

Base.string(::DIMEPlanner) = "DIMEPlanner"

function NeuroPlannerExperiments.materialize(o::DIMEPlanner, heuristic::DIMEHeuristic)
    max_time = Float64(o.max_time)
    max_nodes = o.max_nodes
    search_noise = o.search_noise
    # GBFS: g_mult=0, h_mult=1, heuristic = DIME acquisition function
    NeuroPlannerExperiments.ForwardPlannerX(;
        heuristic,
        g_mult = 0f0,
        max_time, max_nodes, search_noise,
        save_search = true,
        save_parents = true,
        save_children = true
    )
end

"""
    solve_problem(pddld, problem, model::DIMEModel, planner::DIMEPlanner)

Override solve_problem to use DIMEHeuristic (which maintains search tree context)
instead of the standard NeuroHeuristic.
"""
function NeuroPlannerExperiments.solve_problem(pddld, problem::GenericProblem,
                                               model::DIMEModel,
                                               planner::DIMEPlanner)
    domain = pddld.domain
    state = PDDL.initstate(domain, problem)
    hfun = DIMEHeuristic(pddld, problem, model; λ = planner.λ)
    concrete_planner = NeuroPlannerExperiments.materialize(planner, hfun)
    solution_time = @elapsed sol = concrete_planner(domain, state, PDDL.get_goal(problem))
    stats = (;
        solution_time,
        sol_length   = length(sol.trajectory),
        expanded     = sol.expanded,
        generated    = length(sol.search_tree),
        solved       = sol.status == :success,
        time_in_heuristic = hfun.t[]
    )
    (; sol, stats)
end
