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

Backbone embeddings are cached: each call runs the GNN only on the new
query state, then appends its embedding column to `ctx_embeddings`.
The aggregator (SegmentedSumMax) then runs on the cached matrix — O(1)
GNN calls per step instead of O(n) (no re-batching of all context).

Fields:
- `pddle`         : problem-specialized extractor
- `model`         : DIMEModel
- `λ`             : CMI exploration weight (λ=0 → pure predictor; ignored for "product")
- `acquisition`   : "additive" (UCB-style, default) or "product" (EI-style, β-free)
- `context`       : Vector of raw Mill nodes (used for length tracking / reset)
- `ctx_embeddings`: Float32 matrix (d × n_ctx) of cached backbone embeddings
- `t`             : timer (for profiling, compatible with NeuroHeuristic)
- `freeze_context`: if true, never accumulate context (context-free inference diagnostic)
"""
mutable struct DIMEHeuristic <: SymbolicPlanners.Heuristic
    pddle::Any
    model::DIMEModel
    λ::Float32
    acquisition::String              # "additive" or "product"
    context::Vector{Any}            # raw Mill nodes (for reset; not re-batched)
    ctx_embeddings::Matrix{Float32} # (d × n_ctx) cached backbone embeddings
    t::Base.RefValue{Float64}
    freeze_context::Bool
end

function DIMEHeuristic(pddld, problem::GenericProblem, model::DIMEModel;
                       λ::Real = 0.5f0,
                       acquisition::String = "additive",
                       freeze_context::Bool = false,
                       goal_state = NeuroPlanner.goalstate(pddld.domain, problem))
    pddle = NeuroPlanner.add_goalstate(pddld, problem, goal_state)
    DIMEHeuristic(pddle, model, Float32(λ), acquisition,
                  Any[], zeros(Float32, 0, 0), Ref(0.0), freeze_context)
end

"""
    SymbolicPlanners.compute(h::DIMEHeuristic, domain, state, spec)

Evaluate the DIME acquisition function a(s) = log f(s|T) + λ·v(s|T).
After evaluation, the state's embedding is appended to the context cache.
"""
function SymbolicPlanners.compute(h::DIMEHeuristic, _domain::Domain, state::GenericState, _spec::Specification)
    t = @elapsed begin
        # 1. Embed the query state (GNN forward pass on one state only)
        x_s     = h.pddle(state)
        x_query = Mill.batch([x_s])
        q_emb   = h.model.backbone(x_query)   # (d × 1)

        n_ctx = size(h.ctx_embeddings, 2)

        if n_ctx == 0
            # No context yet — aggregator sees an empty bag
            ctx = h.model.aggregator(
                Mill.BagNode(Mill.ArrayNode(q_emb), Mill.AlignedBags([1:0])))
        else
            # Pool the cached context embeddings (no GNN re-run)
            ctx = h.model.aggregator(
                Mill.BagNode(Mill.ArrayNode(h.ctx_embeddings),
                             Mill.AlignedBags([1:n_ctx])))
        end

        # 2. Head forward passes (Option B: f_head is context-free, v_head is context-aware)
        f_val   = h.model.f_head(q_emb)[1, 1]          # (d → 1), no context
        joint   = vcat(q_emb, ctx)                      # (3d × 1)
        v_val   = h.model.v_head(joint)[1, 1]           # (3d → 1), with context

        # 3. Acquisition: lower = better (min-heap convention, same as lgbfs).
        # Training gives trajectory states LOW f; low a → high priority → expanded first.
        v_plus = max(v_val, 0f0)
        a = if h.acquisition == "product"
            # β-free EI-style: a = f / (1 + max(v,0))
            # High CMI → larger denominator → smaller a → higher priority
            f_val / (1f0 + v_plus)
        else
            # Additive UCB-style: a = f - λ*max(v,0)
            f_val - h.λ * v_plus
        end

        # 4. Append query embedding to context cache (skip if freeze_context)
        if !h.freeze_context
            h.ctx_embeddings = if n_ctx == 0
                Matrix{Float32}(q_emb)
            else
                hcat(h.ctx_embeddings, Matrix{Float32}(q_emb))
            end
            push!(h.context, x_s)
        end
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
    h.ctx_embeddings = zeros(Float32, 0, 0)
    h
end

# ---- Planner config ---------------------------------------------------------

"""
    DIMEPlanner <: PlannerType

Config struct for a DIME-guided GBFS planner.

Fields:
- `max_nodes`   : expansion budget
- `max_time`    : wall-clock time limit (seconds)
- `λ`           : CMI exploration weight (used only for "additive"; ignored for "product")
- `acquisition` : "additive" (a = f - λ·v, UCB-style) or "product" (a = f/(1+v), EI-style)
- `g_mult`      : path-cost multiplier (1.0 = A*, 0.0 = pure GBFS)
- `freeze_context` : diagnostic — context-free inference if true
- `search_noise` : optional Boltzmann noise
"""
@confdef struct DIMEPlanner <: NeuroPlannerExperiments.PlannerType
    max_nodes::Int64 = typemax(Int64)
    max_time::Int64 = 30
    λ::Float32 = 0.5f0
    acquisition::String = "additive"   # "additive" or "product"
    g_mult::Float32 = 1.0f0
    freeze_context::Bool = false
    search_noise = nothing
end

Base.string(::DIMEPlanner) = "DIMEPlanner"

# Make DIMEPlanner round-trippable through NeuroPlannerExperiments.load_config/parse_config.
# @confdef only registers parse_type in the local (DIMEPlanning) module; parse_config
# calls NeuroPlannerExperiments.parse_type, so we add the method there explicitly.
NeuroPlannerExperiments.parse_type(::Val{:DIMEPlanner}) = DIMEPlanner

function NeuroPlannerExperiments.materialize(o::DIMEPlanner, heuristic::DIMEHeuristic)
    NeuroPlannerExperiments.ForwardPlannerX(;
        heuristic,
        g_mult     = o.g_mult,
        max_time   = Float64(o.max_time),
        max_nodes  = o.max_nodes,
        search_noise = o.search_noise,
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
    hfun = DIMEHeuristic(pddld, problem, model;
                         λ = planner.λ,
                         acquisition = planner.acquisition,
                         freeze_context = planner.freeze_context)
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
