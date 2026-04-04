####################################################################
# DIME Loss
#
# Joint loss for DIMEModel:
#   L_total = L_pred + α * L_value
#
# L_pred: lgbfs ranking loss on f_scores (same as L_gbfs in LEAH)
#   — penalises open-set states ranking above trajectory states
#   — trains f to recover P(s on optimal plan | T)
#
# L_value: MSE loss on v_scores against Δ_targets
#   — Δ(s) = L_pred(T_before expanding s) - L_pred(T_after expanding s)
#   — trains v to recover CMI I(y; expand(s) | T) (DIME Lemma 1)
#
# At λ=0 (α=0), L_total = L_pred = L_gbfs, recovering the LEAH baseline.
####################################################################

"""
    dime_pred_loss(f_scores, H₊, H₋, surrogate=softplus)

Ranking loss on predictor scores f(s|T). Identical in form to lgbfsloss:
trajectory states (H₋) should have lower f than open-set states (H₊).
Note: path cost g is NOT added here, matching lgbfsloss exactly.
"""
function dime_pred_loss(f_scores, H₊, H₋, surrogate=softplus)
    o = f_scores * H₋ - f_scores * H₊
    isempty(o) && return zero(eltype(o))
    mean(surrogate.(o))
end

"""
    dime_value_loss(v_scores, Δ_targets)

MSE loss between value network predictions and Δ targets.
v_scores shape: (1, n_queries); Δ_targets shape: (n_queries,)
"""
function dime_value_loss(v_scores, Δ_targets)
    isempty(Δ_targets) && return zero(Float32)
    Δ = reshape(Δ_targets, 1, :)
    mean((v_scores .- Δ).^2)
end

"""
    compute_delta_targets!(mb::DIMEMiniBatch, model::DIMEModel)

Fill in Δ_targets in-place using the current predictor.

For each query state s_k with context T_k:
    Δ(s_k) = L_pred(T_k) - L_pred(T_k ∪ {s_k expanded})

Approximation: run the model on all unique states (queries + their paired
trajectory states) with empty context, then:
    Δ(s_k) ≈ surrogate(f(s_k) - f(s⁻_k))
where s⁻_k is the paired trajectory state from H₋.
This matches the dime_pred_loss convention (no g offset).
"""
function compute_delta_targets!(mb::DIMEMiniBatch, model::DIMEModel, surrogate=softplus)
    isempty(mb.query_ids) && return mb

    # Collect all unique state indices needed: queries + their H₋ pairs
    H₋_dense = Array(mb.H₋)
    n_queries = length(mb.query_ids)
    traj_ids = Int[]
    for k in 1:n_queries
        tid = findfirst(>(0), H₋_dense[:, k])
        push!(traj_ids, tid === nothing ? 0 : tid)
    end

    # Run model on all queries (with their actual context bags)
    f_q, _ = model(mb.x, mb.context_bags, mb.query_ids)

    # Run model on trajectory states with empty context for their scores
    unique_tids = unique(filter(>(0), traj_ids))
    empty_bags  = Mill.ScatteredBags([Int[] for _ in unique_tids])
    f_t_all, _ = model(mb.x, empty_bags, unique_tids)
    tid_to_score = Dict(tid => f_t_all[1, i] for (i, tid) in enumerate(unique_tids))

    for k in 1:n_queries
        tid = traj_ids[k]
        tid == 0 && continue
        mb.Δ_targets[k] = surrogate(f_q[1, k] - tid_to_score[tid])
    end
    mb
end

"""
    dime_loss(model::DIMEModel, mb::DIMEMiniBatch; α=0.1f0)

Full DIME loss. Computes Δ_targets from current model, then:
    L = L_pred + α * L_value

α=0.1 weights the value loss lower than the predictor loss initially.
At α=0, recovers the lgbfs baseline exactly.
"""
function dime_loss(model::DIMEModel, mb::DIMEMiniBatch; α=0.1f0)
    isempty(mb.query_ids) && return zero(Float32)

    # Option B: f_head is context-free (d → 1), scores ALL states in one backbone pass.
    # This is identical to lgbfsloss: score everything, rank with H₊/H₋.
    # v_head is context-aware (3d → 1), only scored for query states.
    all_ids = ChainRulesCore.@ignore_derivatives collect(1:size(mb.H₊, 1))
    empty_bags_all = ChainRulesCore.@ignore_derivatives Mill.ScatteredBags([Int[] for _ in all_ids])
    f_full, _  = model(mb.x, empty_bags_all, all_ids)   # (1 × N) — all states, no context
    _, v_scores = model(mb.x, mb.context_bags, mb.query_ids)  # (1 × n_q) — queries with context

    L_pred = dime_pred_loss(f_full, mb.H₊, mb.H₋)

    α == 0 && return L_pred

    # Value loss — Δ targets are treated as constants (stop-gradient):
    # they are regression targets computed from a detached forward pass.
    ChainRulesCore.@ignore_derivatives compute_delta_targets!(mb, model)
    L_value = dime_value_loss(v_scores, mb.Δ_targets)

    L_pred + α * L_value
end

# Dispatch for training loop (same interface as NeuroPlanner.loss)
NeuroPlanner.loss(model::DIMEModel, mb::DIMEMiniBatch) = dime_loss(model, mb)
