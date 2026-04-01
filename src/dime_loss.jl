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
    dime_pred_loss(f_scores, path_cost, H₊, H₋, surrogate=softplus)

Ranking loss on predictor scores f(s|T). Identical in form to lₛloss
but applied to the context-aware f_scores instead of model(x).

Open-set states (H₊) should have higher f+g than trajectory states (H₋).
"""
function dime_pred_loss(f_scores, path_cost, H₊, H₋, surrogate=softplus)
    g = reshape(path_cost, 1, :)
    f = f_scores + g
    o = f * H₋ - f * H₊
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

Approximation used here: we compute L_pred on the full minibatch once,
then approximate Δ(s_k) as the per-sample contribution of s_k to the loss.
This avoids re-running the model n_queries times.

Specifically:
    Δ(s_k) ≈ surrogate(f(s_k) + g(s_k) - f(s⁻_k) - g(s⁻_k))
where s⁻_k is the paired trajectory state (from H₋).
"""
function compute_delta_targets!(mb::DIMEMiniBatch, model::DIMEModel, surrogate=softplus)
    isempty(mb.query_ids) && return mb
    f_scores, _ = model(mb.x, mb.context_bags, mb.query_ids)
    g = mb.path_cost

    # For each query k: paired trajectory state index comes from H₋
    # H₋ is a one-hot matrix: H₋[:, k] has a 1 at the paired trajectory state
    H₋_dense = Array(mb.H₋)  # (N × n_pairs) — same size as H₊
    n_queries = length(mb.query_ids)

    for k in 1:n_queries
        qid = mb.query_ids[k]
        # Find paired trajectory state (column k of H₋ has one non-zero entry)
        col = H₋_dense[:, k]   # this is the k-th column pairing
        tid = findfirst(>(0), col)
        tid === nothing && continue
        f_q = f_scores[1, k] + g[qid]
        f_t = f_scores[1, k] + g[tid]   # reuse f_scores for trajectory state
        mb.Δ_targets[k] = surrogate(f_q - f_t)
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

    # Forward pass
    f_scores, v_scores = model(mb.x, mb.context_bags, mb.query_ids)

    # Predictor loss (lgbfs ranking on f_scores)
    # Project f_scores (1 × n_q) to full state space (1 × N) without mutation:
    # build a constant scatter matrix Q (n_q × N) with Q[k, query_ids[k]] = 1,
    # then f_full = f_scores * Q  — only the matmul is differentiated.
    N = size(mb.H₊, 1)
    n_q = length(mb.query_ids)
    # Q is a constant scatter matrix — mark @ignore_derivatives so Zygote
    # treats it as a fixed matrix and only differentiates through the matmul.
    Q = ChainRulesCore.@ignore_derivatives begin
        Q_buf = zeros(Float32, n_q, N)
        for (k, qid) in enumerate(mb.query_ids)
            Q_buf[k, qid] = 1.0f0
        end
        Q_buf
    end
    f_full = f_scores * Q
    L_pred = dime_pred_loss(f_full, mb.path_cost, mb.H₊, mb.H₋)

    α == 0 && return L_pred

    # Value loss — Δ targets are treated as constants (stop-gradient):
    # they are regression targets computed from a detached forward pass.
    ChainRulesCore.@ignore_derivatives compute_delta_targets!(mb, model)
    L_value = dime_value_loss(v_scores, mb.Δ_targets)

    L_pred + α * L_value
end

# Dispatch for training loop (same interface as NeuroPlanner.loss)
NeuroPlanner.loss(model::DIMEModel, mb::DIMEMiniBatch) = dime_loss(model, mb)
