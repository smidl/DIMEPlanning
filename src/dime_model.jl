####################################################################
# DIMEModel
#
# Two-headed model for DIME-Planning:
#   - f_head: predictor  P(s on optimal plan | s, T)
#   - v_head: value net  E[CMI gained by expanding s | s, T]
#
# Shared GNN backbone. Context T is pooled via SegmentedSumMax and
# concatenated with the query embedding before the output heads.
#
# Input to both heads:
#   [GNN(s) ; SegmentedSumMax({GNN(t) : t ∈ T})]
#
# At inference (planning), the context T grows as nodes are expanded.
####################################################################

"""
    DIMEModel

Wraps a shared GNN backbone with two output heads.

Fields:
- `backbone`  : KnowledgeModel — maps a state (Mill node) to embedding ∈ R^d
- `aggregator`: Mill aggregation over context bag → R^d (e.g. SegmentedSumMax → R^2d)
- `f_head`    : MLP R^d → R^1, predicts log-odds P(s on optimal plan) — context-free, identical to lgbfs
- `v_head`    : MLP R^(d + 2d) → R^1, predicts CMI proxy Δ(s) — context-aware
"""
struct DIMEModel{B,A,F,V}
    backbone::B
    aggregator::A
    f_head::F
    v_head::V
end

Flux.@layer DIMEModel

"""
    embed(m::DIMEModel, x)

Apply the backbone GNN to a batch of states, returning embeddings ∈ R^(d × n).
The backbone is a KnowledgeModel; we intercept before the final pooled_model head.
"""
function embed(m::DIMEModel, x)
    m.backbone(x)  # R^(1 × n) — we rely on backbone having a 1-d output
    # Note: for DIME we actually need the pre-head representation.
    # See construct_dime_model for how the backbone is wired to stop before the
    # scalar head and instead output the hidden embedding.
end

"""
    (m::DIMEModel)(x, context_bags, query_ids)

Forward pass for training.

Arguments:
- `x`            : Mill node, all states (both query and context)
- `context_bags` : AlignedBags, one segment per query — indices into x of context states
- `query_ids`    : Vector{Int}, index of each query state in x

Returns `(f_scores, v_scores)` each of shape (1, n_queries).
"""
function (m::DIMEModel)(x, context_bags, query_ids::Vector{Int})
    # 1. Embed all states with shared backbone
    embeddings = m.backbone(x)   # (d × N)

    # 2. Extract query embeddings
    q = embeddings[:, query_ids]      # (d × n_queries)

    # 3. f_head: context-free (d → 1), identical to lgbfs scalar head
    f_scores = m.f_head(q)            # (1 × n_queries)

    # 4. Pool context for v_head: SegmentedSumMax over context embeddings
    context_node = Mill.BagNode(Mill.ArrayNode(embeddings), context_bags)
    ctx = m.aggregator(context_node)  # (2d × n_queries) for SumMax

    # 5. v_head: context-aware (3d → 1)
    joint    = vcat(q, ctx)           # (3d × n_queries)
    v_scores = m.v_head(joint)        # (1 × n_queries)

    (f_scores, v_scores)
end

"""
    f_score(m::DIMEModel, x, context_bags, query_ids)

Return only the predictor scores f(s|T). Used during planning as heuristic.
"""
function f_score(m::DIMEModel, x, context_bags, query_ids::Vector{Int})
    first(m(x, context_bags, query_ids))
end

"""
    construct_dime_model(pddld, problem, conf::Model, λ)

Build a DIMEModel from config. The backbone is built identically to the standard
model but wired to output a hidden embedding (not a scalar), followed by two
separate MLP heads.

The backbone hidden dim is taken from `conf.message_pass_model.output_dim`.
Context aggregation uses SegmentedSumMax, so the context vector has 2×hidden_dim.
Both heads take hidden_dim + 2*hidden_dim = 3*hidden_dim as input.
"""
function construct_dime_model(pddld, dataset::NeuroPlannerExperiments.Dataset, conf::Model)
    problem = NeuroPlanner.load_problem(first(dataset.train_files))
    construct_dime_model(pddld, problem, conf)
end

function construct_dime_model(pddld, problem::GenericProblem, conf::Model)
    message_pass_model = NeuroPlannerExperiments.parse_config(conf.message_pass_model)
    aggregation = NeuroPlannerExperiments.get_aggregation(conf.pooling)

    # Build backbone: same GNN as standard model, but output is the hidden embedding
    # (output_dim = hidden_dim, NOT 1). We do this by setting the pooled model to
    # output hidden_dim dimensions.
    embedding_dim = conf.message_pass_model.output_dim

    state = PDDL.initstate(pddld.domain, problem)
    pddle = NeuroPlanner.add_goalstate(pddld, problem)
    h₀ = pddle(state)

    # Backbone outputs embedding_dim-dimensional vectors, not scalars.
    # backbone_output = Dense(embedding_dim → embedding_dim, relu)  — just one layer,
    # no layernorm, so that backbone + f_head matches the lgbfs pooled_model exactly:
    #   backbone(·) + f_head = Dense(d→e,relu) + Dense(e→1) ≡ pooled_model(layers=2,odim=1)
    backbone_output = NeuroPlannerExperiments.FFNN(
        output_dim = embedding_dim,
        layers = 1,
        layernorm = false,
        sigma = conf.message_pass_model.sigma
    )
    backbone = NeuroPlanner.reflectinmodel(
        h₀, message_pass_model, aggregation;
        fsm = Dict("" => backbone_output)
    )

    # Aggregator for context bag: SegmentedSumMax over backbone embeddings
    # Input is embedding_dim, output is 2*embedding_dim (sum + max)
    aggregator = Mill.BagModel(
        identity,
        Mill.SegmentedSumMax(embedding_dim),
        identity
    )

    # Option B architecture:
    #   f_head: embedding_dim → 1  (context-free, single linear readout)
    #   v_head: 3*embedding_dim → 1 (context-aware: query + sum + max)
    #
    # f_head is a single Dense layer (no hidden layers) so that
    # backbone + f_head has the same total depth as the lgbfs model
    # (which uses backbone + pooled_model as one unit).
    # All MLP capacity is in the backbone; f_head is just a linear readout.
    # This guarantees λ=0 ≡ lgbfs in training depth and signal.
    head_conf = conf.pooled_model
    f_head = Flux.Dense(embedding_dim => 1)

    joint_dim = embedding_dim + 2 * embedding_dim   # = 3 * embedding_dim
    v_head = NeuroPlannerExperiments.FFNN(
        hidden_dim = head_conf.hidden_dim,
        output_dim = 1,
        layers = head_conf.layers,
        layernorm = head_conf.layernorm
    )(joint_dim)

    DIMEModel(backbone, aggregator, f_head, v_head)
end
