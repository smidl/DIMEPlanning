"""
Symmetry test for DIMEPlanning.

The key property DIME adds over plain lgbfs is context-awareness:
two states that look *identical* to a context-free GNN should get
the same score with empty context, but different scores once one of
them enters the context bag.

Test setup — ferry problem with 2 symmetric cars:

    s₀  ──board(car1,loc1)──→  s_a   (car1 on ferry, car2 at loc1)
        ╲──board(car2,loc1)──→  s_b   (car2 on ferry, car1 at loc1)

atom-binary-fe encodes states by lifted predicate structure, not object
identity, so s_a and s_b are GNN-isomorphic: f(s_a|∅) ≈ f(s_b|∅).

Once s_a enters the context bag the aggregator sees different input for
s_b, breaking the tie: f(s_b|{s_a}) ≠ f(s_b|∅).

Run with:
    julia --project=. test/test_symmetry.jl
"""

using DIMEPlanning
using NeuroPlannerExperiments
using NeuroPlanner
using NeuroPlanner.PDDL
using NeuroPlanner.SymbolicPlanners
using NeuroPlanner.Mill
using Test

# ---- problem ----------------------------------------------------------------
# Inline PDDL: two cars symmetric at loc1, goal = both at loc2.
DOMAIN_FILE = joinpath(@__DIR__, "..", "NeuroPlannerExperiments.jl",
                       "data", "ipc23_learning", "ferry", "domain.pddl")
PROBLEM_STR = """
(define (problem ferry-symmetry-test)
 (:domain ferry)
 (:objects car1 car2 - car  loc1 loc2 - location)
 (:init (empty-ferry) (at-ferry loc1) (at car1 loc1) (at car2 loc1))
 (:goal (and (at car1 loc2) (at car2 loc2))))
"""

@testset "DIME context breaks GNN symmetry" begin

    domain  = load_domain(DOMAIN_FILE)
    problem = parse_problem(PROBLEM_STR)
    s₀      = PDDL.initstate(domain, problem)

    # ---- get the two symmetric successor states ----------------------------
    acts    = collect(PDDL.available(domain, s₀))
    board_c1 = only(filter(a -> a.name == :board && a.args[1].name == :car1, acts))
    board_c2 = only(filter(a -> a.name == :board && a.args[1].name == :car2, acts))
    s_a = PDDL.transition(domain, s₀, board_c1)   # car1 on ferry
    s_b = PDDL.transition(domain, s₀, board_c2)   # car2 on ferry

    # ---- extractor + model (random weights) --------------------------------
    pddld = NeuroPlannerExperiments.materialize(
        Extractor(; architecture="atombinaryfe", graph_layers=2), domain)
    pddle = NeuroPlanner.add_goalstate(pddld, problem)

    conf = Model(
        message_pass_model = FFNN(hidden_dim=16, output_dim=16, layers=1, layernorm=true),
        pooling = "SegmentedSumMax",
        pooled_model = FFNN(hidden_dim=16, output_dim=1, layers=2, layernorm=true)
    )
    model = construct_dime_model(pddld, problem, conf)

    x_a = Mill.batch([pddle(s_a)])   # single-obs batched KnowledgeBase
    x_b = Mill.batch([pddle(s_b)])

    # ---- Test 1: GNN symmetry — no context ---------------------------------
    # s_a and s_b have identical predicate structure under car1↔car2 permutation.
    # atom-binary-fe features are lifted (no object-name features), so the GNN
    # backbone produces the same embedding for both.
    # Expected: |f(s_a|∅) − f(s_b|∅)| < 1e-5  (floating-point only)
    @testset "No context: symmetric states get equal scores" begin
        f_a, _ = model(x_a, Mill.AlignedBags([1:0]), [1])
        f_b, _ = model(x_b, Mill.AlignedBags([1:0]), [1])
        @test abs(f_a[1,1] - f_b[1,1]) < 1e-5
        println("  f(s_a|∅) = $(f_a[1,1])")
        println("  f(s_b|∅) = $(f_b[1,1])")
        println("  |diff|   = $(abs(f_a[1,1] - f_b[1,1]))")
    end

    # ---- Test 2: context breaks symmetry -----------------------------------
    # Add s_a to the context bag and re-query s_b.
    # The aggregator now receives a non-empty bag containing s_a's embedding,
    # producing a different joint representation for s_b.
    # Expected: f(s_b|{s_a}) ≠ f(s_b|∅)  — any non-zero weight breaks the tie.
    @testset "With context: scores change for the symmetric sibling" begin
        # batch = [s_a (context), s_b (query)]; context covers obs 1, query is obs 2
        x_ctx_query = reduce(Mill.catobs, [x_a, x_b])
        f_ctx, v_ctx = model(x_ctx_query, Mill.AlignedBags([1:1]), [2])
        f_empty, v_empty = model(x_b, Mill.AlignedBags([1:0]), [1])

        @test abs(f_ctx[1,1] - f_empty[1,1]) > 1e-3   # context changes f score
        @test abs(v_ctx[1,1] - v_empty[1,1]) > 1e-3   # context changes v score
        println("  f(s_b|∅)      = $(f_empty[1,1])")
        println("  f(s_b|{s_a})  = $(f_ctx[1,1])")
        println("  v(s_b|∅)      = $(v_empty[1,1])")
        println("  v(s_b|{s_a})  = $(v_ctx[1,1])")
    end

    # ---- Test 3: DIMEHeuristic produces the same split ---------------------
    # The same effect should appear through the full DIMEHeuristic API.
    @testset "DIMEHeuristic: context via compute() breaks symmetry" begin
        spec  = SymbolicPlanners.Specification(problem)

        # Evaluate s_a first (adds it to context), then s_b
        hfun = DIMEHeuristic(pddld, problem, model; λ=0.5f0)
        val_a  = SymbolicPlanners.compute(hfun, domain, s_a, spec)
        val_b_with_ctx = SymbolicPlanners.compute(hfun, domain, s_b, spec)

        # Evaluate s_b alone (empty context)
        reset_context!(hfun)
        val_b_no_ctx = SymbolicPlanners.compute(hfun, domain, s_b, spec)

        @test abs(val_b_with_ctx - val_b_no_ctx) > 1e-3
        println("  a(s_a)         = $val_a")
        println("  a(s_b|{s_a})   = $val_b_with_ctx")
        println("  a(s_b|∅)       = $val_b_no_ctx")
        println("  |diff|         = $(abs(val_b_with_ctx - val_b_no_ctx))")
    end

    # ---- Test 4: asymmetry is specific to the sibling ----------------------
    # A state from a *different* orbit (e.g. after sailing) has a different
    # baseline score; adding s_a to context should still change it, but the
    # zero-context gap between s_a and s_b must hold only for the isomorphic pair.
    @testset "Non-symmetric state scores differ even without context" begin
        sail = only(filter(a -> a.name == :sail, acts))
        s_sail = PDDL.transition(domain, s₀, sail)
        x_sail = Mill.batch([pddle(s_sail)])

        f_a, _    = model(x_a,    Mill.AlignedBags([1:0]), [1])
        f_sail, _ = model(x_sail, Mill.AlignedBags([1:0]), [1])

        @test abs(f_a[1,1] - f_sail[1,1]) > 1e-3   # sail state is structurally different
        println("  f(s_a|∅)    = $(f_a[1,1])")
        println("  f(s_sail|∅) = $(f_sail[1,1])")
    end

end
