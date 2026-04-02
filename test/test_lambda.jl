"""
Test that λ controls the f/v tradeoff in the DIME acquisition function.

Acquisition: a(s) = -log σ(f(s)) - λ · max(v(s), 0)
  λ=0   → pure predictor ranking (same as lgbfs baseline)
  λ→∞   → dominated by v (CMI-guided exploration)

Two testsets:
1. Pure formula test — algebraic, deterministic, no PDDL/Mill.
   Constructs hand-picked f/v scores where f and v disagree, verifies
   the ranking flips at the analytically-derived crossover λ*.

2. Model-level test — uses DIMEHeuristic on the symmetric ferry pair.
   Measures actual f/v scores from the (random-weight) model, identifies
   the disagree regime, and confirms compute() respects the tradeoff.

Run with:
    julia --project=. test/test_lambda.jl
"""

using DIMEPlanning
using NeuroPlannerExperiments
using NeuroPlanner
using NeuroPlanner.PDDL
using NeuroPlanner.SymbolicPlanners
using NeuroPlanner.Mill
using NeuroPlanner.Mill.Flux: σ
using Test

# ============================================================================
# Part 1 — pure formula unit test
# ============================================================================
@testset "Acquisition formula: λ controls f/v tradeoff" begin

    # State A: strong predictor signal, low CMI value
    # State B: weak predictor signal, high CMI value
    # → at λ=0 A wins (low a is better = high priority in min-heap)
    # → at large λ B wins (higher v penalises A more)
    f_a, v_a = 2.0f0, 0.2f0
    f_b, v_b = 0.5f0, 3.0f0

    acq(f, v, λ) = -log(σ(f)) - λ * max(v, 0f0)

    @testset "λ=0 ranks by f only" begin
        a_A = acq(f_a, v_a, 0f0)
        a_B = acq(f_b, v_b, 0f0)
        # A has higher f → lower -log σ(f) → lower a → higher priority
        @test a_A < a_B
        # v is irrelevant: same result even with v zeroed out
        @test acq(f_a, 0f0, 0f0) == acq(f_a, v_a, 0f0)
        println("  a(A; λ=0) = $a_A   a(B; λ=0) = $a_B")
    end

    @testset "Large λ ranks by v (ranking flips)" begin
        a_A = acq(f_a, v_a, 20f0)
        a_B = acq(f_b, v_b, 20f0)
        # v_a < v_b → -λ*v_a > -λ*v_b → A is penalised less by v
        # but v_b is so much larger that B wins
        @test a_A > a_B
        println("  a(A; λ=20) = $a_A   a(B; λ=20) = $a_B")
    end

    @testset "Crossover λ* derived analytically" begin
        # acq(f_a, v_a, λ*) = acq(f_b, v_b, λ*)
        # -log σ(fₐ) - λ*·vₐ = -log σ(f_b) - λ*·v_b
        # ⟹  λ* = (log σ(fₐ) - log σ(f_b)) / (v_b - vₐ)
        λ_star = (log(σ(f_a)) - log(σ(f_b))) / (v_b - v_a)
        @test λ_star > 0f0   # sensible: f_a > f_b but v_a < v_b
        println("  λ* = $λ_star")

        ε = 0.05f0
        # Just below λ*: A still wins
        @test acq(f_a, v_a, λ_star - ε) < acq(f_b, v_b, λ_star - ε)
        # Just above λ*: B wins
        @test acq(f_a, v_a, λ_star + ε) > acq(f_b, v_b, λ_star + ε)
    end

    @testset "v ≤ 0 is clamped: negative v has no effect" begin
        # max(v, 0) means negative CMI estimates are ignored
        @test acq(f_a, -5f0, 10f0) == acq(f_a, 0f0, 10f0)
        @test acq(f_a, -5f0, 10f0) == acq(f_a, -99f0, 10f0)
    end
end

# ============================================================================
# Part 2 — model-level test via DIMEHeuristic
# ============================================================================
DOMAIN_FILE = joinpath(@__DIR__, "..", "NeuroPlannerExperiments.jl",
                       "data", "ipc23_learning", "ferry", "domain.pddl")
PROBLEM_STR = """
(define (problem ferry-lambda-test)
 (:domain ferry)
 (:objects car1 car2 - car  loc1 loc2 - location)
 (:init (empty-ferry) (at-ferry loc1) (at car1 loc1) (at car2 loc1))
 (:goal (and (at car1 loc2) (at car2 loc2))))
"""

@testset "DIMEHeuristic: λ changes ranking when f and v disagree" begin

    import Random

    domain  = load_domain(DOMAIN_FILE)
    problem = parse_problem(PROBLEM_STR)
    s₀      = PDDL.initstate(domain, problem)

    acts    = collect(PDDL.available(domain, s₀))
    board_c1 = only(filter(a -> a.name == :board && a.args[1].name == :car1, acts))
    sail     = only(filter(a -> a.name == :sail, acts))
    s_board  = PDDL.transition(domain, s₀, board_c1)   # structurally "richer"
    s_sail   = PDDL.transition(domain, s₀, sail)        # structurally different

    pddld = NeuroPlannerExperiments.materialize(
        Extractor(; architecture="atombinaryfe", graph_layers=2), domain)
    conf = Model(
        message_pass_model = FFNN(hidden_dim=16, output_dim=16, layers=1, layernorm=true),
        pooling = "SegmentedSumMax",
        pooled_model = FFNN(hidden_dim=16, output_dim=1, layers=2, layernorm=true)
    )
    # Seed 123: with context s₀, both states get positive v and f/v rankings disagree
    # (f_board > f_sail but v_board < v_sail) → ranking flip is detectable
    Random.seed!(123)
    model = construct_dime_model(pddld, problem, conf)
    spec  = SymbolicPlanners.Specification(problem)

    @testset "λ=0 acquisition equals -log σ(f)" begin
        # Verify formula: at λ=0, compute() must equal -log σ(f_score) exactly.
        pddle = NeuroPlanner.add_goalstate(pddld, problem)
        x = Mill.batch([pddle(s_board)])

        f_scores, _ = model(x, Mill.AlignedBags([1:0]), [1])
        expected = -log(σ(f_scores[1, 1]))

        hfun = DIMEHeuristic(pddld, problem, model; λ=0.0f0)
        actual = SymbolicPlanners.compute(hfun, domain, s_board, spec)

        @test actual ≈ expected  atol=1e-5
        println("  expected a(s; λ=0) = $expected   got $actual")
    end

    @testset "λ changes acquisition value (with context so v > 0)" begin
        # With no context, random-weight v heads often produce negative values,
        # which are clamped to 0 by max(v,0), making λ have no effect.
        # Use s₀ as context so the aggregator sees something non-trivial and
        # v is reliably non-zero.
        pddle = NeuroPlanner.add_goalstate(pddld, problem)
        x_s0    = Mill.batch([pddle(s₀)])
        x_board = Mill.batch([pddle(s_board)])

        # Raw model call: context=s₀ (obs 1), query=s_board (obs 2)
        x_ctx_q  = reduce(Mill.catobs, [x_s0, x_board])
        f_raw, v_raw = model(x_ctx_q, Mill.AlignedBags([1:1]), [2])
        println("  f(s_board|{s₀}) = $(f_raw[1,1])   v(s_board|{s₀}) = $(v_raw[1,1])")

        # v must be non-zero with context (otherwise the whole CMI idea is dead)
        @test v_raw[1,1] != 0f0

        # Now through DIMEHeuristic: evaluate s_board after s₀ was seen
        hfun_0 = DIMEHeuristic(pddld, problem, model; λ=0.0f0)
        hfun_1 = DIMEHeuristic(pddld, problem, model; λ=1.0f0)

        # Prime both heuristics with s₀ as context
        SymbolicPlanners.compute(hfun_0, domain, s₀, spec)
        SymbolicPlanners.compute(hfun_1, domain, s₀, spec)

        a0 = SymbolicPlanners.compute(hfun_0, domain, s_board, spec)
        a1 = SymbolicPlanners.compute(hfun_1, domain, s_board, spec)

        # λ has an effect iff max(v, 0) > 0 — which we verified above
        if v_raw[1,1] > 0f0
            @test a0 != a1
            println("  a(s_board|ctx; λ=0) = $a0   a(s_board|ctx; λ=1) = $a1")
        else
            # v < 0: max clamps to 0, λ legitimately has no effect
            @test a0 == a1
            println("  v ≤ 0 after context — λ correctly has no effect")
        end
    end

    @testset "Ranking flip when f and v disagree (with context)" begin
        # Query s_board and s_sail both with s₀ as context.
        # With context, v is non-trivially non-zero and f/v often disagree.
        pddle   = NeuroPlanner.add_goalstate(pddld, problem)
        x_s0    = Mill.batch([pddle(s₀)])
        x_board = Mill.batch([pddle(s_board)])
        x_sail  = Mill.batch([pddle(s_sail)])

        # context=s₀ for both queries
        x_ctx_board = reduce(Mill.catobs, [x_s0, x_board])
        x_ctx_sail  = reduce(Mill.catobs, [x_s0, x_sail])

        f_b_raw, v_b_raw = model(x_ctx_board, Mill.AlignedBags([1:1]), [2])
        f_s_raw, v_s_raw = model(x_ctx_sail,  Mill.AlignedBags([1:1]), [2])

        fb, vb = f_b_raw[1,1], max(v_b_raw[1,1], 0f0)
        fs, vs = f_s_raw[1,1], max(v_s_raw[1,1], 0f0)

        println("  f_board|ctx=$fb  v_board|ctx=$vb")
        println("  f_sail|ctx=$fs   v_sail|ctx=$vs")

        f_disagrees_with_v = (fb > fs) != (vb > vs) && vb != vs

        if f_disagrees_with_v
            # Analytically derive λ* and verify ranking flip
            # a_board = a_sail  ⟹  λ* = (log σ(fb) - log σ(fs)) / (vs - vb)
            λ_star = Float32((log(σ(fb)) - log(σ(fs))) / (vs - vb))
            @test λ_star > 0f0
            println("  λ* = $λ_star")

            hfun_lo = DIMEHeuristic(pddld, problem, model; λ = max(0f0, λ_star - 0.3f0))
            hfun_hi = DIMEHeuristic(pddld, problem, model; λ = λ_star + 0.3f0)

            # Prime with s₀ as context
            SymbolicPlanners.compute(hfun_lo, domain, s₀, spec)
            a_board_lo = SymbolicPlanners.compute(hfun_lo, domain, s_board, spec)
            reset_context!(hfun_lo)
            SymbolicPlanners.compute(hfun_lo, domain, s₀, spec)
            a_sail_lo  = SymbolicPlanners.compute(hfun_lo, domain, s_sail, spec)

            SymbolicPlanners.compute(hfun_hi, domain, s₀, spec)
            a_board_hi = SymbolicPlanners.compute(hfun_hi, domain, s_board, spec)
            reset_context!(hfun_hi)
            SymbolicPlanners.compute(hfun_hi, domain, s₀, spec)
            a_sail_hi  = SymbolicPlanners.compute(hfun_hi, domain, s_sail, spec)

            println("  λ < λ*: a_board=$a_board_lo  a_sail=$a_sail_lo")
            println("  λ > λ*: a_board=$a_board_hi  a_sail=$a_sail_hi")
            @test (a_board_lo < a_sail_lo) != (a_board_hi < a_sail_hi)
        else
            # With seed 123 this branch should not be reached.
            # If it is (e.g. different Julia version), verify λ still changes values.
            @info "f and v rankings agree with this seed — flip test skipped"
            state_with_pos_v = vs > 0f0 ? s_sail : s_board
            hfun_0 = DIMEHeuristic(pddld, problem, model; λ=0.0f0)
            hfun_5 = DIMEHeuristic(pddld, problem, model; λ=5.0f0)
            SymbolicPlanners.compute(hfun_0, domain, s₀, spec)
            SymbolicPlanners.compute(hfun_5, domain, s₀, spec)
            a0 = SymbolicPlanners.compute(hfun_0, domain, state_with_pos_v, spec)
            a5 = SymbolicPlanners.compute(hfun_5, domain, state_with_pos_v, spec)
            @test a0 != a5
        end
    end

end
