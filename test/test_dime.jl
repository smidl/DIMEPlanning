"""
Integration test for DIMEPlanning.

Tests:
1. DIMEMiniBatch construction from a real ferry problem
2. DIMEModel construction
3. Forward pass (model(x, bags, query_ids))
4. dime_loss computation
5. Gradient flow (Zygote.gradient through dime_loss)
6. DIMEHeuristic: compute() on a single state

Run with:
    julia --project=. test/test_dime.jl
"""

using DIMEPlanning
using NeuroPlannerExperiments
using NeuroPlanner
using NeuroPlanner.PDDL
using NeuroPlanner.SymbolicPlanners
using NeuroPlanner.Mill.Flux
using Zygote
using Test

# ---- paths ------------------------------------------------------------------
DATA = joinpath(@__DIR__, "..", "NeuroPlannerExperiments.jl", "src", "..",
                "data", "ipc23_learning", "ferry")
DOMAIN_FILE  = joinpath(DATA, "domain.pddl")
PROBLEM_FILE = joinpath(DATA, "training", "easy", "p01.pddl")
PLAN_FILE    = replace(PROBLEM_FILE, ".pddl" => ".plan")

# Shared state (declared at module scope so @testset can write and later testsets can read)
_mb      = nothing
_pddld   = nothing
_domain  = nothing
_problem = nothing
_model   = nothing

@testset "DIMEPlanning integration" begin

    # ------------------------------------------------------------------
    @testset "Data files exist" begin
        @test isfile(DOMAIN_FILE)
        @test isfile(PROBLEM_FILE)
        @test isfile(PLAN_FILE)
    end

    # ------------------------------------------------------------------
    @testset "DIMEMiniBatch construction" begin
        global _domain  = load_domain(DOMAIN_FILE)
        global _problem = load_problem(PROBLEM_FILE)
        plan    = NeuroPlannerExperiments.load_plan(PLAN_FILE)
        global _pddld   = NeuroPlannerExperiments.materialize(
                      Extractor(; architecture="atombinaryfe", graph_layers=2), _domain)

        global _mb = DIMEMiniBatch(_pddld, _domain, _problem, plan)
        @test _mb isa DIMEMiniBatch
        @test !isempty(_mb.query_ids)
        @test length(_mb.Δ_targets) == length(_mb.query_ids)
        @test size(_mb.H₊, 2) == length(_mb.query_ids)
        @test size(_mb.H₋, 2) == length(_mb.query_ids)
        println("  n_states=$(length(_mb.path_cost))  n_queries=$(length(_mb.query_ids))")
    end

    # ------------------------------------------------------------------
    @testset "DIMEModel construction" begin
        conf = Model(
            message_pass_model = FFNN(hidden_dim=16, output_dim=16, layers=1, layernorm=true),
            pooling = "SegmentedSumMax",
            pooled_model = FFNN(hidden_dim=16, output_dim=1, layers=2, layernorm=true)
        )
        global _model = construct_dime_model(_pddld, _problem, conf)
        @test _model isa DIMEModel
        println("  backbone type: ", typeof(_model.backbone))
    end

    # ------------------------------------------------------------------
    @testset "Forward pass" begin
        f_scores, v_scores = _model(_mb.x, _mb.context_bags, _mb.query_ids)
        @test size(f_scores) == (1, length(_mb.query_ids))
        @test size(v_scores) == (1, length(_mb.query_ids))
        @test all(isfinite, f_scores)
        @test all(isfinite, v_scores)
        println("  f range: [$(minimum(f_scores)), $(maximum(f_scores))]")
        println("  v range: [$(minimum(v_scores)), $(maximum(v_scores))]")
    end

    # ------------------------------------------------------------------
    @testset "dime_loss (α=0, recovers pred loss only)" begin
        loss_val = dime_loss(_model, _mb; α=0f0)
        @test loss_val isa Float32
        @test isfinite(loss_val)
        @test loss_val >= 0
        println("  L_pred (α=0): $loss_val")
    end

    @testset "dime_loss (α=0.1, full joint loss)" begin
        loss_val = dime_loss(_model, _mb; α=0.1f0)
        @test loss_val isa Float32
        @test isfinite(loss_val)
        println("  L_total (α=0.1): $loss_val")
    end

    # ------------------------------------------------------------------
    @testset "Gradient flow through dime_loss" begin
        grads = Zygote.gradient(_model) do m
            dime_loss(m, _mb; α=0.1f0)
        end
        @test grads[1] !== nothing
        # Check at least one parameter has a non-zero gradient
        has_grad = Ref(false)
        Flux.fmap(grads[1]) do g
            if g isa AbstractArray && any(!iszero, g)
                has_grad[] = true
            end
            g
        end
        @test has_grad[]
        println("  gradients flow OK")
    end

    # ------------------------------------------------------------------
    @testset "DIMEHeuristic single state evaluation" begin
        hfun = DIMEHeuristic(_pddld, _problem, _model; λ=0.5f0)
        state = PDDL.initstate(_domain, _problem)
        spec  = SymbolicPlanners.Specification(_problem)
        val   = SymbolicPlanners.compute(hfun, _domain, state, spec)
        @test val isa Float32
        @test isfinite(val)
        @test length(hfun.context) == 1   # state was added to context
        println("  a(s₀) = $val")

        # Second call: context should grow
        next_states = NeuroPlanner._next_states(_domain, _problem, state, nothing)
        if !isempty(next_states)
            val2 = SymbolicPlanners.compute(hfun, _domain, first(next_states).state, spec)
            @test isfinite(val2)
            @test length(hfun.context) == 2
            println("  a(s₁) = $val2  (context size=$(length(hfun.context)))")
        end

        reset_context!(hfun)
        @test isempty(hfun.context)
    end

end
