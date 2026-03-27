# Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal

**Source:** Chrestien, Pevný, Edelkamp, Komenda (NeurIPS 2023)

## Problem Formulation

A search problem instance is defined by a directed weighted graph Γ = (S, E, w) with:
- S: set of states
- s_0 ∈ S: initial state
- S* ⊆ S: goal states
- w: E → R^{≥0}: edge weights (action costs)

A path (plan) π = (e_1, ..., e_l) solves task Γ iff it connects s_0 to some s_l ∈ S*. An optimal path π* minimizes the cost w(π*) = Σ w(e_i).

**Forward search** uses a merit function f(s) = αg(s) + βh(s) where:
- g(s): accumulated cost from s_0 to s
- h(s): heuristic function
- α = β = 1 for A*, α = 0 and β = 1 for GBFS

## Key Insight: Strict Optimal Efficiency

**Definition (Strict optimal efficiency):** Forward search is *strictly optimally efficient* iff it expands *only* states on one optimal solution path. If the optimal path has l+1 states, the search expands only l states.

**Key observation:** The popular cost-to-goal heuristic h* is optimal for A* (in terms of node expansions up to tie-breaking), but h* does **not guarantee** strict optimal efficiency. Even with perfect h*, GBFS may not find optimal solutions.

The paper argues: **what matters for efficiency is the ranking of states in the Open list, not the absolute heuristic values.**

## Perfect Ranking Heuristic

**Definition 1 (Perfect ranking heuristic).** A heuristic h(s) is a *perfect ranking* in forward search with merit function f(s) = αg(s) + βh(s) for problem instance Γ iff there exists an optimal plan π = ((s_0,s_1), ..., (s_{l-1}, s_l)) such that:

- g(s) is the cost from s_0 to s expanding only states on π
- ∀i ∈ {1,...,l} and ∀s_j ∈ O_i \ S^{π_i}: f(s_j) > f(s_i)

where O_i is the Open list after expanding states (s_0, ..., s_{i-1}) and S^{π_i} is the set of states on the optimal subpath up to step i.

**In words:** At every step, the state on the optimal path has strictly lower merit than all off-path states in the Open list.

**Theorem 1.** Forward search with merit function f(s) = αg(s) + βh(s) and heuristic h is strictly optimally efficient on problem instance Γ **if and only if** h is a perfect ranking on Γ.

*Proof sketch:*
- **Sufficiency:** If h is a perfect ranking, the state on the optimal path always has the lowest merit, so it's always selected. The search never leaves the optimal path.
- **Necessity:** If the search is strictly optimally efficient, it always selects the on-path state, which means that state must have had the lowest merit at each step — precisely the perfect ranking condition.

## Why Cost-to-Goal is Suboptimal

**Example 1.** While h* is the best heuristic for A* (up to tie-breaking), for GBFS, h* does not necessarily yield optimal solutions.

Consider graph with nodes A, B, C, D, E:
- w(A,B) = 8, w(B,E) = 3, w(A,C) = 2, w(C,D) = 4, w(D,E) = 4
- h*(A) = 10, h*(B) = 3, h*(C) = 8, h*(D) = 4, h*(E) = 0
- s_0 = A, goal E ∈ S*

GBFS with h* returns path (A, B, E) with cost 11, but optimal path (A, C, D, E) has cost 10.

**Key problems with regression on h*:**

1. **False sense of optimality:** A* with h* is optimal only up to tie-breaking. With many equal-cost optimal solutions, A* may be inefficient unless ties are broken well.

2. **Solution set size:** The set of functions satisfying the ranking loss is likely larger than those satisfying regression loss (which targets exact h* values). Ranking is invariant to strictly monotone transformations.

3. **Regression ignores off-path states:** The L_2 regression loss uses only states on the optimal path. Even with zero loss, the search can be inefficient if off-path states have lower heuristic values than on-path states.

4. **Dead-end handling:** h* = ∞ for dead-ends causes gradient issues in L_2 loss. Ranking losses handle this naturally.

## Loss Functions

### Ranking Loss (L_01)

The ideal loss counts violated ranking conditions:

```
L_01(h, Γ, π) = Σ_{s_i ∈ S^π} Σ_{s_j ∈ O_i \ S^{π_i}} [[r(s_i, s_j, θ) > 0]]
```

where:
```
r(s_i, s_j, θ) = α(g(s_i) - g(s_j)) + β(h(s_i, θ) - h(s_j, θ))
```

and [[·]] is the Iverson bracket (1 if true, 0 otherwise).

### Surrogate Ranking Loss (L_gbfs)

Replace the 0-1 loss with a convex surrogate (logistic loss):

```
L_gbfs(h, Γ, π) = Σ_{s_i ∈ S^π} Σ_{s_j ∈ O_i \ S^{π_i}} log(1 + exp(r(s_i, s_j, θ)))
```

For A* (α = β = 1), use L* with the same formula.

For GBFS (α = 0, β = 1):
```
r(s_i, s_j, θ) = h(s_i, θ) - h(s_j, θ)
```

### Regression Loss (L_2)

The standard approach optimizes mean squared error on cost-to-goal:

```
L_2 = Σ_{s_i ∈ S^π} (h(s_i, θ) - h*(s_i))^2
```

**Critical difference:** L_2 uses only on-path states. L_gbfs uses both on-path and off-path states (the entire Open list at each step).

## Theoretical Advantages of Ranking Losses

### Convergence Speed

From VC theory:
- **Ranking loss** (classification-like): excess error converges at rate √(ln n_p) / √n_p
- **Regression loss:** excess error converges at rate √(ln n_s) / (√n_s - √(ln n_s))

where n_p is the number of state pairs and n_s is the number of states.

Since n_p grows quadratically with states, ranking loss converges at least 1/√n_s factor faster than regression.

### Invariance Properties

Ranking loss is invariant to:
- Translation: h(s) + c gives the same ranking
- Strictly monotone transformations: any f(h(s)) where f is strictly increasing

Regression loss requires exact values, which is a stronger (unnecessary) requirement.

### Dead-End Handling

- **Ranking:** Dead-ends just need higher values than on-path states
- **Regression:** h* = ∞ causes gradient issues; finite approximations can destabilize optimization

## Key Theoretical Results

**Theorem 2.** Let h be a perfect ranking for A* search on problem instance Γ with constant non-negative action costs. Then h is also a perfect ranking for GBFS on Γ.

*Proof:* By induction on expanded nodes. If h is strictly optimally efficient for A*, the ranking inequalities h(s_j) > h(s_i) hold for off-path states. These same inequalities ensure GBFS (which uses only h) also selects on-path states.

**Note:** The converse is not true — a perfect ranking for GBFS may not be perfect for A*.

**Appendix 8.4:** There exist problem instances where no strictly optimally efficient heuristic exists for GBFS. The perfect ranking definition requires specific relationships that may be unsatisfiable for some graph structures.

## Relation to Other Losses

### L_rt (Ranking on trajectory only)

```
L_rt(h, Γ, π) = Σ_{(s_{i-1}, s_i) ∈ π} log(1 + exp(h(s_i, θ) - h(s_{i-1}, θ)))
```

Compares only consecutive states on the optimal trajectory. Simpler but less informative than L_gbfs.

### L_be (Bellman-inspired)

Includes Bellman-like constraints requiring parent nodes to have higher heuristic than children, plus cost-to-goal regression. More complex, less effective than pure ranking.

### L_le (Levin-style)

Combines policy and heuristic from [Orseau & Lelis, 2021]. Trains neural network with two heads. Better than L_2 but worse than L_gbfs in experiments.

## Closest Related Work

**Orseau et al. (2018); Orseau & Lelis (2021):** Derive GBFS heuristics from policy networks, with bounds on maximum expansions. Their neural network estimates both policy and heuristic. This paper shows pure ranking on the heuristic is simpler and more effective.

**Garrett et al. (2016):** Learning to rank for synthesizing planning heuristics. Valid for GBFS only, doesn't handle ties, requires true cost-to-goal for training set construction.

**Ståhlberg et al. (2022):** Learning optimal policies with GNNs. Uses Bellman-inspired loss forcing children to have smaller heuristic than parent. This paper's L_gbfs outperforms this approach.

**Wilt & Ruml (2012, 2016); Xu et al. (2009):** Statistical analysis of heuristics, learning linear ranking functions. Recognized importance of ranking but didn't provide the formal characterization (Definition 1, Theorem 1) or neural network instantiation.
