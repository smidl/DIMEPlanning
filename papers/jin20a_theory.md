# Provably Efficient Reinforcement Learning with Linear Function Approximation

**Source:** Jin, Yang, Wang, Jordan (COLT 2020)

**Note:** This is an extended abstract; full version at arXiv:1907.05388.

## Core Problem

The fundamental question: **Can we design provably efficient RL algorithms that incorporate function approximation?**

By "efficient" the authors mean:
- **Runtime**: polynomial, independent of number of states
- **Sample complexity**: polynomial, dependent on intrinsic complexity of function class (not |S|)

This is challenging because:
1. Function approximation introduces bias (optimal value may not be representable)
2. Exploration/exploitation tradeoff is harder without visiting all states
3. Sparsity: most state neighborhoods never visited during learning

## Setting: Linear MDP

**Episodic MDP** with:
- State space S (potentially infinite/continuous)
- Action space A
- Episode length H
- Total steps T = K × H (K episodes)
- Feature dimension d

**Linear MDP Assumption:** Both transition dynamics and reward function are linear in a known feature map φ(s,a) ∈ R^d:
```
P(s' | s, a) = ⟨φ(s,a), μ(s')⟩    (for some unknown measure μ)
r(s, a) = ⟨φ(s,a), θ⟩            (for some unknown θ)
```

This implies the optimal Q-function is linear:
```
Q*(s, a) = ⟨φ(s,a), w*⟩
```

## Main Result

**Theorem (informal):** Optimistic LSVI achieves regret:
```
Regret(T) = Õ(√(d³H³T))
```

Key properties:
- **Independent of |S| and |A|** — scales only with feature dimension d
- **√T dependence** — sublinear, approaches optimal rate
- **No simulator required** — learns from online interaction
- **Polynomial runtime:** O(d²AKT) time, O(d²H + dAT) space

## Algorithm: Optimistic LSVI

Based on Least-Squares Value Iteration (Bradtke & Barto, 1996) with **optimistic bonus** for exploration.

### Standard LSVI
Fits Q-function via least squares regression:
```
w_h = argmin_w Σ_τ (⟨φ(s_h^τ, a_h^τ), w⟩ - r_h^τ - max_{a'} Q_{h+1}(s_{h+1}^τ, a'))²
```

### Optimistic Modification
Add exploration bonus based on uncertainty:
```
Q_h(s,a) = min(⟨φ(s,a), w_h⟩ + β · ‖φ(s,a)‖_{Λ_h^{-1}}, H)
```

where:
- Λ_h = Σ_τ φ(s_h^τ, a_h^τ)φ(s_h^τ, a_h^τ)^⊤ + λI is the design matrix
- ‖x‖_{Λ^{-1}} = √(x^⊤ Λ^{-1} x) is the uncertainty measure
- β = Õ(dH) is the bonus coefficient

The bonus ‖φ(s,a)‖_{Λ^{-1}} is large when (s,a) is far from previously visited state-actions in feature space.

## Key Insight: Optimism Principle

**Optimism in the face of uncertainty:**
- Maintain optimistic Q-value estimates
- Bonus encourages visiting uncertain states
- As data accumulates, bonus shrinks → converges to true Q*

This is the RL analogue of UCB (Upper Confidence Bound) from bandits.

## Comparison with Related Settings

### vs. Tabular RL
| Aspect | Tabular | Linear MDP |
|--------|---------|------------|
| Regret | Õ(√(H²SAT)) | Õ(√(d³H³T)) |
| Scales with |S|? | Yes (√S) | No |
| Representation | Table | Features |

Tabular is minimax optimal but cannot handle large S. Linear MDP exploits structure.

### vs. Linear Bandits
Linear bandits are special case with H = 1 (single step).

| Setting | Best Regret |
|---------|-------------|
| Linear bandits | Õ(d√T) |
| Linear contextual bandits | Õ(√(dT)) |
| Linear MDP (this work) | Õ(√(d³H³T)) |

Key difference: MDP has **temporal structure**. Naive adaptation of bandit algorithms yields regret exponential in H.

### vs. Prior Work with Function Approximation

| Prior Work | Limitation |
|------------|------------|
| Yang & Wang (2019a) | Requires simulator |
| Wen & Van Roy (2013, 2017) | Deterministic transitions only |
| Du et al. (2019) | Low variance transitions |
| Yang & Wang (2019b) | Small matrix parametrization |
| Jiang et al. (2017) Olive | Not computationally efficient |

This work: **No simulator, no restrictive assumptions, polynomial runtime.**

## Bellman Rank Connection

Jiang et al. (2017) introduced "Bellman rank" as a complexity measure for RL with function approximation.

**Observation:** Under the linear MDP assumption, Bellman rank ≤ d.

This means the generic Olive algorithm is sample efficient in this setting, but:
- Olive is not computationally efficient
- Olive doesn't achieve √T regret

Optimistic LSVI achieves both.

## Technical Challenges

### Why is MDP harder than bandits?
The temporal structure creates **error propagation**:
- Errors in Q_{h+1} affect the regression target for Q_h
- These errors compound across H steps
- Need careful analysis to prevent exponential blowup

### Exploration Difficulty
Without a simulator:
- Cannot query arbitrary (s,a) pairs
- Must reach states through sequential actions
- Exploration policy affects what data is collected

The optimistic bonus addresses this by encouraging visits to uncertain regions.

## Implications

1. **Theoretical:** First complete answer to fundamental efficiency question for linear MDPs
2. **Practical:** LSVI is a classical algorithm; optimistic version is simple to implement
3. **Foundation:** Linear case provides building block for understanding more complex function classes (neural networks)

## Limitations

- Assumes linear MDP (both transitions and rewards linear)
- Feature map φ(s,a) must be known
- Regret has H³ dependence (later work improved this)
- Full version (arXiv) contains complete proofs and tighter analysis
