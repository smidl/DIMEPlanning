# DIME: Estimating Conditional Mutual Information for Dynamic Feature Selection

**Source:** Gadgil, Covert, Lee (ICLR 2024)

## Problem Formulation

Let **x** = (x_1, ..., x_d) be input features and **y** a response variable. S denotes a subset of indices, x_S the corresponding features.

The goal is to learn:
- A **predictor** f(x_S) that makes predictions given any feature subset
- A **selection policy** pi(x_S) -> [d] that outputs the next feature to observe

The idealized greedy policy selects the feature with maximum **conditional mutual information (CMI)**:

```
pi*(x_S) = argmax_i I(y; x_i | x_S)
```

where CMI is defined as the KL divergence:

```
I(y; x_i | x_S) = D_KL( p(x_i, y | x_S) || p(x_i | x_S) p(y | x_S) )
```

This policy is near-optimal under certain assumptions (Chen et al., 2015), but difficult to implement because it requires oracle access to p(y | x_S) and p(x_i | x_S).

## Core Method: Discriminative CMI Estimation

DIME parameterizes two networks:

1. **Predictor network** f(x_S; theta) -- e.g., a classifier outputting predictions in the simplex
2. **Value network** v(x_S; phi) in R^d -- estimates CMI for each feature: v_i(x_S; phi) ~ I(y; x_i | x_S)

Both use zero-masking for missing features (plus a binary indicator mask).

### Training Objectives

**Predictor objective** -- accurate predictions given any feature set:

```
min_theta  E_{x,y} E_s [ l(f(x_s; theta), y) ]                  (2)
```

**Value network objective** -- regression on incremental loss improvement:

```
min_phi  E_{x,y} E_s E_i [ (v_i(x_s; phi) - Delta(x_s, x_i, y))^2 ]   (3)
```

where the loss improvement is:

```
Delta(x_S, x_i, y) = l(f(x_S; theta), y) - l(f(x_{S+i}; theta), y)
```

### Key Theoretical Results

**Lemma 1.** When the Bayes classifier p(y | x_S) is used as predictor and l is cross-entropy loss, the incremental loss improvement is an **unbiased estimator** of the CMI:

```
E_{y, x_i | x_S} [ Delta(x_S, x_i, y) ] = I(y; x_i | x_S)
```

*Proof sketch:* Expected cross-entropy with the Bayes classifier equals the conditional entropy H(y | x_S). The expected loss after adding x_i equals H(y | x_S, x_i). Their difference is the CMI by definition.

**Theorem 1.** When l is cross-entropy loss, objectives (2) and (3) are jointly optimized by:
- Predictor: f(x_S; theta*) = p(y | x_S)
- Value network: v_i(x_S; phi*) = I(y; x_i | x_S) for all i

This allows end-to-end training via SGD -- the value network learns the exact CMI at optimality, without generative models.

**Theorem 3 (Regression).** When l is MSE, the value network recovers the reduction in conditional variance:

```
v(x_S; phi*) = Var(E[y | x_S, x_i] | x_S)
```

### Effect of Predictor Suboptimality

If the classifier outputs q(y | x_S) instead of p(y | x_S), the expected loss reduction becomes:

```
E[Delta] = I(y; x_i | x_S) + D_KL(p(y|x_S) || q(y|x_S)) - E_{x_i}[D_KL(p(y|x_S,x_i) || q(y|x_S,x_i))]
```

The bias is a *difference* of KL terms (can be positive or negative) and shrinks to zero as the classifier improves.

## Incorporating Prior Information

Given prior information **z** (e.g., demographic features, low-resolution input), modify:
- Selections based on I(y; x_i | x_S, z)
- Predictions via p(y | x_S, z)

**Theorem 2.** The modified objectives are jointly optimized by f(x_S, z; theta*) = p(y | x_S, z) and v_i(x_S, z; phi*) = I(y; x_i | x_S, z).

## Variable Feature Budgets and Non-Uniform Costs

### Non-Uniform Costs

Each feature has cost c_i > 0. Inspired by adaptive submodular optimization, selections use the cost-normalized CMI:

```
argmax_i  I(y; x_i | x_S) / c_i
```

### Stopping Criteria

Three options from a single trained model:

1. **Budget-constrained:** stop when sum of costs <= k
2. **Confidence-constrained:** stop when H(y | x_S) <= m
3. **Penalized:** stop when max_i I(y; x_i | x_S) / c_i < lambda

**Proposition 1.** Policies with *per-prediction* budget/confidence constraints are Pareto-dominated by those satisfying constraints *on average*. This motivates the penalized stopping criterion, which allows variable per-sample budgets.

*Proof:* The set of budget-constrained policies Pi_k is a subset of unconstrained policies Pi. Optimizing over the larger set cannot yield worse results.

### Training Details

- Pre-train predictor with random feature masks (uniform cardinality sampling)
- Joint training with exploration probability epsilon (annealed toward 0)
- Optional parameter sharing via shared backbone (separate output heads)
- Value network outputs constrained: 0 <= v_i <= H(y | x_S) via sigmoid scaling
- For prior information z: detach gradients in predictor path to prevent overfitting

## Closest Related Work

**Greedy CMI-based methods:**
- **Chen et al. (2015):** Analyzed greedy CMI theoretically, proved near-optimal performance under adaptive submodularity-like conditions. Foundation for the CMI selection criterion.
- **EDDI (Ma et al., 2019):** Uses a partial VAE (generative model) to sample unknown features and estimate CMI. Slow at inference due to iterating over candidates and sampling.
- **Argmax Direct (Chattopadhyay et al., 2023; Covert et al., 2023):** Discriminative methods that directly predict the *argmax* of CMI rather than estimating CMI values. Simpler and faster than generative approaches, but bypass CMI estimation itself, losing the ability to determine when to stop or handle non-uniform costs.

**DIME's position:** Combines the simplicity of discriminative methods with the capabilities of generative ones -- estimates the CMI itself (enabling variable budgets, non-uniform costs, stopping criteria) without requiring generative models.

**RL-based DFS methods:**
- **CwCF (Janisch et al., 2019; 2020):** Formulates DFS as MDP with deep Q-learning. Supports variable budgets and non-uniform costs but suffers from RL training instabilities.
- **OL (Kachuee et al., 2018):** RL approach with Q-network and P-network. Underperforms greedy CMI methods in practice.

**Static feature selection:**
- **CAE (Balin et al., 2019):** Concrete Autoencoder with differentiable selection. Strong static baseline but selects the same features for all samples.

**MI estimation with deep learning:**
- Belghazi et al. (2018), Poole et al. (2019), Song & Ermon (2019): Estimate MI between high-dimensional variables. Unlike DIME, these do not condition on arbitrary feature sets or estimate many CMI terms via a single model.
