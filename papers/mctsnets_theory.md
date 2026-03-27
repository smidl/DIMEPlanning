# Learning to Search with MCTSnets

**Source:** Guez, Weber, Antonoglou, Simonyan, Vinyals, Wierstra, Munos, Silver (ICML 2018)

## Core Idea

MCTSnets incorporate simulation-based search **inside** a neural network by treating the search algorithm as a differentiable computation graph. Instead of hand-designing the rules for MCTS (where to traverse, what to evaluate, how to back-up), all components are learned end-to-end via gradient-based optimization.

The key insight: represent the internal state of the search at each node by a **memory vector** h ∈ R^n, and learn neural networks to manipulate these vectors during the search process.

## Background: Value-Network MCTS

Standard MCTS (Algorithm 1) has four stages per simulation:

1. **Forward simulation:** From root s_A, traverse tree by sampling actions from simulation policy π(a | s_t, {N(s_t), N(s_t,a), Q(s_t,a)}) until reaching a leaf node (N(s_t) = 0).

2. **Evaluate leaf:** Obtain value estimate V(s_L) at depth L.

3. **Back-up:** Update Q-values along the traversed path toward Monte-Carlo return:
   ```
   Q(s,a) ← Q(s,a) + 1/(N(s,a)+1) * (R_t - Q(s,a))
   ```
   where R_t = Σ_{t'=t}^{L-1} γ^{t'-t} r_{t'} + γ^{L-t} V(s_L)

4. **Update visit counts:** N(s) ← N(s) + 1, N(s,a) ← N(s,a) + 1

The simulation policy π balances exploration and exploitation (e.g., UCT: π(s) = argmax_a Q(s,a) + c√(log N(s) / N(s,a))).

## MCTSnet Architecture

MCTSnet generalizes MCTS by replacing scalar statistics {N(s), N(s,a), Q(s,a)} with **vector-valued memory** h_s ∈ R^n, and replacing hand-coded rules with learned neural networks.

### Components

| MCTS Component | MCTSnet Equivalent | Parameters |
|----------------|-------------------|------------|
| Initialize statistics | **Embedding network** ε(s; θ_e) | θ_e |
| Simulation policy | **Simulation policy** π(a \| h_s; θ_s) | θ_s |
| Value evaluation | (implicit in embedding) | — |
| Backup rule | **Backup network** β(h_parent, h_child, r, a; θ_b) | θ_b |
| Final decision | **Readout network** ρ(h_{s_A}; θ_r) | θ_r |

### Algorithm (MCTSnet)

For m = 1...M simulations:

1. **Forward simulation** from root state s_0 = s_A until leaf node (N(s_t) = 0):
   - Sample action a_t ~ π(a | h_{s_t}; θ_s)
   - Compute reward r_t = r(s_t, a_t) and next state s_{t+1} = T(s_t, a_t)

2. **Evaluate leaf** s_L at depth L:
   - Initialize memory: h_{s_L} ← ε(s_L; θ_e)

3. **Back-up phase** from leaf s_L, for each t < L:
   - Update parent memory using backup network:
     ```
     h_{s_t} ← β(h_{s_t}, h_{s_{t+1}}, r_t, a_t; θ_b)
     ```

After M simulations, **readout** action distribution from root memory: ρ(h_{s_A}; θ_r).

### Memory Update Equations

For a fixed tree expansion with path p^{m+1} = s_0, a_0, s_1, a_1, ..., s_L for simulation m+1:

**Leaf initialization:**
```
h_{s_L}^{m+1} ← ε(s_L)
```

**Backup (nodes on path):**
```
h_{s_t}^{m+1} ← β(h_{s_t}^m, h_{s_{t+1}}^{m+1}, r(s_t, a_t), a_t; θ_b)
```

**Non-visited nodes:**
```
h_s^{m+1} ← h_s^m    (memory unchanged)
```

The tree path is sampled as:
```
p^{m+1} ~ P(s_0 a_0 ... s_L | h^m) ∝ Π_{t=0}^{L-1} π(a_t | h_{s_t}^m; θ_s) 𝟙[s_{t+1} = T(s_t, a_t)]
```

## Network Design Choices

### Backup Network β

Uses a **gated residual connection** to selectively incorporate information from the subtree:
```
h_s ← β(φ; θ_b) = h_s + g(φ; θ_b) · f(φ; θ_b)
```
where:
- φ = (h_s, h_{s'}, r, a) combines parent statistics, child statistics, reward, and action
- g(φ; θ_b) ∈ [0,1] is a learned gating function
- f(φ; θ_b) is the learned update function

The gating allows the network to ignore information from certain subtrees when appropriate.

### Simulation Policy π

**Basic form:** MLP mapping statistics h_s to action logits ψ(s,a):
```
π(a | s; θ_s) ∝ exp(ψ(s,a))
```

**Modulated form:** Combines a learned policy prior with child statistics:
```
ψ(s,a) = w_0 ψ_p(s,a) + w_1 u(h_s, h_{T(s,a)})
```
where:
- ψ_p(s,a) = log μ(s,a; θ_p) is a policy prior (similar to PUCT)
- u is a small network combining parent and child statistics
- This mirrors how PUCT uses both prior policy and value estimates

### Embedding ε and Readout ρ

- **Embedding:** Standard residual convolutional network; initializes h at tree nodes
- **Readout:** Simple MLP transforming root memory h_{s_A} to action distribution

## Training

### Loss Function

Given state s and target action a*, the loss is:
```
ℓ(s, a*) = E_{z~π(z|s)} [-log p_θ(a* | s, z)]
```
where z = z_{≤M} is the set of all stochastic actions taken during M simulations.

This is a lower bound on the log-likelihood of the marginal p_θ(a | s).

### Gradient Estimation

The gradient has two terms:
```
∇_θ ℓ(s, a*) = -E_z [∇_θ log p_θ(a* | s, z) + (∇_θ log π(z | s; θ_s)) log p_θ(a* | s, z)]
```

1. **First term:** Gradient through the differentiable path (embedding, backup, readout)
2. **Second term:** REINFORCE gradient for the simulation policy (non-differentiable action selection)

### Credit Assignment for Anytime Algorithms

The REINFORCE gradient has high variance because many stochastic decisions (O(M log M) to O(M²)) contribute to a single decision a*.

**Key insight:** Cast loss minimization as a sequential decision problem. Define:
- ℓ_m = ℓ(p_θ(a | s, z_{≤m})) — loss after m simulations
- r̄_m = -(ℓ_m - ℓ_{m-1}) — surrogate reward (loss reduction at step m)
- R_m = Σ_{m'≥m} r̄_{m'} — return from step m

The REINFORCE term becomes:
```
A = Σ_m ∇_θ log π(z_m | s, z_{<m}; θ_s) R_m
```

**Variance reduction:** Use baselined, discounted returns:
```
R_m^γ = Σ_{m'≥m} γ^{m'-m} r̄_{m'}
```

With γ < 1, later (noisier) improvements are down-weighted. The final gradient:
```
∇_θ ℓ(s, a*) = E_z [-∇_θ log p_θ(a* | s, z) + Σ_m ∇_θ log π(z_m | s; θ_s) R_m^γ]
```

## Key Properties

### Tree-Structured Memory

Unlike flat RNNs, MCTSnet maintains **tree-structured memory**: each node s_k has its own statistics h_k. This enables:
- Partial replanning: after taking action a from root s_A to s'_A, initialize the network for s'_A using the subtree rooted at s'_A with previously computed statistics
- Recurrence across real time-steps

### Anytime Behavior

MCTSnet can run for arbitrary M ≥ 1 simulations. With larger M:
- More opportunities to query the environment model
- Statistics become richer and more informative
- Performance generally improves (Figure 5 shows monotonic improvement up to M=100)

### Computation Flow

The computation depends not just on the final tree, but on the **order in which nodes are visited** (the tree expansion process). This distinguishes MCTSnet from static architectures that process a fixed tree.

## Relation to Other Approaches

**vs. Standard MCTS:** MCTSnet replaces scalar statistics (N, Q) with vector memories h, and hand-coded rules with learned networks. Both use the same environment model T(s,a), r(s,a) at test time.

**vs. AlphaGo/AlphaZero:** AlphaGo uses a fixed MCTS with learned policy/value networks. MCTSnet learns the search procedure itself, not just the evaluation functions.

**vs. TreeQN (Farquhar et al., 2017):** TreeQN performs planning over a fixed-depth tree expansion using an implicit transition model. MCTSnet uses an explicit transition model and learns the simulation/backup strategy.

**vs. I2A (Weber et al., 2017):** I2A aggregates results of several simulations into the network computation. MCTSnet generalizes this with tree-structured memory and learned tree expansion (rather than rolling out each action from root).

**vs. Predictron (Silver et al., 2017b):** Predictron aggregates over multiple simulations using an implicit model. MCTSnet uses explicit tree structure and learned expansion strategy.

## Theoretical Interpretation

MCTSnet can be viewed from two perspectives:

1. **Search algorithm:** A generalization of MCTS where statistics, simulation policy, and backup rule are all learned rather than hand-designed.

2. **Neural network:** A stochastic feed-forward network with:
   - Single input (initial state)
   - Single output (action distribution)
   - Tree-structured memory with skip connections
   - Control flow determined by learned simulation policy

The architecture maintains desirable structural properties of MCTS (model-based, iterative local computation, structured memory) while allowing flexibility through learning.
