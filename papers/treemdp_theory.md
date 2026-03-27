# Tree-MDP: Treating Tree Search as Markov Decision Process

**Source:** (IJCAI 2025 submission)

## Problem Formulation

Given a set of states S, actions A, and a deterministic transition function delta. The goal, given initial state s_0, is to find a sequence of actions (a_0, ..., a_{n-1}) such that s_{i+1} = delta(s_i, a_i) and s_n maximizes a bounded objective function o: S -> R.

The standard MDP formulation M = (S, A, delta, R) uses reward R(s, a) = o(delta(s, a)) - o(s), so the cumulative reward telescopes to o(s_n) - o(s_0). For satisfying planning, reward is non-zero only at goal states (sparse reward).

**Key insight:** Standard RL optimizes policies/value functions for agents acting in the original problem space. But forward search algorithms operate in a *different* space -- they incrementally build a search tree by selecting which nodes to expand. Naively applying RL ignores the dynamics of the search process itself.

## Core Method: Tree-MDP Formulation

For a given S, A, delta, objective o, and initial state s_0, the Tree-MDP is M_tilde = (S_tilde, A_tilde, delta_tilde, R_tilde):

- **States S_tilde:** The set of states a forward-search algorithm can attain from s_0. For GBFS, this corresponds to the search tree + open list + closed list. Each state T is a *whole partially expanded search tree*.
- **Actions A_tilde:** At state T, the set of nodes in the open list that can be expanded (i.e., an action corresponds to expanding a leaf node).
- **Transition delta_tilde:** Expansion of a leaf in GBFS -- generate all children, resolve duplicates, append to search tree.
- **Reward R_tilde(T, a_tilde):** Rewards finding a state improving o:

```
R_tilde(T, a_tilde) = max_{s in delta_tilde(T, a_tilde)} o(s) - max_{s in T} o(s)
```

Tree-MDP is a valid MDP: (i) Markov property is satisfied, (ii) reward is maximized when the state maximizing o is reached.

### Policy Compatible with GBFS

To make the policy usable as a GBFS heuristic, define:

```
pi(a_tilde = s | T) = exp(-h(s)) / sum_{s' in O(T)} exp(-h(s'))
```

where O(T) is the open list and h is the heuristic function. The state with minimal heuristic value has maximum probability. Since h does not depend on the Tree-MDP state T, it can be used directly as a GBFS heuristic.

## Relation of Tree-MDP to Ranking Losses

The L_gbfs loss [Chrestien et al., 2023; Piepenbrock et al., 2022; Hao et al., 2024] for imitation learning optimizes search efficiency by forcing states on the example trajectory to rank highest:

```
L_gbfs(h, xi) = sum_{i=1}^{n} sum_{s' in O(T_i)} ell(h(s_i) - h(s'))
```

Replacing the inner sum with logsoftmax gives:

```
L_softmax(h, xi) = - sum_{i=1}^{n} log( exp(-h(s_i)) / sum_{s' in O(T_i)} exp(-h(s')) )
```

The term inside the sum is exactly the Tree-MDP policy pi(a_tilde | T) from Equation (3). This reveals that **L_gbfs with softmax is equivalent to policy optimization in Tree-MDP**.

This also explains why ranking-based heuristics empirically outperform cost-to-goal estimation -- it parallels the observation in RL that policy methods often outperform value-based methods.

## Reward Design Challenges

### Definition: Natural Order

A reward function G(xi) satisfies the **natural order** property if:

1. If o(s'_{n'}) > o(s''_{n''}), then G(xi') > G(xi'') -- better solutions always preferred regardless of path length.
2. If o(s'_{n'}) = o(s''_{n''}) and n' < n'', then G(xi') > G(xi'') -- among equal-quality solutions, shorter paths preferred.

### Case gamma = 1

G(xi) = o(s_n) - o(s_0) captures maximizing o(s_n) but does **not penalize path length**. Multiple paths to the goal appear equally good. In pathological cases the optimal heuristic becomes non-informative.

### Case gamma < 1

G(xi) = sum gamma^i (o(s_{i+1}) - o(s_i)) satisfies natural order, but gamma too small introduces **local minima**: the agent prefers nearby suboptimal goals over distant optimal ones when n > 1 - log(2)/log(gamma). GBFS mitigates dead ends via the open list, but local minima still slow down search.

## Optimizing Policy without Explicit Reward

An alternative avoids defining a reward altogether by assuming an absolute ordering of trajectories.

### Bootstrapping with L_gbfs

Let xi* be the best trajectory in the current search tree (shortest path to the best node). Define an empirical policy pi* that deterministically follows xi*. Minimizing KL(pi* || pi_theta) yields:

```
- sum_{i=1}^{n} log( exp(-h(s*_i)) / sum_{s' in O(T_i)} exp(-h(s')) )
```

This **recovers the L_gbfs loss with logsoftmax** and corresponds to a bootstrapping protocol [Arfaee et al., 2011].

### Policy-Optimizing Loss L_po (Key Contribution)

The above L_gbfs loss may be **off-policy** since the best trace xi* can differ greatly from the actual GBFS expansion sequence.

Let (s_bar_0, ..., s_bar_n_bar) be the states actually expanded by GBFS, producing Tree-MDP states (T_bar_1, ..., T_bar_{n_bar+1}). Let xi* again be the shortest path to the best state. The best known action at Tree-MDP state T_bar_i is to expand the state from the open list closest to s*_n:

```
s_bar*_{T_bar_i} = argmax_{s*_j in O(T_bar_i)} j
```

The loss becomes:

```
L_po = - sum_{i=1}^{n_bar} log( exp(-h(s_bar*_{T_bar_i})) / sum_{s' in O(T_bar_i)} exp(-h(s')) )
```

**Key properties of L_po:**
- Combines planning (extraction of best trajectory) with RL (using the actual GBFS expansion sequence)
- If an important state is frequently not expanded, it appears in multiple open lists and automatically receives higher priority
- Does not require an explicitly defined reward function -- only a total ordering on trajectories
- Complexity is linear (not quadratic as in original L_gbfs)

## Closest Related Work

**Ranking-based heuristic optimization:**
- **Chrestien et al. (2023):** Proposed L_gbfs for imitation learning, showed ranking-based heuristics outperform cost-to-goal estimation. Foundation that Tree-MDP generalizes.
- **Piepenbrock et al. (2022); Hao et al. (2024):** Learned pairwise rankings for GBFS/A*. Tree-MDP exposes these as policy gradient methods.

**RL for heuristic search:**
- **Orseau et al. (2018):** Derives GBFS heuristic from an A3C-trained Linear-MDP policy network. Does not consider tree search dynamics. Subsequent work [Orseau and Lelis, 2021] uses bootstrap, removing the trained policy network.
- **Bhardwaj et al. (2017):** Formulates a problem similar to Tree-MDP but trains by imitating oracles with full environment knowledge. Tree-MDP uses no oracle.
- **Bernhard et al. (2018):** Estimates cost-to-goal from Q-values in the original problem formulation; does not consider GBFS dynamics.

**RL without tree search:**
- **Chen and Tian (2019); Zoph and Le (2017):** Use RL to solve problems directly without tree search, losing advantages like duplicate elimination and backtracking.

**Tree-structured RL:**
- **TreeQN [Farquhar et al., 2017]:** Constructs a small tree to improve Q-value precision, but the tree is not used for search.
- **MCTSNets [Guez et al., 2018]:** Generalizes MCTS with trainable networks. Different search algorithm (MCTS vs GBFS).
- **AlphaZero [Silver et al., 2016]:** Optimizes MCTS for games rather than planning.

**Neural network heuristics for planning:**
- Representation and architecture research [Horčík and Šír, 2024; Toyer et al., 2020; Chen et al., 2024; Ferber et al., 2020; Ståhlberg et al., 2022] is orthogonal -- Tree-MDP addresses *how* to optimize the heuristic, not *how* to represent states.
