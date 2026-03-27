# Papers

This directory contains PDFs and theory summaries for all papers directly relevant to
this project. When a new paper becomes relevant, add it here following the same convention.

---

## Convention for Adding New Papers

1. **Add the PDF**: `papers/{key}.pdf` where `{key}` is the first author + year
   (e.g., `chrestien2023.pdf`, or a descriptive short name like `leah2022.pdf`)

2. **Write a theory summary**: `papers/{key}_theory.md` with this structure:
   ```
   # Full Paper Title

   **Source:** Author(s), Venue Year

   ## Problem Formulation
   ## Core Method / Key Ideas
   ## Key Theoretical Results
   ## Closest Related Work
   ```
   Focus on: formal definitions, theorem statements, what the paper proves vs assumes.
   Do NOT summarise experimental results unless they directly inform our method.

3. **Add a pointer in this README** under the appropriate section below.

4. **When citing in docs/**: Use the format `Author et al. (Year)` and reference the
   `_theory.md` file for the formal statement. Never paraphrase key claims from memory
   — always check the theory file.

---

## Papers Index

### Core Theory

| Key | Paper | Theory Summary | Notes |
|---|---|---|---|
| `leah2022` | Chrestien, Pevný, Edelkamp, Komenda. *Optimize Planning Heuristics to Rank, not to Estimate Cost-to-Goal.* NeurIPS 2023. | [leah2022_theory.md](leah2022_theory.md) | Introduces L_gbfs, L_01 ranking loss, perfect ranking heuristic definition. Public code. |
| `treemdp` | (IJCAI 2025 submission). *Tree-MDP: Treating Tree Search as Markov Decision Process.* | [treemdp_theory.md](treemdp_theory.md) | Introduces Tree-MDP formulation, L_po (on-policy) loss. Code in PrivateNeuroPlanner.jl. |
| `dime` | Gadgil, Covert, Lee. *Estimating Conditional Mutual Information for Dynamic Feature Selection.* ICLR 2024. | [dime_theory.md](dime_theory.md) | DIME Lemma 1: value network → CMI. Acquisition function a(S,i) = log f(x_S) + λ v(x_S). |

### Background

| Key | Paper | Theory Summary | Notes |
|---|---|---|---|
| `mctsnets` | Guez et al. *Learning to Search with MCTSnets.* ICML 2018. | [mctsnets_theory.md](mctsnets_theory.md) | Learned MCTS backup via tree-conditioned memory h_s. Foundation for MCTS-DIME. |
| `jin20a` | Jin, Yang, Wang, Jordan. *Provably Efficient RL with Linear Function Approximation.* COLT 2020. | [jin20a_theory.md](jin20a_theory.md) | Optimistic LSVI, linear MDP regret bounds. Background on exploration theory. |

---

## Critical Distinction: LEAH vs TreeMDP

These are **two separate papers** that are easy to conflate because they share an author
(Pevný) and build on the same L_gbfs loss:

### LEAH (leah2022)
- **Paper**: Chrestien et al., NeurIPS 2023
- **Contribution**: Proposes L_gbfs (ranking loss over open list), proves it outperforms
  L_2 regression (h* estimation). Introduces the perfect ranking heuristic concept.
  Defines GBFS training as ranking on-path vs off-path nodes.
- **Code**: Public — `github.com/aicenter/Optimize-Planning-Heuristics-to-Rank`
  (also in `NeuroPlannerExperiments.jl` submodule, which implements all LEAH losses
  and architectures)
- **Key claim**: Ranking loss ≡ better convergence rate AND handles dead-ends naturally

### TreeMDP (treemdp)
- **Paper**: IJCAI 2025 submission (unpublished, handle carefully)
- **Contribution**: Reframes GBFS as an MDP over partially-expanded search trees
  (Tree-MDP). Shows L_gbfs with softmax = policy gradient in Tree-MDP. Introduces
  L_po (on-policy loss using actual GBFS expansion sequence) which avoids off-policy
  distribution shift.
- **Code**: Private — `PrivateNeuroPlanner.jl` submodule
- **Key claim**: L_po is more stable than L_gbfs because it uses the actual expansion
  trace (not the optimal trace, which can diverge from what GBFS actually expanded)

### When to cite which
- Cite **LEAH** when referring to: ranking vs regression, L_gbfs loss definition,
  the experimental pipeline (NeuroPlannerExperiments.jl), GNN architectures
- Cite **TreeMDP** when referring to: the MDP formulation of GBFS, L_po loss,
  the theoretical justification for why L_gbfs ~ policy gradient
- Both: they are cited together when referring to the general framework of
  learned GBFS heuristics
