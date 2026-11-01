# Mixed-Integer Linear Programming Solver (Gomory Cutting Planes)

A Mixed-Integer Linear Programming (MILP) solver built on top of the **Two-Phase Simplex** algorithm, using **Gomory cutting planes** to enforce integrality constraints, with a branch-and-bound finish when the cuts stop making progress.

---

## Features

✅ **Solve MILP / LP Problems**
- **Minimization** and **Maximization** problems
- **Pure LP** (no integrality), **Pure ILP** (all variables integer), or **Mixed-Integer LP**
- **Positive** (≥ 0), **Negative** (≤ 0), and **Free** (unrestricted) variables
- Constraint types: `<=`, `>=`, `=`

✅ **Two-Phase Simplex Engine (LP Relaxation)**
- Phase I: Find initial feasible solution using artificial variables
- Phase II: Optimize the objective function
- Negative RHS normalization
- **Dantzig's rule** for speed, switching to **Bland's Rule** on degenerate stalls (no cycling)
- Vectorized NumPy pivots

✅ **Gomory Cutting Planes**
- **Gomory mixed-integer (GMI) cuts**, valid when continuous variables are present
  (the plain fractional cut is not, and used to cut off the true optimum)
- Several cuts per round, **Dual Simplex** to restore feasibility after each round
- Inactive cuts are removed, so the tableau does not keep growing
- Numerically unsafe cuts (nearly-integral rows, badly scaled coefficients) are skipped

✅ **Branch-and-Bound Finish**
- If the cuts stall, the cut-strengthened LP is finished by best-first branch-and-bound
  with plunging, re-optimized with the dual simplex at every node
- Prunes with objective integrality (e.g. if all costs are multiples of 4, so is the optimum)
- Every answer reported as optimal is proven optimal

✅ **Variable Types**
- `1` = Positive (`x ≥ 0`)
- `2` = Negative (`x ≤ 0`)
- `3` = Free (unrestricted, split internally as `x = x⁺ − x⁻`)

---

## Installation

No external dependencies beyond NumPy.

```bash
pip install numpy
python milp_gomory.py
```

---

## How It Works

### Algorithm Overview

```
1. Read the LP/MILP problem
2. Solve the LP relaxation (ignore integrality)
3. If all integer-required variables are already integral → done
4. Take the most fractional integer variables in the current basis
5. Generate a Gomory mixed-integer cut from each of those tableau rows
6. Append the cuts; run Dual Simplex; drop cuts that are no longer binding
7. Go to step 3 while the cuts make progress
8. Otherwise finish with branch-and-bound on the strengthened LP
```

### Gomory Mixed-Integer Cut

For a canonical tableau row `i` with basic variable `x_Bi`:

```
x_Bi + Σ_j ā_ij · x_j = b̄_i
```

Let `f₀ = frac(b̄_i)` and `f_j = frac(ā_ij)`. Each non-basic variable gets the coefficient:

```
integer x_j,     f_j ≤ f₀ :  f_j
integer x_j,     f_j > f₀ :  f₀ (1 − f_j) / (1 − f₀)
continuous x_j,  ā_ij ≥ 0 :  ā_ij
continuous x_j,  ā_ij < 0 :  f₀ (−ā_ij) / (1 − f₀)
```

and the cut is:

```
Σ_j  coef_j · x_j  ≥  f₀
```

Which is inserted into the tableau as:

```
−Σ_j coef_j · x_j  +  s  =  −f₀     (s ≥ 0, new slack)
```

The new slack starts at `−f₀ < 0` and the Dual Simplex restores feasibility.

On a pure-integer row this is at least as strong as Gomory's fractional cut
`Σ_j f_j x_j ≥ f₀`; with continuous variables the fractional cut is **not valid**.
Slack variables of constraints with integer data on integer variables are treated as integer.

### Branch-and-Bound Finish

Pure cutting planes tend to stall (each new cut only shaves off a little) and lose accuracy
in floating point. When there has been no objective progress for `STALL_ROUNDS` rounds,
`MAX_CUTS` cuts have been added, or no safe cut remains, the solver branches on the most
fractional integer variable (`x_j ≤ ⌊v⌋` / `x_j ≥ ⌈v⌉`). Each bound is added as a tableau row
and the node is re-optimized with the dual simplex.

- **Plunging:** after branching, the nearer child is solved right away, warm-started from
  its parent's tableau, for up to `PLUNGE_DEPTH` levels. This finds integer solutions early.
- **Best bound first:** the other children wait in a queue ordered by their LP bound. This
  prevents endless dives along unbounded integer variables. Queued nodes store only their
  bounds, not a tableau, so memory stays small.
- **Pruning:** a node is discarded when its LP bound cannot beat the best integer solution
  found so far (rounded down to a multiple of the objective's gcd when all costs are integers
  on integer variables).

---

## Usage Example

### Example 1: Pure Integer LP

```
maximize:    x1 + x2
subject to:
             2*x1 + 2*x2 <= 7
             x1, x2 >= 0   and   integer
```

**Input steps:**

```
Number of decision variables: 2
Enter 2 objective coefficients: 1 1
Maximization problem? (yes/no): yes
Number of constraints: 1
Constraint 1 (coefficients + RHS): 2 2 7
Type (<= / >= / =): <=
Variable bounds:  1 = x>=0   2 = x<=0   3 = free
  x1: 1
  x2: 1

Integrality:
  Enter 1-based indices of integer variables (space-separated).
  Integer variable indices: 1 2
```

**Output:**

```
==========================================================
  LP RELAXATION
==========================================================

  LP relaxation solution:
    x1 = 3.500000  [integer required]
    x2 = 0.000000  [integer required]
  Objective = 3.500000

==========================================================
  GOMORY CUTTING PLANE ITERATIONS
==========================================================

  Cut   1 | x1 = 3.500000 (frac = 0.500000)
          | obj = 3.000000
          | [*] x1 = 3.000000
          | [*] x2 = 0.000000

==========================================================
  OPTIMAL SOLUTION
==========================================================
  x1 = 3.000000  [integer]
  x2 = 0.000000  [integer]

  Optimal objective value = 3.000000
==========================================================
```

---

### Example 2: Mixed-Integer LP

```
maximize:    5*x1 + 4*x2
subject to:
             x1 + x2 <= 5
             10*x1 + 6*x2 <= 45
             x1 >= 0 (integer),  x2 >= 0 (continuous)
```

**Input:**

```
Number of decision variables: 2
Enter 2 objective coefficients: 5 4
Maximization problem? (yes/no): yes
Number of constraints: 2
Constraint 1 (coefficients + RHS): 1 1 5
Type (<= / >= / =): <=
Constraint 2 (coefficients + RHS): 10 6 45
Type (<= / >= / =): <=
  x1: 1
  x2: 1
Integer variable indices: 1
```

**Output (final part):**

```
  Cut   1 | x1 = 3.750000 (frac = 0.750000)
          | obj = 23.333333
          | [*] x1 = 4.000000
          | [ ] x2 = 0.833333

==========================================================
  OPTIMAL SOLUTION
==========================================================
  x1 = 4.000000  [integer]
  x2 = 0.833333

  Optimal objective value = 23.333333
==========================================================
```

(The earlier fractional-cut version answered `x1 = 3, x2 = 2, objective 23`, which is not optimal.)

---

### Example 3: Pure LP (no integrality)

Simply press **Enter** when asked for integer variable indices:

```
Integer variable indices:   ← (press Enter)
```

The solver skips the cutting-plane loop and returns the LP optimum directly.

---

### Example 4: Free Variables

```
minimize:    x1 + 2*x2
subject to:
             x1 + x2 = 3
             2*x1 - x2 >= 1
             x1 >= 0,  x2 free
```

**Input:**

```
Number of decision variables: 2
Enter 2 objective coefficients: 1 2
Maximization problem? (yes/no): no
Number of constraints: 2
Constraint 1 (coefficients + RHS): 1 1 3
Type (<= / >= / =): =
Constraint 2 (coefficients + RHS): 2 -1 1
Type (<= / >= / =): >=
  x1: 1
  x2: 3
Integer variable indices:   ← (press Enter for pure LP)
```

**Output:**

```
Error: UNBOUNDED problem.
```

Because `x2 = 3 − x1`, the objective equals `6 − x1`, and `x1` can grow without limit.

---

## Input Format Reference

| Prompt | Description |
|--------|-------------|
| Number of decision variables | Total count of `x` variables |
| Objective coefficients | Space-separated, e.g. `5 4 3` |
| Maximization? | `yes` / `no` |
| Constraint `(coefficients + RHS)` | All coefficients then RHS, e.g. `1 2 6` for `x1 + 2*x2 ≤ 6` |
| Constraint type | `<=`, `>=`, or `=` (asked again if invalid) |
| Variable bound type | `1` (≥0), `2` (≤0), `3` (free) |
| Integer variable indices | 1-based, space-separated; Enter to skip |

---

## Using It from Python

```python
from milp_gomory import solve_milp

# Example 2: x1 integer, x2 continuous (int_vars are 0-based here)
x, obj, status = solve_milp([5, 4], [([1, 1], "<=", 5), ([10, 6], "<=", 45)],
                            int_vars=[0])
# x = [4., 0.8333], obj = 23.333, status = "optimal"
```

`status` is `"optimal"`, `"feasible"` (node limit reached; best solution found, not proven
optimal) or `"limit"` (node limit reached before any integer solution; `x` is the LP bound).
Infeasible and unbounded problems raise `ValueError`.

---

## Code Structure

### Classes

| Class | Description |
|-------|-------------|
| `TwoPhaseSimplex` | Core LP solver using two-phase simplex |
| `GomoryMILP` | Extends `TwoPhaseSimplex`; adds Gomory cuts, branch-and-bound and integrality |

### Key Methods

| Method | Class | Description |
|--------|-------|-------------|
| `read_problem()` | `TwoPhaseSimplex` | Read objective, constraints, variable types |
| `load_problem()` | both | Set the problem from Python (no prompts) |
| `build_tableau()` | `TwoPhaseSimplex` | Build Phase I tableau with slacks and artificials |
| `run_simplex()` | `TwoPhaseSimplex` | Primal simplex (Dantzig's rule, Bland's Rule fallback) |
| `run_dual_simplex()` | `TwoPhaseSimplex` | Dual simplex to restore feasibility after cuts/branching (Bland's Rule fallback) |
| `end_phase1()` | `TwoPhaseSimplex` | Feasibility check; pivot out / remove artificials |
| `build_phase2_objective()` | `TwoPhaseSimplex` | Set up Phase II objective row |
| `read_integer_vars()` | `GomoryMILP` | Read which variables must be integer |
| `_fractional_rows()` | `GomoryMILP` | Rows of fractional integer basic variables, most fractional first |
| `_gomory_cut()` | `GomoryMILP` | Build a GMI cut from one tableau row |
| `_add_gomory_cuts()` | `GomoryMILP` | Insert cuts into the tableau |
| `_drop_inactive_cuts()` | `GomoryMILP` | Remove cuts that are no longer binding |
| `_branch_and_bound()` | `GomoryMILP` | Finish the search when cuts stall |
| `optimize()` | `GomoryMILP` | Solve the loaded problem, return `(x, objective)` |
| `solve()` | `GomoryMILP` | Full interactive read → solve → print |

### Helper Functions

```python
safe_int(prompt)              # Validated integer input
safe_float_list(prompt, n)    # Validated float list input
safe_choice(prompt, choices)  # Validated choice input (constraint type)
frac(x)                       # Fractional part: x - floor(x)
solve_milp(c, constraints, var_types=None, maximize=True, int_vars=())
```

---

## Algorithm Parameters

```python
EPS            = 1e-9    # Pivot / reduced-cost tolerance
FEAS_TOL       = 1e-7    # Feasibility tolerance (relative to the largest |RHS|)
FRAC_TOL       = 1e-6    # Tolerance to consider a value "integer"
MAX_CUTS       = 200     # Gomory cuts before switching to branch-and-bound
CUTS_PER_ROUND = 10      # Cuts added per round before re-optimizing
AWAY           = 0.01    # Only cut rows whose fractional part is in [AWAY, 1 - AWAY]
MAX_DYNAMISM   = 1e6     # Skip cuts with largest/smallest coefficient above this
STALL_ROUNDS   = 5       # Rounds without progress before branching takes over
MAX_NODES      = 20_000  # Branch-and-bound node limit
PLUNGE_DEPTH   = 25      # Depth-first steps before returning to best-bound order
DEGENERATE_LIMIT = 50    # Degenerate pivots before the simplex methods switch to Bland's Rule
```

---

## Performance

Average time and number of correct answers on random bounded pure-integer problems
(20 problems per size, checked against an independent MILP solver; output discarded):

| Size (variables × constraints) | Previous: time | Previous: correct | Current: time | Current: correct |
|---|---|---|---|---|
| 4 × 3   | 0.019 s | 18/20 | 0.007 s | 20/20 |
| 6 × 4   | 0.052 s | 15/20 | 0.009 s | 20/20 |
| 8 × 5   | 0.095 s | 12/20 | 0.009 s | 20/20 |
| 10 × 6  | 0.158 s | 10/20 | 0.024 s | 20/20 |
| 15 × 8  | 0.293 s |  3/20 | 0.039 s | 20/20 |

The previous version's misses were runs that hit the 200-cut limit (and then printed
a non-integral point as "OPTIMAL SOLUTION") or returned a wrong value.

---

## Error Messages

| Message | Cause |
|---------|-------|
| `UNBOUNDED problem.` | Objective can improve without bound (for MILPs: the LP relaxation is unbounded and an integer solution exists) |
| `Problem is INFEASIBLE.` | No feasible solution exists after Phase I |
| `Problem is INFEASIBLE after adding cut.` | Cuts proved there is no integer solution |
| `Problem is INFEASIBLE (no integer solution).` | Branch-and-bound (or a divisibility check on an equality) proved there is no integer solution |
| `UNBOUNDED LP relaxation; integer feasibility undecided ...` | The LP relaxation is unbounded and the node limit was reached before an integer solution was found or ruled out |

---

## Technical Notes

### Negative RHS Normalization
Constraints with negative RHS are automatically flipped before building the tableau:
```
x1 - x2 <= -3   →   -x1 + x2 >= 3
```

### Free Variable Splitting
Free variables (`type 3`) are split into two non-negative variables:
```
x  →  x⁺ - x⁻,   x⁺ ≥ 0,  x⁻ ≥ 0
```

### Degenerate Artificials (Phase I → Phase II)
If an artificial variable stays in the basis at value zero after Phase I, it is pivoted
out on any non-zero entry of its row; if the row has none, the constraint is redundant and
the row is removed. (Earlier versions only dropped the artificial, which left the row
without a basic variable and could produce infeasible "optimal" solutions.)

### Dual Simplex Pre-condition
After a primal-optimal LP solution, all reduced costs are non-positive, so the tableau is dual feasible. Adding a Gomory cut or a branching bound (with negative RHS) only violates primal feasibility — exactly the condition the dual simplex is designed to fix.

### Windows Console
All printed text is plain ASCII, so the program also works when its output is redirected
to a file or pipe on Windows.

---

## Verbose Output

To print every simplex pivot (Phase I, Phase II, dual simplex):

```bash
python milp_gomory.py --steps
```

---

## Running the Tests

```bash
pip install pytest scipy
python -m pytest tests
```

SciPy is only used by the tests, as an independent reference solver.

---

## Requirements

```
python  >= 3.9
numpy   >= 1.20
```

---

## References

- **Gomory, R.E.** (1958). *Outline of an algorithm for integer solutions to linear programs.* Bulletin of the American Mathematical Society.
- **Gomory, R.E.** (1960). *An algorithm for the mixed integer problem.* RAND Report RM-2597.
- **Two-Phase Simplex Method:** Classical LP textbook algorithm.
- **Dual Simplex Method:** Lemke (1954).
- **Bland's Rule:** Prevents cycling; selects smallest-index entering variable.

---

## License

Free to use and modify.

---

## Author

👨‍💻 Payamkou
