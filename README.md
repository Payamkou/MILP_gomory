# Mixed-Integer Linear Programming Solver (Gomory Cutting Planes)

A Mixed-Integer Linear Programming (MILP) solver built on top of the **Two-Phase Simplex** algorithm, using **Gomory's Fractional Cutting Plane** method to enforce integrality constraints.

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
- **Bland's Rule** to prevent cycling

✅ **Gomory Cutting Plane**
- Automatically generates Gomory fractional cuts
- **Dual Simplex** to restore feasibility after each cut
- Most-fractional variable selection strategy
- Configurable cut limit (`MAX_CUTS = 200`)

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
4. Find the most fractional integer variable in the current basis
5. Generate a Gomory cut from that tableau row
6. Append cut to the tableau; run Dual Simplex to restore feasibility
7. Go to step 3
```

### Gomory Cut Formula

For a canonical tableau row `i` with basic variable `x_Bi`:

```
x_Bi = b̄_i − Σ_j ā_ij · x_j
```

Let `f₀ = frac(b̄_i)`. The cut is:

```
Σ_j  frac(ā_ij) · x_j  ≥  f₀
```

Which is inserted into the tableau as:

```
−Σ_j frac(ā_ij) · x_j  +  s  =  −f₀     (s ≥ 0, new slack)
```

The new slack starts at `−f₀ < 0` and the Dual Simplex restores feasibility.

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
Variable bounds:  1 = x≥0   2 = x≤0   3 = free
  x1: 1
  x2: 1

Integrality:
  Enter 1-based indices of integer variables (space-separated).
  Integer variable indices: 1 2
```

**Output:**

```
===========================================================
  LP RELAXATION
===========================================================
  LP relaxation solution:
    x1 = 3.500000  [integer required]
    x2 = 0.000000  [integer required]
  Objective = 3.500000

===========================================================
  GOMORY CUTTING PLANE ITERATIONS
===========================================================

  Cut   1 | x1 = 3.500000 (frac = 0.500000)
          | obj = 3.000000
          | [*] x1 = 3.000000
          | [*] x2 = 0.000000

===========================================================
  OPTIMAL SOLUTION
===========================================================
  x1 = 3.000000  [integer]
  x2 = 0.000000  [integer]

  Optimal objective value = 3.000000
===========================================================
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

---

## Input Format Reference

| Prompt | Description |
|--------|-------------|
| Number of decision variables | Total count of `x` variables |
| Objective coefficients | Space-separated, e.g. `5 4 3` |
| Maximization? | `yes` / `no` |
| Constraint `(coefficients + RHS)` | All coefficients then RHS, e.g. `1 2 6` for `x1 + 2*x2 ≤ 6` |
| Constraint type | `<=`, `>=`, or `=` |
| Variable bound type | `1` (≥0), `2` (≤0), `3` (free) |
| Integer variable indices | 1-based, space-separated; Enter to skip |

---

## Code Structure

### Classes

| Class | Description |
|-------|-------------|
| `TwoPhaseSimplex` | Core LP solver using two-phase simplex |
| `GomoryMILP` | Extends `TwoPhaseSimplex`; adds Gomory cuts and integrality |

### Key Methods

| Method | Class | Description |
|--------|-------|-------------|
| `read_problem()` | `TwoPhaseSimplex` | Read objective, constraints, variable types |
| `build_tableau()` | `TwoPhaseSimplex` | Build Phase I tableau with slacks and artificials |
| `run_simplex()` | `TwoPhaseSimplex` | Primal simplex with Bland's Rule |
| `run_dual_simplex()` | `TwoPhaseSimplex` | Dual simplex to restore feasibility after cuts |
| `build_phase2_objective()` | `TwoPhaseSimplex` | Set up Phase II objective row |
| `read_integer_vars()` | `GomoryMILP` | Read which variables must be integer |
| `_find_fractional_row()` | `GomoryMILP` | Pick the most-fractional integer basic variable |
| `_add_gomory_cut()` | `GomoryMILP` | Generate and insert Gomory cut into tableau |
| `solve()` | `GomoryMILP` | Full MILP solve loop |

### Helper Functions

```python
safe_int(prompt)              # Validated integer input
safe_float_list(prompt, n)    # Validated float list input
frac(x)                       # Fractional part: x - floor(x)
```

---

## Algorithm Parameters

```python
EPS      = 1e-9    # Numerical precision for pivot comparisons
FRAC_TOL = 1e-6    # Tolerance to consider a value "integer"
MAX_CUTS = 200     # Maximum Gomory cuts before stopping
```

---

## Error Messages

| Message | Cause |
|---------|-------|
| `UNBOUNDED problem.` | Objective can improve without bound |
| `Problem is INFEASIBLE.` | No feasible solution exists after Phase I |
| `Problem is INFEASIBLE after adding cut.` | Dual simplex finds no entering column |
| `WARNING: Cut limit (200) reached.` | Too many cuts without convergence |

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
If an artificial variable stays in the basis at value zero after Phase I, it is marked as `None` in the basis and skipped safely.

### Dual Simplex Pre-condition
After a primal-optimal LP solution, all reduced costs are non-positive, so the tableau is dual feasible. Adding a Gomory cut (with negative RHS) only violates primal feasibility — exactly the condition the dual simplex is designed to fix.

---

## Verbose Output

To print each simplex iteration tableau:

```python
self._solve_lp(print_steps=True)     # LP relaxation steps
self.run_dual_simplex(print_steps=True)  # Dual simplex steps after each cut
```

---

## Requirements

```
python  >= 3.6
numpy   >= 1.0
```

---

## References

- **Gomory, R.E.** (1958). *Outline of an algorithm for integer solutions to linear programs.* Bulletin of the American Mathematical Society.
- **Two-Phase Simplex Method:** Classical LP textbook algorithm.
- **Dual Simplex Method:** Lemke (1954).
- **Bland's Rule:** Prevents cycling; selects smallest-index entering variable.

---

## License

Free to use and modify.

---

## Author

👨‍💻 Payamkou
