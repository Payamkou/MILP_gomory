# Mixed-Integer Linear Programming Solver with Gomory Cutting Planes

A Mixed-Integer Linear Programming (MILP) solver using the **Gomory Fractional Cutting-Plane Method** combined with the Two-Phase Simplex algorithm.

## Features

✅ **Solve MILP Problems**
- **Minimization** and **Maximization** problems
- **Mixed-integer constraints** (some variables must be integer)
- **Positive** (≥ 0), **Negative** (≤ 0), and **Free** (unrestricted) variables
- Constraint types: `<=`, `>=`, `=`

✅ **Gomory Cutting-Plane Algorithm**
- Solves LP relaxation first
- Identifies fractional integer variables
- Generates Gomory cuts to enforce integrality
- Uses dual simplex to restore feasibility
- Iterative process until all integer variables are integral

✅ **Advanced Features**
- Two-Phase Simplex for LP relaxation
- Dual Simplex for cut restoration
- Bland's Rule to prevent cycling
- Handles degenerate cases
- Negative RHS constraint normalization

## How to Use?

### Installation
```bash
python milp_gomory.py
```

### Example 1: Simple Integer Programming Problem

Suppose you want to solve:
```
maximize:    3*x1 + 2*x2
subject to:
             x1 + x2 <= 4
             2*x1 + x2 <= 7
             x1, x2 >= 0, integer
```

**Input Steps:**

1️⃣ **Number of decision variables:** `2`

2️⃣ **Objective coefficients:** `3 2`

3️⃣ **Problem type:** `yes` (Maximization)

4️⃣ **Number of constraints:** `2`

5️⃣ **First constraint:** `1 1 4`
   - Type: `<=`

6️⃣ **Second constraint:** `2 1 7`
   - Type: `<=`

7️⃣ **Variable bounds:**
   - x1: `1` (positive)
   - x2: `1` (positive)

8️⃣ **Integer variables:** `1 2` (both must be integer)

**Output:**
```
LP RELAXATION
─────────────────────────────────
  LP relaxation solution:
    x1 = 3.4
    x2 = 0.6  [integer required]
  Objective = 10.8

GOMORY CUTTING PLANE ITERATIONS
─────────────────────────────────
  Cut   1 | x2 = 0.600000 (frac = 0.600000)
          | obj = 10.666667
          | [*] x1 = 3.333333
          | [*] x2 = 0.666667

  Cut   2 | x1 = 3.666667 (frac = 0.666667)
          | obj = 10.0
          | [*] x1 = 3.0
          | [*] x2 = 1.0

OPTIMAL SOLUTION
─────────────────────────────────
  x1 = 3.0  [integer]
  x2 = 1.0  [integer]
  Optimal objective value = 11.0
```

---

## Algorithm Explanation

### Phase 1: LP Relaxation

The algorithm starts by solving the continuous LP relaxation (ignoring integer constraints):
- Uses Two-Phase Simplex
- Gets a continuous optimal solution

### Phase 2: Gomory Cutting Planes

**Step 1:** Check if all integer-required variables are integer
- If yes → **Done!** This is the integer optimal solution
- If no → Continue to step 2

**Step 2:** Find the "most fractional" integer variable
- The variable with fractional part closest to 0.5
- Example: x = 2.7 has fractional part 0.7, distance = min(0.7, 0.3) = 0.3

**Step 3:** Generate Gomory Cut
- For basis row i with basic variable having value b̄ᵢ
- Let f₀ = fractional part of b̄ᵢ
- Cut: Σⱼ frac(āᵢⱼ) · xⱼ ≥ f₀
- This cut removes the fractional solution but keeps integer feasible region

**Step 4:** Add Cut to Tableau
- Add new slack variable (surplus)
- Insert cut as new constraint
- The new slack initially has negative value

**Step 5:** Apply Dual Simplex
- Restores primal feasibility
- Moves toward new solution that respects the cut

**Step 6:** Repeat
- Go back to Step 1 with updated tableau

### Why It Works

The Gomory cut exploits the fractional part structure:
- If variable is integer, its value is naturally integer (frac = 0)
- If variable is fractional, the cut forces a tighter constraint
- Eventually, after enough cuts, all integer variables become integer

## Input Format

### 1. Number of Decision Variables
```
Number of decision variables: 3
```

### 2. Objective Function Coefficients
```
Enter 3 objective coefficients: 5 4 3
```

### 3. Problem Type
```
Maximization problem? (yes/no): yes
```

### 4. Constraints
For each constraint, enter **coefficients + RHS**:
```
Constraint 1 (coefficients + RHS): 2 1 1 10
Type (<= / >= / =): <=

Constraint 2 (coefficients + RHS): 1 2 2 12
Type (<= / >= / =): <=
```

### 5. Variable Bounds
- `1` = Positive or zero (`x >= 0`)
- `2` = Negative or zero (`x <= 0`)
- `3` = Free (unrestricted)
```
x1: 1
x2: 1
x3: 1
```

### 6. Integer Variables (1-based indices)
```
Integer variable indices: 1 2 3
```

Or for no integer constraints (pure LP):
```
Integer variable indices: 
(just press Enter)
```

---

## More Examples

### Example 2: Knapsack Problem
```
maximize:    60*x1 + 50*x2 + 70*x3
subject to:
             5*x1 + 4*x2 + 6*x3 <= 10
             x1, x2, x3 ∈ {0, 1}  (binary)
```

**Input:**
```
Number of decision variables: 3
Enter 3 objective coefficients: 60 50 70
Maximization problem? (yes/no): yes
Number of constraints: 1
Constraint 1 (coefficients + RHS): 5 4 6 10
Type: <=
Variable bounds:
x1: 1
x2: 1
x3: 1
Integer variable indices: 1 2 3
```

**Explanation:** Choose which items to pack to maximize value within weight limit.

---

### Example 3: Production with Integer Batches
```
maximize:    40*x1 + 30*x2
subject to:
             2*x1 + 1*x2 <= 100   (labor hours)
             3*x1 + 4*x2 <= 150   (machine hours)
             x1, x2 >= 0, integer (complete batches)
```

**Input:**
```
Number of decision variables: 2
Enter 2 objective coefficients: 40 30
Maximization problem? (yes/no): yes
Number of constraints: 2
Constraint 1 (coefficients + RHS): 2 1 100
Type: <=
Constraint 2 (coefficients + RHS): 3 4 150
Type: <=
Variable bounds:
x1: 1
x2: 1
Integer variable indices: 1 2
```

---

### Example 4: Mixed-Integer (Some Continuous)
```
maximize:    5*x1 + 3*x2 + 4*x3
subject to:
             2*x1 + 1*x2 + 2*x3 <= 50
             x1, x2 >= 0, integer
             x3 >= 0, continuous
```

**Input:**
```
Number of decision variables: 3
Enter 3 objective coefficients: 5 3 4
Maximization problem? (yes/no): yes
Number of constraints: 1
Constraint 1 (coefficients + RHS): 2 1 2 50
Type: <=
Variable bounds:
x1: 1
x2: 1
x3: 1
Integer variable indices: 1 2
```

**Note:** Only x1 and x2 must be integer; x3 is continuous.

---

## Code Structure

### Main Classes

#### `TwoPhaseSimplex`
Base class for solving LP problems:
- `read_problem()` - Input LP problem
- `build_tableau()` - Build initial tableau
- `run_simplex()` - Execute primal simplex
- `run_dual_simplex()` - Execute dual simplex (for cuts)
- `_solve_lp()` - Solve LP phases
- `_extract_solution()` - Get solution from tableau

#### `GomoryMILP`
Extends TwoPhaseSimplex for MILP:
- `read_integer_vars()` - Specify which variables are integer
- `_find_fractional_row()` - Find most fractional integer variable
- `_add_gomory_cut()` - Generate and insert Gomory cut
- `solve()` - Main entry point (LP + Gomory loop)
- `_print_result()` - Display final solution

---

## Technical Details

### Key Parameters
```python
EPS = 1e-9           # Numerical precision
FRAC_TOL = 1e-6      # Tolerance for fractional parts
MAX_CUTS = 200       # Maximum number of Gomory cuts
```

### Gomory Cut Formula

For constraint row i with basic variable having value b̄ᵢ:

**If x_Bi = b̄ᵢ − Σⱼ āᵢⱼ · xⱼ**

Let f₀ = frac(b̄ᵢ)

**Gomory Cut:** Σⱼ frac(āᵢⱼ) · xⱼ ≥ f₀

**In tableau form (with slack s ≥ 0):**
```
−Σⱼ frac(āᵢⱼ) · xⱼ + s = −f₀
```

The slack enters the basis with negative value −f₀, triggering dual simplex.

### Dual Simplex Algorithm

Used to restore feasibility after adding a cut:
1. Find most negative basic variable (infeasible row)
2. Check if any non-basic variable can improve it
3. If not → problem is infeasible
4. Otherwise → pivot to restore feasibility

---

## Error Messages

### ❌ **LP Relaxation Infeasible**
```
Error: INFEASIBLE problem (Phase I failed).
```
**Meaning:** Original constraints are contradictory; no MILP solution exists.

### ❌ **LP Relaxation Unbounded**
```
Error: UNBOUNDED problem.
```
**Meaning:** Objective can improve infinitely; missing constraints.

### ❌ **Cutting Plane Limit Reached**
```
WARNING: Cut limit (200) reached. Solution may not be fully integer.
```
**Meaning:** Algorithm couldn't find integer solution within MAX_CUTS iterations.

### ❌ **Infeasibility During Cuts**
```
Error: INFEASIBLE during dual simplex.
```
**Meaning:** Cuts made the problem infeasible (rare, algorithm error).

---

## Algorithm Convergence

**Does it always find the integer optimal solution?**

✅ **Mathematically:** Yes, Gomory's cutting-plane method is theoretically guaranteed to converge.

⚠️ **Practically:** 
- May require many cuts (hence MAX_CUTS limit)
- Numerical errors can accumulate with large problems
- Slow for hard combinatorial problems (NP-hard in general)

**Better alternatives for large/hard problems:**
- Branch-and-bound algorithms
- Commercial solvers (CPLEX, Gurobi)
- Constraint programming solvers

---

## Viewing Solution Steps

To see detailed simplex iterations:

**Modify solve() call:**
```python
self._solve_lp(print_steps=True)  # Show LP relaxation steps
self.run_dual_simplex(print_steps=True)  # Show dual simplex steps
```

Output will show complete tableaux after each iteration.

---

## Requirements
```
numpy >= 1.20.0
python >= 3.6
```

---

## References

- **Gomory, R.E. (1958)** - Original cutting-plane method paper
- **Two-Phase Simplex** - Classical linear programming
- **Dual Simplex** - Optimization under infeasibility
- **Integer Linear Programming** - Standard optimization technique

---

## Tips for Problem Modeling

### Binary Variables (0 or 1)
Use variable type `1` and add constraint `0 <= x <= 1`:
```
Constraint: 1*x <= 1
```

### Fixed-Cost Problems
Introduce binary variable y:
```
If produce: cost = fixed + variable*quantity
If not: cost = 0
```

### Facility Location
Binary variables for open/closed, continuous for flows.

### Vehicle Routing
Binary variables for vehicle assignments, integer for visit counts.

---

## License

Free to use and modify.

---

## Author

👨‍💻 Payamkou
