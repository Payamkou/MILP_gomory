import numpy as np
import math

EPS = 1e-9
FRAC_TOL = 1e-6   
MAX_CUTS = 200 


# ─────────────────────────────────────────────
# Input helpers
# ─────────────────────────────────────────────

def safe_int(prompt):
    while True:
        try:
            return int(input(prompt))
        except ValueError:
            print("Invalid integer. Try again.")


def safe_float_list(prompt, expected_len):
    while True:
        try:
            vals = list(map(float, input(prompt).split()))
            if len(vals) != expected_len:
                print(f"Expected {expected_len} numbers.")
                continue
            return vals
        except ValueError:
            print("Enter numeric values separated by spaces.")


def frac(x):
    """Fractional part of x, always in [0, 1)."""
    return x - math.floor(x)


# ─────────────────────────────────────────────
# Two-Phase Simplex  (LP relaxation engine)
# ─────────────────────────────────────────────

class TwoPhaseSimplex:
    """
    Solves a general LP using the two-phase simplex method.

    Tableau convention
    ──────────────────
    • Rows 0 … m-1   : constraint rows (canonical form).
    • Row  -1 (last)  : objective row.  Holds  z_row = -(c_B · B⁻¹ · A - c).
      A column j should enter the basis when  z_row[j] > 0.
      At optimality:  -z_row[-1]  is the objective value (multiply by -1 for max).

    Variable types
    ──────────────
      1 → x ≥ 0           (one column in tableau)
      2 → x ≤ 0           (substituted as x = -x', x' ≥ 0)
      3 → free             (split: x = x⁺ − x⁻, two columns)
    """

    def __init__(self):
        self.n = 0            # number of original decision variables
        self.c = None         # objective coefficients (internally always maximise)
        self.is_max = True
        self.constraints = [] # list of (np.array coeffs, sense, float rhs)
        self.var_types = []   # one entry per decision variable: 1, 2, or 3
        self.tableau = None
        self.basis = []       # basis[i] = tableau column index of basic var in row i
        self.artificial_cols = []
        self.col_map = []     # col_map[j] = list of tableau column indices for x_j

    # ── Input ────────────────────────────────────────────────────────────

    def read_problem(self):
        self.n = safe_int("Number of decision variables: ")
        self.c = np.array(
            safe_float_list(f"Enter {self.n} objective coefficients: ", self.n)
        )
        ans = input("Maximization problem? (yes/no): ").lower()
        self.is_max = ans in ["yes", "y"]
        if not self.is_max:
            self.c = -self.c   # internally always maximise

        m = safe_int("Number of constraints: ")
        print("Constraint types: <=, >=, =")
        for i in range(m):
            row = safe_float_list(
                f"Constraint {i+1} (coefficients + RHS): ", self.n + 1
            )
            sense = input("Type (<= / >= / =): ").strip()
            if sense not in ["<=", ">=", "="]:
                raise ValueError(f"Unknown constraint type '{sense}'.")
            self.constraints.append((np.array(row[:-1]), sense, row[-1]))

        print("Variable bounds:  1 = x≥0   2 = x≤0   3 = free")
        for i in range(self.n):
            while True:
                t = safe_int(f"  x{i+1}: ")
                if t in [1, 2, 3]:
                    self.var_types.append(t)
                    break
                print("  Enter 1, 2, or 3.")

    # ── Build initial tableau ─────────────────────────────────────────────

    def build_tableau(self):
        self.basis = []
        self.artificial_cols = []
        self.col_map = []

        # Assign tableau columns to decision variables
        col = 0
        for t in self.var_types:
            if t in [1, 2]:
                self.col_map.append([col]); col += 1
            else:                               # free: x = x⁺ − x⁻
                self.col_map.append([col, col + 1]); col += 2
        base_vars = col
        m = len(self.constraints)

        # Normalise: negative RHS → flip row & sense so RHS becomes positive.
        # Must happen BEFORE counting columns (a <= with neg RHS becomes >=,
        # requiring an artificial that wasn't budgeted otherwise).
        normalized = []
        for coeffs, sense, rhs in self.constraints:
            if rhs < -EPS:
                coeffs = -coeffs
                rhs    = -rhs
                if   sense == "<=": sense = ">="
                elif sense == ">=": sense = "<="
                # "=" stays "="
            normalized.append((coeffs, sense, rhs))

        slack_count = 0
        artificial_count = 0
        for _, s, _ in normalized:
            if s == "<=":
                slack_count += 1
            elif s == ">=":
                slack_count += 1
                artificial_count += 1
            else:
                artificial_count += 1

        total_cols = base_vars + slack_count + artificial_count
        A = np.zeros((m, total_cols))
        b = np.zeros(m)
        slack_ptr = base_vars
        art_ptr   = base_vars + slack_count

        for i, (coeffs, sense, rhs) in enumerate(normalized):
            for j in range(self.n):
                cs = self.col_map[j]
                if   self.var_types[j] == 1: A[i, cs[0]] =  coeffs[j]
                elif self.var_types[j] == 2: A[i, cs[0]] = -coeffs[j]
                else:
                    A[i, cs[0]] =  coeffs[j]
                    A[i, cs[1]] = -coeffs[j]

            if sense == "<=":
                A[i, slack_ptr] = 1
                self.basis.append(slack_ptr)
                slack_ptr += 1
            elif sense == ">=":
                A[i, slack_ptr] = -1;  slack_ptr += 1
                A[i, art_ptr]   =  1
                self.basis.append(art_ptr)
                self.artificial_cols.append(art_ptr)
                art_ptr += 1
            else:   # equality
                A[i, art_ptr] = 1
                self.basis.append(art_ptr)
                self.artificial_cols.append(art_ptr)
                art_ptr += 1

            b[i] = rhs

        self.tableau = np.hstack([A, b.reshape(-1, 1)])

        # Phase I objective: minimise Σ artificials  ≡  maximise −Σ artificials
        obj = np.zeros(total_cols)
        for c in self.artificial_cols:
            obj[c] = 1
        phase1_row = np.zeros(total_cols + 1)
        phase1_row[:-1] = -obj
        self.tableau = np.vstack([self.tableau, phase1_row])

        # Canonicalise Phase I row (zero out costs of current basic artificials)
        for i, c in enumerate(self.basis):
            if c in self.artificial_cols:
                self.tableau[-1] += self.tableau[i]

    # ── Pivot ─────────────────────────────────────────────────────────────

    def pivot(self, r, c):
        self.tableau[r] /= self.tableau[r, c]
        for i in range(self.tableau.shape[0]):
            if i != r:
                self.tableau[i] -= self.tableau[i, c] * self.tableau[r]

    # ── Primal Simplex ────────────────────────────────────────────────────

    def run_simplex(self, print_steps=False):
        m = len(self.basis)
        iteration = 0
        while True:
            iteration += 1
            costs = self.tableau[-1, :-1]
            # Bland's rule: smallest-index positive reduced cost
            entering = next((j for j in range(len(costs)) if costs[j] > EPS), -1)
            if entering == -1:
                return   # optimal

            ratios = [
                self.tableau[i, -1] / self.tableau[i, entering]
                if self.tableau[i, entering] > EPS else np.inf
                for i in range(m)
            ]
            if all(r == np.inf for r in ratios):
                raise ValueError("UNBOUNDED problem.")

            leaving = int(np.argmin(ratios))
            self.pivot(leaving, entering)
            self.basis[leaving] = entering

            if print_steps:
                print(f"    Primal iter {iteration}")

    # ── Dual Simplex  (used after adding Gomory cuts) ─────────────────────

    def run_dual_simplex(self, print_steps=False):
        """
        Restores primal feasibility after a cut introduces a negative RHS.

        Pre-condition: the current basis is dual feasible, i.e. the objective
        row has no positive entries (all reduced costs ≤ 0 for maximisation).
        This is guaranteed immediately after a primal-optimal LP solution.
        """
        m = len(self.basis)
        iteration = 0
        while True:
            iteration += 1
            rhs = self.tableau[:m, -1]

            # Leaving row: most-negative RHS
            leaving = int(np.argmin(rhs))
            if rhs[leaving] >= -FRAC_TOL:
                return   # all RHS ≥ 0 → primal feasible

            row = self.tableau[leaving, :-1]

            # Entering column: minimum |obj_row[j] / row[j]| for row[j] < 0
            ratios = [
                abs(self.tableau[-1, j] / row[j])
                if row[j] < -EPS else np.inf
                for j in range(len(row))
            ]
            if all(r == np.inf for r in ratios):
                raise ValueError("Problem is INFEASIBLE after adding cut.")

            entering = int(np.argmin(ratios))
            self.pivot(leaving, entering)
            self.basis[leaving] = entering

            if print_steps:
                print(f"    Dual iter {iteration}")

    # ── Phase II objective row ────────────────────────────────────────────

    def build_phase2_objective(self):
        cols = self.tableau.shape[1] - 1
        obj  = np.zeros(cols)
        for j in range(self.n):
            mp = self.col_map[j]
            if   self.var_types[j] == 1: obj[mp[0]] =  self.c[j]
            elif self.var_types[j] == 2: obj[mp[0]] = -self.c[j]
            else:
                obj[mp[0]] =  self.c[j]
                obj[mp[1]] = -self.c[j]

        new_row = np.zeros(cols + 1)
        new_row[:-1] = obj
        self.tableau[-1] = new_row

        # Canonicalise: zero out reduced costs of current basic variables
        for i, col in enumerate(self.basis):
            if col is None:   # degenerate: artificial stuck at 0
                continue
            coeff = self.tableau[-1, col]
            if abs(coeff) > EPS:
                self.tableau[-1] -= coeff * self.tableau[i]

    # ── Solve LP relaxation ───────────────────────────────────────────────

    def _solve_lp(self, print_steps=False):
        self.build_tableau()

        if print_steps: print("  [Phase I]")
        self.run_simplex(print_steps=print_steps)

        if abs(self.tableau[-1, -1]) > EPS:
            raise ValueError("Problem is INFEASIBLE.")

        # Remove artificial columns
        keep = [j for j in range(self.tableau.shape[1] - 1)
                if j not in self.artificial_cols]
        self.tableau = self.tableau[:, keep + [-1]]
        self.basis = [
            keep.index(b) if b in keep else None
            for b in self.basis
        ]

        if print_steps: print("  [Phase II]")
        self.build_phase2_objective()
        self.run_simplex(print_steps=print_steps)

    # ── Extract current solution from tableau ────────────────────────────

    def _extract_solution(self):
        solution = np.zeros(self.n)
        for i, col in enumerate(self.basis):
            if col is None:
                continue
            for j in range(self.n):
                if col in self.col_map[j]:
                    idx = self.col_map[j].index(col)
                    if self.var_types[j] == 3:
                        if idx == 0: solution[j] += self.tableau[i, -1]
                        else:        solution[j] -= self.tableau[i, -1]
                    elif self.var_types[j] == 1:
                        solution[j] =  self.tableau[i, -1]
                    else:
                        solution[j] = -self.tableau[i, -1]

        obj_val = self.tableau[-1, -1]
        if self.is_max:
            obj_val = -obj_val
        return solution, obj_val


# ─────────────────────────────────────────────
# Mixed-Integer LP  via Gomory Cutting Planes
# ─────────────────────────────────────────────

class GomoryMILP(TwoPhaseSimplex):
    """
    Extends TwoPhaseSimplex to handle Mixed-Integer LP using
    the Gomory fractional cutting-plane method.

    Algorithm
    ─────────
    1. Solve the LP relaxation (ignore integrality).
    2. If every integer-required variable in the current BFS already
       has an integer value → optimal ILP solution found.
    3. Otherwise, pick the basis row whose integer variable has the
       largest fractional part (most fractional first).
    4. Derive the Gomory cut from that row and append it to the
       tableau (as a new ≤ constraint with a surplus slack).
    5. Apply the dual simplex to restore primal feasibility.
    6. Go to 2.

    Gomory cut formula
    ──────────────────
    For canonical row i:  x_Bi = b̄_i − Σ_j ā_ij · x_j
    Let f₀ = frac(b̄_i).
    Cut:  Σ_j  frac(ā_ij) · x_j  ≥  f₀
    As a tableau row (with new slack s ≥ 0):
        −Σ_j frac(ā_ij) · x_j  +  s  =  −f₀     (RHS < 0)
    The new slack enters the basis with value −f₀ < 0,
    which is immediately repaired by the dual simplex.
    """

    def __init__(self):
        super().__init__()
        self.int_vars = []   # 0-based indices of integer-constrained variables

    # ── Read which variables must be integer ─────────────────────────────

    def read_integer_vars(self):
        print("\nIntegrality:")
        print("  Enter 1-based indices of integer variables (space-separated).")
        print("  Press Enter with no input for a pure LP (no integer constraints).")
        raw = input("  Integer variable indices: ").strip()
        if not raw:
            self.int_vars = []
            return
        try:
            indices = list(map(int, raw.split()))
            for idx in indices:
                if idx < 1 or idx > self.n:
                    raise ValueError(f"Index {idx} is out of range [1, {self.n}].")
            self.int_vars = sorted(set(i - 1 for i in indices))
        except ValueError as e:
            raise ValueError(f"Invalid integer variable specification: {e}")

    # ── Find most-fractional integer-required basic variable ─────────────

    def _find_fractional_row(self):
        """
        Returns the index of the constraint row whose basic variable:
          (a) maps to an integer-required original variable, AND
          (b) has the largest deviation from an integer value.
        Returns -1 if all integer variables are already integral.
        """
        best_row  = -1
        best_dist = FRAC_TOL   # must exceed tolerance

        for i, col in enumerate(self.basis):
            if col is None:
                continue
            for j in self.int_vars:
                if col in self.col_map[j]:
                    val  = self.tableau[i, -1]
                    f    = frac(val)
                    dist = min(f, 1.0 - f)   # distance to nearest integer
                    if dist > best_dist:
                        best_dist = dist
                        best_row  = i
        return best_row

    # ── Generate and insert a Gomory cut ─────────────────────────────────

    def _add_gomory_cut(self, row_idx):
        """
        Builds the Gomory cut from tableau row `row_idx` and inserts it
        into the tableau as a new constraint row (just before the
        objective row so that the objective row stays last).

        A new slack variable is created for the cut. It enters the basis
        at a negative value (−f₀), so dual simplex must be called after.
        """
        n_cols = self.tableau.shape[1] - 1   # current column count (excl. RHS)
        row    = self.tableau[row_idx, :-1]
        f0     = frac(self.tableau[row_idx, -1])

        # Cut row coefficients: −frac(ā_ij) for each existing column
        cut_row = np.zeros(n_cols + 2)          # +1 new slack, +1 RHS
        for j in range(n_cols):
            cut_row[j] = -frac(row[j])
        cut_row[n_cols] = 1.0                   # new slack coefficient
        cut_row[-1]     = -f0                   # RHS (negative → dual simplex needed)

        # Expand every existing row with a zero entry for the new slack column
        zeros_col = np.zeros((self.tableau.shape[0], 1))
        self.tableau = np.hstack([self.tableau[:, :-1], zeros_col,
                                   self.tableau[:, -1:]])

        # Insert the cut row BEFORE the objective row (which must stay last).
        # Using vstack([all rows except last, cut_row, last row]).
        self.tableau = np.vstack([self.tableau[:-1], cut_row, self.tableau[-1:]])

        # New slack enters the basis for the cut row
        self.basis.append(n_cols)   # n_cols is the index of the new slack column

    # ── Main entry point ─────────────────────────────────────────────────

    def solve(self):
        self.read_problem()
        self.read_integer_vars()

        # ── Step 1: LP relaxation ──────────────────────────────────────
        print("\n" + "=" * 58)
        print("  LP RELAXATION")
        print("=" * 58)
        self._solve_lp(print_steps=True)

        sol, obj = self._extract_solution()
        print("\n  LP relaxation solution:")
        for i in range(self.n):
            tag = "  [integer required]" if i in self.int_vars else ""
            print(f"    x{i+1} = {sol[i]:.6f}{tag}")
        print(f"  Objective = {obj:.6f}")

        if not self.int_vars:
            print("\n  No integrality constraints — LP solution is final.")
            self._print_result(sol, obj)
            return

        # ── Step 2: Gomory cutting-plane loop ─────────────────────────
        print("\n" + "=" * 58)
        print("  GOMORY CUTTING PLANE ITERATIONS")
        print("=" * 58)

        for cut_num in range(1, MAX_CUTS + 1):
            row_idx = self._find_fractional_row()
            if row_idx == -1:
                break   # all integer variables are integral → done

            # Identify which variable is fractional
            frac_var_name = "?"
            for j in self.int_vars:
                if self.basis[row_idx] in self.col_map[j]:
                    frac_var_name = f"x{j+1}"

            frac_val = self.tableau[row_idx, -1]
            print(f"\n  Cut {cut_num:3d} | {frac_var_name} = {frac_val:.6f} "
                  f"(frac = {frac(frac_val):.6f})")

            self._add_gomory_cut(row_idx)
            self.run_dual_simplex(print_steps=True)

            sol, obj = self._extract_solution()
            print(f"          | obj = {obj:.6f}")
            for i in range(self.n):
                marker = "*" if i in self.int_vars else " "
                print(f"          | [{marker}] x{i+1} = {sol[i]:.6f}")

        else:
            print(f"\n  WARNING: Cut limit ({MAX_CUTS}) reached. "
                  "Solution may not be fully integer.")

        sol, obj = self._extract_solution()

        # Round integer variables that are within tolerance
        for j in self.int_vars:
            sol[j] = float(round(sol[j]))

        self._print_result(sol, obj)

    # ── Pretty-print final answer ─────────────────────────────────────────

    def _print_result(self, sol, obj):
        print("\n" + "=" * 58)
        print("  OPTIMAL SOLUTION")
        print("=" * 58)
        for i in range(self.n):
            tag = "  [integer]" if i in self.int_vars else ""
            print(f"  x{i+1} = {sol[i]:.6f}{tag}")
        print(f"\n  Optimal objective value = {obj:.6f}")
        print("=" * 58)


# ─────────────────────────────────────────────
if __name__ == "__main__":
    try:
        solver = GomoryMILP()
        solver.solve()
    except Exception as e:
        print("Error:", e)
