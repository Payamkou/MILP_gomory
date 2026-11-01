import argparse
import heapq
import math

import numpy as np

EPS = 1e-9               # pivot / reduced-cost tolerance
FEAS_TOL = 1e-7          # feasibility tolerance (relative to the largest |RHS|)
FRAC_TOL = 1e-6          # a value within FRAC_TOL of an integer counts as integer
INT_TOL = 1e-9           # tableau coefficients this close to an integer are snapped
MAX_CUTS = 200           # total Gomory cuts before switching to branch-and-bound
CUTS_PER_ROUND = 10      # cuts added at once (one per fractional row) before re-optimising
AWAY = 0.01              # only cut rows whose fractional part lies in [AWAY, 1 − AWAY]
MAX_DYNAMISM = 1e6       # reject cuts whose largest / smallest coefficient exceeds this
STALL_ROUNDS = 5         # rounds without objective progress before branching takes over
MAX_NODES = 20_000       # branch-and-bound node limit
PLUNGE_DEPTH = 25        # depth-first steps before returning to best-bound order
DEGENERATE_LIMIT = 50    # consecutive degenerate pivots before switching to Bland's rule
MAX_ITER = 100_000       # safety net against numerical stalling


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


def safe_choice(prompt, choices):
    while True:
        ans = input(prompt).strip()
        if ans in choices:
            return ans
        print(f"  Choose one of: {' / '.join(choices)}")


def frac(x):
    """Fractional part of x, always in [0, 1)."""
    return x - math.floor(x)


def _clean(v):
    """Avoid printing -0.000000."""
    v = float(v)
    return 0.0 if abs(v) < 5e-7 else v


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
        self.slack_cols = []  # slack_cols[i] = slack/surplus column of row i (or None)
        self.col_map = []     # col_map[j] = list of tableau column indices for x_j
        self.var_matrix = None  # x = var_matrix @ (decision columns of the tableau)
        self.n_dec = 0        # number of decision columns in the tableau
        self.iterations = 0

    # ── Input ────────────────────────────────────────────────────────────

    def load_problem(self, c, constraints, var_types=None, maximize=True):
        """
        Set the problem without prompting.

        c           : objective coefficients
        constraints : list of (coefficients, sense, rhs), sense in "<=", ">=", "="
        var_types   : 1 = x≥0, 2 = x≤0, 3 = free  (default: all x≥0)
        """
        self.n = len(c)
        self.is_max = maximize
        self.c = np.asarray(c, dtype=float) * (1.0 if maximize else -1.0)
        self.constraints = []
        for coeffs, sense, rhs in constraints:
            coeffs = np.asarray(coeffs, dtype=float)
            if coeffs.shape != (self.n,):
                raise ValueError(f"Constraint needs {self.n} coefficients.")
            if sense not in ("<=", ">=", "="):
                raise ValueError(f"Unknown constraint type '{sense}'.")
            self.constraints.append((coeffs, sense, float(rhs)))
        self.var_types = [1] * self.n if var_types is None else list(var_types)
        if len(self.var_types) != self.n or any(t not in (1, 2, 3) for t in self.var_types):
            raise ValueError("Variable types must be 1, 2 or 3 (one per variable).")

    def read_problem(self):
        self.n = safe_int("Number of decision variables: ")
        self.c = np.array(
            safe_float_list(f"Enter {self.n} objective coefficients: ", self.n)
        )
        ans = input("Maximization problem? (yes/no): ").lower()
        self.is_max = ans in ["yes", "y"]
        if not self.is_max:
            self.c = -self.c   # internally always maximise

        self.constraints = []
        m = safe_int("Number of constraints: ")
        print("Constraint types: <=, >=, =")
        for i in range(m):
            row = safe_float_list(
                f"Constraint {i+1} (coefficients + RHS): ", self.n + 1
            )
            sense = safe_choice("Type (<= / >= / =): ", ["<=", ">=", "="])
            self.constraints.append((np.array(row[:-1]), sense, row[-1]))

        self.var_types = []
        print("Variable bounds:  1 = x>=0   2 = x<=0   3 = free")
        for i in range(self.n):
            while True:
                t = safe_int(f"  x{i+1}: ")
                if t in [1, 2, 3]:
                    self.var_types.append(t)
                    break
                print("  Enter 1, 2, or 3.")

    # ── Build initial tableau ─────────────────────────────────────────────

    def build_tableau(self):
        n, m = self.n, len(self.constraints)

        # Assign tableau columns to decision variables
        self.col_map = []
        col = 0
        for t in self.var_types:
            width = 2 if t == 3 else 1          # free: x = x⁺ − x⁻
            self.col_map.append(list(range(col, col + width)))
            col += width
        self.n_dec = col
        M = np.zeros((n, self.n_dec))
        for j, (t, cs) in enumerate(zip(self.var_types, self.col_map)):
            M[j, cs[0]] = -1.0 if t == 2 else 1.0
            if t == 3:
                M[j, cs[1]] = -1.0
        self.var_matrix = M

        A = np.array([coeffs for coeffs, _, _ in self.constraints], dtype=float).reshape(m, n)
        b = np.array([rhs for _, _, rhs in self.constraints], dtype=float)
        senses = [sense for _, sense, _ in self.constraints]

        # Normalise: negative RHS → flip row & sense so RHS becomes positive.
        flip = {"<=": ">=", ">=": "<=", "=": "="}
        neg = b < 0
        A[neg] *= -1
        b[neg] *= -1
        senses = [flip[s] if f else s for s, f in zip(senses, neg)]

        slack_count = sum(s != "=" for s in senses)
        artificial_count = sum(s != "<=" for s in senses)
        total_cols = self.n_dec + slack_count + artificial_count

        T = np.zeros((m + 1, total_cols + 1))
        T[:m, :self.n_dec] = A @ M
        T[:m, -1] = b

        self.basis = [0] * m
        self.artificial_cols = []
        self.slack_cols = [None] * m
        slack_ptr = self.n_dec
        art_ptr   = self.n_dec + slack_count
        for i, sense in enumerate(senses):
            if sense != "=":
                T[i, slack_ptr] = 1.0 if sense == "<=" else -1.0
                self.slack_cols[i] = slack_ptr
                if sense == "<=":
                    self.basis[i] = slack_ptr
                slack_ptr += 1
            if sense != "<=":
                T[i, art_ptr] = 1.0
                self.basis[i] = art_ptr
                self.artificial_cols.append(art_ptr)
                art_ptr += 1

        # Phase I objective: maximise −Σ artificials, already in canonical form
        art_rows = [i for i, c in enumerate(self.basis) if c >= self.n_dec + slack_count]
        T[-1] = T[art_rows].sum(axis=0)
        T[-1, self.artificial_cols] = 0.0

        self.tableau = T
        self.rhs_scale = max(1.0, float(np.abs(b).max(initial=0.0)))

    # ── Pivot ─────────────────────────────────────────────────────────────

    def pivot(self, r, c):
        T = self.tableau
        T[r] /= T[r, c]
        col = T[:, c].copy()
        col[r] = 0.0
        rows = np.flatnonzero(col)
        if 2 * rows.size > T.shape[0]:
            T -= np.outer(col, T[r])            # dense column: update in place
        else:
            T[rows] -= np.outer(col[rows], T[r])
        T[:, c] = 0.0
        T[r, c] = 1.0
        self.basis[r] = c

    # ── Primal Simplex ────────────────────────────────────────────────────

    def run_simplex(self, print_steps=False):
        """
        Dantzig's rule (largest reduced cost) for speed; switches to Bland's
        rule after DEGENERATE_LIMIT consecutive degenerate pivots so it cannot
        cycle.
        """
        T = self.tableau
        m = len(self.basis)
        degenerate = 0
        for iteration in range(1, MAX_ITER + 1):
            costs = T[-1, :-1]
            bland = degenerate >= DEGENERATE_LIMIT
            if bland:
                candidates = np.flatnonzero(costs > EPS)
                if candidates.size == 0:
                    return   # optimal
                entering = int(candidates[0])
            else:
                entering = int(np.argmax(costs))
                if costs[entering] <= EPS:
                    return   # optimal

            column = T[:m, entering]
            rows = np.flatnonzero(column > EPS)
            if rows.size == 0:
                raise ValueError("UNBOUNDED problem.")
            ratios = np.maximum(T[rows, -1], 0.0) / column[rows]
            best = ratios.min()
            ties = rows[ratios <= best + EPS]
            if bland:
                leaving = int(min(ties, key=lambda i: self.basis[i]))
            else:
                leaving = int(ties[np.argmax(column[ties])])   # most stable pivot

            degenerate = degenerate + 1 if best <= EPS else 0
            self.pivot(leaving, entering)
            self.iterations += 1

            if print_steps:
                print(f"    Primal iter {iteration}")
        raise RuntimeError("Iteration limit reached (numerical stalling).")

    # ── Dual Simplex  (used after adding Gomory cuts) ─────────────────────

    def run_dual_simplex(self, print_steps=False):
        """
        Restores primal feasibility after a cut introduces a negative RHS.

        Pre-condition: the current basis is dual feasible, i.e. the objective
        row has no positive entries (all reduced costs ≤ 0 for maximisation).
        This is guaranteed immediately after a primal-optimal LP solution.

        Like the primal simplex, it switches to Bland's rule (smallest basic
        index leaves, smallest column index enters) after DEGENERATE_LIMIT
        consecutive degenerate pivots, so it cannot cycle.
        """
        T = self.tableau
        m = len(self.basis)
        tol = FEAS_TOL * self.rhs_scale
        degenerate = 0
        for iteration in range(1, MAX_ITER + 1):
            rhs = T[:m, -1]
            bland = degenerate >= DEGENERATE_LIMIT

            # Leaving row: most-negative RHS
            if bland:
                negative = np.flatnonzero(rhs < -tol)
                leaving = int(min(negative, key=lambda i: self.basis[i])) if negative.size else -1
            else:
                leaving = int(np.argmin(rhs))
                if rhs[leaving] >= -tol:
                    leaving = -1
            if leaving == -1:
                np.maximum(rhs, 0.0, out=rhs)   # clear round-off negatives
                return   # all RHS ≥ 0 → primal feasible

            row = T[leaving, :-1]

            # Entering column: minimum obj_row[j] / row[j] over row[j] < 0
            cols = np.flatnonzero(row < -EPS)
            if cols.size == 0:
                raise ValueError("Problem is INFEASIBLE after adding cut.")
            ratios = np.minimum(T[-1, cols], 0.0) / row[cols]
            best = ratios.min()
            ties = cols[ratios <= best + EPS]
            if bland:
                entering = int(ties[0])
            else:
                entering = int(ties[np.argmin(row[ties])])   # largest |pivot|
            degenerate = degenerate + 1 if best <= EPS else 0

            self.pivot(leaving, entering)
            self.iterations += 1

            if print_steps:
                print(f"    Dual iter {iteration}")
        raise RuntimeError("Iteration limit reached (numerical stalling).")

    # ── Phase I → Phase II ────────────────────────────────────────────────

    def end_phase1(self):
        """
        Check feasibility, then remove the artificial columns.

        An artificial still basic at value 0 is pivoted out on any non-zero
        entry of its row; if the row has none, it is redundant and deleted.
        (Leaving such a row without a basic variable would let Phase II
        violate the original constraint.)
        """
        T = self.tableau
        if T[-1, -1] > FEAS_TOL * self.rhs_scale:
            raise ValueError("Problem is INFEASIBLE.")

        is_art = np.zeros(T.shape[1] - 1, dtype=bool)
        is_art[self.artificial_cols] = True
        redundant = set()
        for i, col in enumerate(self.basis):
            if not is_art[col]:
                continue
            row = np.abs(T[i, :-1])
            row[is_art] = 0.0
            j = int(np.argmax(row))
            if row[j] > FEAS_TOL:
                T[i, -1] = 0.0       # degenerate pivot keeps every RHS unchanged
                self.pivot(i, j)
            else:
                redundant.add(i)

        keep_rows = [i for i in range(len(self.basis)) if i not in redundant]
        keep_cols = np.flatnonzero(~is_art)
        self._keep_columns(keep_rows, keep_cols)
        self.artificial_cols = []

    def _keep_columns(self, keep_rows, keep_cols):
        """Restrict the tableau to the given constraint rows and columns."""
        T = self.tableau
        new_index = np.full(T.shape[1] - 1, -1)
        new_index[keep_cols] = np.arange(len(keep_cols))
        self.tableau = T[np.ix_(list(keep_rows) + [T.shape[0] - 1],
                                np.append(keep_cols, T.shape[1] - 1))]
        self.basis = [int(new_index[self.basis[i]]) for i in keep_rows]
        self.slack_cols = [None if self.slack_cols[i] is None else int(new_index[self.slack_cols[i]])
                           for i in keep_rows]

    # ── Phase II objective row ────────────────────────────────────────────

    def build_phase2_objective(self):
        T = self.tableau
        obj = np.zeros(T.shape[1])
        obj[:self.n_dec] = self.c @ self.var_matrix
        # Canonicalise: zero out reduced costs of current basic variables
        obj -= obj[self.basis] @ T[:-1]
        T[-1] = obj

    # ── Solve LP relaxation ───────────────────────────────────────────────

    def _solve_lp(self, print_steps=False):
        self.iterations = 0
        self.build_tableau()

        if print_steps: print("  [Phase I]")
        self.run_simplex(print_steps=print_steps)
        self.end_phase1()

        if print_steps: print("  [Phase II]")
        self.build_phase2_objective()
        self.run_simplex(print_steps=print_steps)

    # ── Extract current solution from tableau ────────────────────────────

    def _extract_solution(self):
        T = self.tableau
        values = np.zeros(T.shape[1] - 1)
        values[self.basis] = T[:-1, -1]
        solution = self.var_matrix @ values[:self.n_dec]
        obj_val = float(self.c @ solution)
        if not self.is_max:
            obj_val = -obj_val
        return solution, obj_val


# ─────────────────────────────────────────────
# Mixed-Integer LP  via Gomory Cutting Planes
# ─────────────────────────────────────────────

class GomoryMILP(TwoPhaseSimplex):
    """
    Extends TwoPhaseSimplex to handle Mixed-Integer LP using
    Gomory mixed-integer (GMI) cutting planes.

    Algorithm
    ─────────
    1. Solve the LP relaxation (ignore integrality).
    2. If every integer-required variable in the current BFS already
       has an integer value → optimal MILP solution found.
    3. Otherwise, take the basis rows whose integer variables are
       fractional (most fractional first, up to CUTS_PER_ROUND).
    4. Derive a Gomory mixed-integer cut from each row and append it
       to the tableau (as a new ≥ constraint with its own slack).
    5. Apply the dual simplex to restore primal feasibility, then drop
       cuts that are no longer binding (their slack became basic).
    6. Go to 2 — until the cuts stop making progress (pure cutting planes
       tail off and lose accuracy); then finish with branch-and-bound on the
       strengthened LP, warm-started with the dual simplex at every node.

    Gomory mixed-integer cut
    ────────────────────────
    For canonical row i:  x_Bi + Σ_j ā_ij · x_j = b̄_i,  with f₀ = frac(b̄_i)
    and f_j = frac(ā_ij):
        integer x_j,     f_j ≤ f₀ :  f_j
        integer x_j,     f_j > f₀ :  f₀ (1 − f_j) / (1 − f₀)
        continuous x_j,  ā_ij ≥ 0 :  ā_ij
        continuous x_j,  ā_ij < 0 :  f₀ (−ā_ij) / (1 − f₀)
    Cut:  Σ_j coef_j · x_j ≥ f₀.   It is valid for mixed problems, where the
    plain fractional cut  Σ_j f_j x_j ≥ f₀  can cut off the true optimum,
    and it is at least as strong on pure-integer problems.
    As a tableau row (with new slack s ≥ 0):
        −Σ_j coef_j · x_j  +  s  =  −f₀     (RHS < 0)
    The new slack enters the basis with value −f₀ < 0,
    which is immediately repaired by the dual simplex.
    """

    def __init__(self):
        super().__init__()
        self.int_vars = []    # 0-based indices of integer-constrained variables
        self.int_cols = None  # int_cols[k]: tableau column k is integer in every MILP solution
        self.cut_cols = None  # cut_cols[k]: tableau column k is the slack of a Gomory cut
        self.obj_step = 0     # integer solutions have objective values that are multiples of this
        self.cuts_added = 0
        self.nodes = 0
        self.status = None    # "optimal", "feasible" (node limit, not proven) or "limit"

    @property
    def proven_optimal(self):
        return self.status == "optimal"

    # ── Read which variables must be integer ─────────────────────────────

    def load_problem(self, c, constraints, var_types=None, maximize=True, int_vars=()):
        """Like TwoPhaseSimplex.load_problem; int_vars are 0-based indices."""
        super().load_problem(c, constraints, var_types, maximize)
        if any(j < 0 or j >= self.n for j in int_vars):
            raise ValueError(f"Integer variable index out of range [0, {self.n - 1}].")
        self.int_vars = sorted(set(int(j) for j in int_vars))

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

    # ── Column bookkeeping ───────────────────────────────────────────────

    def build_tableau(self):
        super().build_tableau()
        T = self.tableau
        ncols = T.shape[1] - 1
        self.int_cols = np.zeros(ncols, dtype=bool)
        for j in self.int_vars:
            self.int_cols[self.col_map[j]] = True

        # A slack is integer when its row has integer data on integer columns only
        int_dec = self.int_cols[:self.n_dec]
        for i, col in enumerate(self.slack_cols):
            if col is None:
                continue
            row, rhs = T[i, :self.n_dec], T[i, -1]
            used = row != 0
            if (int_dec[used].all() and np.all(row == np.round(row))
                    and rhs == round(rhs)):
                self.int_cols[col] = True
        self.cut_cols = np.zeros(ncols, dtype=bool)

    def end_phase1(self):
        keep = np.ones(self.tableau.shape[1] - 1, dtype=bool)
        keep[self.artificial_cols] = False
        super().end_phase1()
        self._filter_column_info(keep)

    def _filter_column_info(self, keep):
        """Apply a column mask to the per-column bookkeeping."""
        self.int_cols = self.int_cols[keep]
        self.cut_cols = self.cut_cols[keep]

    def _append_rows(self, rows, rhs, is_cut):
        """
        Append constraint rows (already in canonical form over the current
        columns), each with its own new basic slack column. The new rows go
        just before the objective row, which stays last.
        """
        T = self.tableau
        m, ncols, k = len(self.basis), T.shape[1] - 1, len(rows)
        new = np.zeros((m + k + 1, ncols + k + 1))
        new[:m, :ncols] = T[:m, :-1]
        new[:m, -1]     = T[:m, -1]
        new[-1, :ncols] = T[-1, :-1]
        new[-1, -1]     = T[-1, -1]
        for r in range(k):
            new[m + r, :ncols]    = rows[r]
            new[m + r, ncols + r] = 1.0       # new slack coefficient
            new[m + r, -1]        = rhs[r]
            self.basis.append(ncols + r)      # new slack enters the basis
            self.slack_cols.append(ncols + r)
        self.tableau = new
        self.int_cols = np.append(self.int_cols, np.zeros(k, dtype=bool))
        self.cut_cols = np.append(self.cut_cols, np.full(k, is_cut))

    # ── Find fractional integer-required variables ───────────────────────

    def _fractional_vars(self, sol):
        """Integer-required variables with a fractional value, most fractional first."""
        idx = np.asarray(self.int_vars, dtype=int)
        dist = np.abs(sol[idx] - np.round(sol[idx]))
        order = np.argsort(-dist, kind="stable")
        return [int(idx[k]) for k in order if dist[k] > FRAC_TOL]

    def _fractional_rows(self):
        """
        Rows whose basic variable is a decision column of an integer variable
        with a fractional part in [AWAY, 1 − AWAY], most fractional first.
        (Rows that are almost integral give weak, badly scaled cuts.)
        """
        basis = np.asarray(self.basis, dtype=int)
        vals = self.tableau[:len(basis), -1]
        f = vals - np.floor(vals)
        dist = np.minimum(f, 1.0 - f)   # distance to nearest integer
        is_int_dec = self.int_cols[basis] & (basis < self.n_dec)
        rows = np.flatnonzero(is_int_dec & (dist >= AWAY))
        return rows[np.argsort(-dist[rows], kind="stable")]

    def _find_fractional_row(self):
        """Index of the most fractional row, or -1 if there is none."""
        rows = self._fractional_rows()
        return int(rows[0]) if rows.size else -1

    # ── Generate and insert Gomory cuts ──────────────────────────────────

    def _gomory_cut(self, row_idx):
        """
        Coefficients (over current columns) and f₀ of the GMI cut from a row,
        or None if the cut is too badly scaled to be used safely.
        """
        T  = self.tableau
        a  = T[row_idx, :-1].copy()
        f0 = frac(T[row_idx, -1])

        snapped = np.round(a)
        a = np.where(np.abs(a - snapped) < INT_TOL, snapped, a)   # strip round-off
        f = a - np.floor(a)
        coef = np.where(
            self.int_cols,
            np.where(f <= f0, f, f0 * (1.0 - f) / (1.0 - f0)),
            np.where(a >= 0.0, a, f0 * (-a) / (1.0 - f0)),
        )
        coef[self.basis] = 0.0          # basic columns never appear in the cut
        if not np.any(coef > 0.0):
            return coef, f0             # 0 ≥ f₀: the row has no integer solution
        significant = coef[coef > EPS]
        if significant.size == 0 or significant.max() > MAX_DYNAMISM * significant.min():
            return None
        return coef, f0

    def _add_gomory_cuts(self, rows):
        """Append one GMI cut per row in `rows`; returns the rows actually used."""
        used, coefs, rhs = [], [], []
        for i in rows:
            cut = self._gomory_cut(i)
            if cut is not None:
                used.append(i)
                coefs.append(-cut[0])
                rhs.append(-cut[1])            # RHS < 0 → dual simplex needed
        if used:
            self._append_rows(coefs, rhs, is_cut=True)
            self.cuts_added += len(used)
        return used

    def _add_gomory_cut(self, row_idx):
        return bool(self._add_gomory_cuts([row_idx]))

    def _drop_inactive_cuts(self):
        """
        A cut whose slack is basic is not binding; its row and slack column
        (a unit column) can be removed without changing the current solution.
        This keeps the tableau from growing with every cut.
        """
        inactive = {i for i, col in enumerate(self.basis) if self.cut_cols[col]}
        if not inactive:
            return
        keep_rows = [i for i in range(len(self.basis)) if i not in inactive]
        keep = np.ones(self.tableau.shape[1] - 1, dtype=bool)
        keep[[self.basis[i] for i in inactive]] = False
        self._keep_columns(keep_rows, np.flatnonzero(keep))
        self._filter_column_info(keep)

    # ── Branch-and-bound ─────────────────────────────────────────────────

    def _add_bound(self, j, sense, k):
        """Append  x_j <= k  or  x_j >= k  as a new canonical tableau row."""
        T = self.tableau
        m = len(self.basis)
        row = np.zeros(T.shape[1])
        row[:self.n_dec] = self.var_matrix[j]   # x_j in tableau columns
        row[-1] = k
        if sense == ">=":
            row = -row                          # −x_j ≤ −k
        row -= row[self.basis] @ T[:m]          # eliminate basic columns
        self._append_rows([row[:-1]], [row[-1]], is_cut=False)

    def _snapshot(self):
        return (self.tableau.copy(), list(self.basis), list(self.slack_cols),
                self.int_cols, self.cut_cols)

    def _restore(self, snap):
        tableau, basis, slack_cols, int_cols, cut_cols = snap
        self.tableau = tableau.copy()
        self.basis = list(basis)
        self.slack_cols = list(slack_cols)
        self.int_cols, self.cut_cols = int_cols, cut_cols

    def _objective_step(self):
        """
        If only integer variables have (integer) objective coefficients, every
        integer solution's objective is a multiple of their gcd; else 0.
        """
        used = np.flatnonzero(self.c)
        if used.size == 0 or not set(used) <= set(self.int_vars) \
                or np.any(self.c[used] != np.round(self.c[used])):
            return 0
        return math.gcd(*(int(abs(a)) for a in self.c[used]))

    def _cannot_improve(self, z, best_z):
        """True if an LP bound z (maximised) cannot beat the incumbent best_z."""
        if self.obj_step:
            z = math.floor(z / self.obj_step + 1e-6) * self.obj_step
        return z <= best_z + 1e-9 * max(1.0, abs(best_z))

    def _branch_and_bound(self, print_steps, say):
        """
        Best-first branch-and-bound with plunging on the strengthened LP.

        A node is just its set of bounds (x_j <= u / x_j >= l, the tightest
        per variable and side). After branching, the search plunges into the
        nearer child (warm-started from the parent's tableau with one extra
        bound row) for up to PLUNGE_DEPTH levels, which finds integer solutions
        early; the other children wait in a queue ordered by LP bound and are
        solved by adding their bounds to the tableau left by the cutting phase.
        Best-bound order avoids endless dives along unbounded integer
        variables, and queueing bounds instead of tableaux keeps memory small.
        Returns (best solution or None, complete).
        """
        root = self._snapshot()
        self.obj_step = self._objective_step()
        best_z, best_sol = -np.inf, None
        heap = [(-np.inf, 0, {})]       # (−parent LP bound, insertion order, bounds)
        order = 1
        self.nodes = 0
        dive, depth = None, 0           # child being plunged into: (bounds, new bound)
        while heap or dive:
            if self.nodes >= MAX_NODES:
                say(f"\n  WARNING: Node limit ({MAX_NODES}) reached.")
                return best_sol, False
            if dive is not None:
                bounds, new_bound = dive
                dive = None
                self._add_bound(*new_bound)           # warm start from the parent
            else:
                neg_bound, _, bounds = heapq.heappop(heap)
                if best_sol is not None and self._cannot_improve(-neg_bound, best_z):
                    continue                          # parent bound already too weak
                self._restore(root)
                for (j, sense), k in bounds.items():
                    self._add_bound(j, sense, k)
                depth = 0
            if bounds:
                try:
                    self.run_dual_simplex(print_steps=print_steps)
                except ValueError:
                    continue                          # infeasible branch
                self._drop_inactive_cuts()
            self.nodes += 1

            sol, _ = self._extract_solution()
            z = float(self.c @ sol)                   # internal (maximised) objective
            if best_sol is not None and self._cannot_improve(z, best_z):
                continue                              # cannot beat the incumbent
            frac_vars = self._fractional_vars(sol)
            if not frac_vars:
                best_z, best_sol = z, sol.copy()
                say(f"  Node {self.nodes:5d} | new incumbent, obj = "
                    f"{_clean(z if self.is_max else -z):.6f}")
                continue

            j = frac_vars[0]
            lo = math.floor(sol[j])
            children = [(j, "<=", lo), (j, ">=", lo + 1)]
            if sol[j] - lo >= 0.5:
                children.reverse()                    # nearer side first
            for i, (jj, sense, k) in enumerate(children):
                child = dict(bounds)
                child[jj, sense] = k
                if i == 0 and depth < PLUNGE_DEPTH:
                    dive, depth = (child, (jj, sense, k)), depth + 1
                else:
                    heapq.heappush(heap, (-z, order, child))
                    order += 1
        return best_sol, True

    # ── Solve ─────────────────────────────────────────────────────────────

    def optimize(self, print_steps=False, log=None):
        """
        Solve the loaded problem; returns (solution, objective value).

        `log` (e.g. print) receives a progress trace. self.status is
        "optimal", "feasible" (node limit hit; best solution found so far)
        or "limit" (node limit hit before any integer solution; the returned
        point is the LP solution after cutting, i.e. a bound).
        """
        say = log if log is not None else (lambda *_: None)
        self.cuts_added = 0
        self.nodes = 0
        self.status = None
        self._check_divisibility()

        # ── Step 1: LP relaxation ──────────────────────────────────────
        say("\n" + "=" * 58)
        say("  LP RELAXATION")
        say("=" * 58)
        try:
            self._solve_lp(print_steps=print_steps)
        except ValueError as e:
            if "UNBOUNDED" in str(e) and self.int_vars:
                self._classify_unbounded(print_steps)
            raise

        sol, obj = self._extract_solution()
        say("\n  LP relaxation solution:")
        for i in range(self.n):
            tag = "  [integer required]" if i in self.int_vars else ""
            say(f"    x{i+1} = {_clean(sol[i]):.6f}{tag}")
        say(f"  Objective = {_clean(obj):.6f}")

        if not self.int_vars:
            say("\n  No integrality constraints - LP solution is final.")
            self.status = "optimal"
            return sol, obj

        # ── Step 2: Gomory cutting-plane loop ─────────────────────────
        say("\n" + "=" * 58)
        say("  GOMORY CUTTING PLANE ITERATIONS")
        say("=" * 58)

        history = [obj]
        while self._fractional_vars(sol):
            if self.cuts_added >= MAX_CUTS:
                say(f"\n  Cut limit ({MAX_CUTS}) reached.")
                break
            if len(history) > STALL_ROUNDS and \
                    abs(history[-1 - STALL_ROUNDS] - history[-1]) <= 1e-6 * max(1.0, abs(obj)):
                say(f"\n  No progress in the last {STALL_ROUNDS} rounds.")
                break
            rows = self._fractional_rows()[:min(CUTS_PER_ROUND, MAX_CUTS - self.cuts_added)]
            first = self.cuts_added + 1
            rows = self._add_gomory_cuts(rows)
            if not rows:
                say("\n  No usable cut (remaining values are almost integral).")
                break
            for k, i in enumerate(rows):
                name = next(f"x{j+1}" for j in self.int_vars
                            if self.basis[i] in self.col_map[j])
                val = self.tableau[i, -1]
                say(f"\n  Cut {first + k:3d} | {name} = {val:.6f} "
                    f"(frac = {frac(val):.6f})")

            self.run_dual_simplex(print_steps=print_steps)
            self._drop_inactive_cuts()

            sol, obj = self._extract_solution()
            history.append(obj)
            say(f"          | obj = {_clean(obj):.6f}")
            for i in range(self.n):
                marker = "*" if i in self.int_vars else " "
                say(f"          | [{marker}] x{i+1} = {_clean(sol[i]):.6f}")

        # ── Step 3: branch-and-bound on the strengthened LP ────────────
        if self._fractional_vars(sol):
            say("\n" + "=" * 58)
            say("  BRANCH AND BOUND  (after cuts)")
            say("=" * 58)
            best, complete = self._branch_and_bound(print_steps, say)
            say(f"\n  Explored {self.nodes} nodes.")
            if best is None:
                if complete:
                    raise ValueError("Problem is INFEASIBLE (no integer solution).")
                self.status = "limit"
                return sol, obj
            sol = best
            self.status = "optimal" if complete else "feasible"
        else:
            self.status = "optimal"

        # Snap integer variables that are within tolerance
        sol[self.int_vars] = np.round(sol[self.int_vars])
        obj = float(self.c @ sol) * (1.0 if self.is_max else -1.0)
        return sol, obj

    def _check_divisibility(self):
        """
        An equality  Σ a_j x_j = b  over integer variables with integer a_j has
        no integer solution unless gcd(a_j) divides b (e.g. 2x + 4y = 7).
        Cuts and branching cannot prove this when the variables are unbounded.
        """
        is_int = np.zeros(self.n, dtype=bool)
        is_int[self.int_vars] = True
        for coeffs, sense, rhs in self.constraints:
            used = coeffs != 0
            if (sense != "=" or not used.any() or not is_int[used].all()
                    or np.any(coeffs != np.round(coeffs)) or rhs != round(rhs)):
                continue
            g = math.gcd(*(int(a) for a in coeffs[used]))
            if int(rhs) % g:
                raise ValueError("Problem is INFEASIBLE (no integer solution).")

    def _classify_unbounded(self, print_steps=False):
        """
        The LP relaxation is unbounded. With rational data the MILP is then
        unbounded if it has any integer-feasible point, else infeasible.
        Look for one with a zero objective (which cannot be unbounded).
        """
        c = self.c
        self.c = np.zeros(self.n)
        try:
            self.optimize(print_steps)      # raises if no integer point exists
        finally:
            self.c = c
        if self.status in ("optimal", "feasible"):
            raise ValueError("UNBOUNDED problem.")
        raise ValueError("UNBOUNDED LP relaxation; integer feasibility "
                         f"undecided after {MAX_NODES} nodes.")

    def solve(self, print_steps=False):
        self.read_problem()
        self.read_integer_vars()
        sol, obj = self.optimize(print_steps=print_steps, log=print)
        self._print_result(sol, obj)

    # ── Pretty-print final answer ─────────────────────────────────────────

    def _print_result(self, sol, obj):
        heading, label = {
            "optimal":  ("OPTIMAL SOLUTION", "Optimal objective value"),
            "feasible": ("BEST SOLUTION FOUND (NOT PROVEN OPTIMAL)", "Objective value"),
            "limit":    ("NO INTEGER SOLUTION FOUND - LP BOUND", "LP bound"),
        }[self.status]
        print("\n" + "=" * 58)
        print(f"  {heading}")
        print("=" * 58)
        for i in range(self.n):
            tag = "  [integer]" if i in self.int_vars else ""
            print(f"  x{i+1} = {_clean(sol[i]):.6f}{tag}")
        print(f"\n  {label} = {_clean(obj):.6f}")
        print("=" * 58)


def solve_milp(c, constraints, var_types=None, maximize=True, int_vars=()):
    """Convenience wrapper: returns (solution, objective value, status)."""
    solver = GomoryMILP()
    solver.load_problem(c, constraints, var_types, maximize, int_vars)
    sol, obj = solver.optimize()
    return sol, obj, solver.status


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="MILP solver (Gomory cutting planes)")
    parser.add_argument("--steps", action="store_true",
                        help="print every simplex pivot")
    args = parser.parse_args()
    try:
        solver = GomoryMILP()
        solver.solve(print_steps=args.steps)
    except (ValueError, RuntimeError) as e:
        print("Error:", e)
    except (EOFError, KeyboardInterrupt):
        print("\nInput aborted.")
