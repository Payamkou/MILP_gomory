import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import milp_gomory  # noqa: E402
from milp_gomory import GomoryMILP, solve_milp  # noqa: E402


def feasible(x, cons, tol=1e-6):
    for coeffs, sense, rhs in cons:
        v = float(np.dot(coeffs, x))
        if (sense == "<=" and v > rhs + tol) or (sense == ">=" and v < rhs - tol) \
                or (sense == "=" and abs(v - rhs) > tol):
            return False
    return True


def test_readme_example_1_pure_integer():
    x, obj, status = solve_milp([1, 1], [([2, 2], "<=", 7)], int_vars=[0, 1])
    assert status == "optimal"
    assert obj == pytest.approx(3)
    assert x.sum() == pytest.approx(3)


def test_readme_example_2_mixed_integer():
    # The plain fractional Gomory cut is invalid here (x2 is continuous) and
    # cut off the optimum: the old code returned 23 at (3, 2).
    cons = [([1, 1], "<=", 5), ([10, 6], "<=", 45)]
    x, obj, status = solve_milp([5, 4], cons, int_vars=[0])
    assert status == "optimal"
    assert np.allclose(x, [4, 5 / 6])
    assert obj == pytest.approx(70 / 3)


def test_pure_lp_when_no_integer_variables():
    x, obj, status = solve_milp([3, 2], [([1, 1], "<=", 4), ([2, 1], "=", 7)])
    assert status == "optimal"
    assert np.allclose(x, [3, 1])
    assert obj == pytest.approx(11)


def test_readme_example_4_free_variable_lp_is_unbounded():
    with pytest.raises(ValueError, match="UNBOUNDED"):
        solve_milp([1, 2], [([1, 1], "=", 3), ([2, -1], ">=", 1)],
                   var_types=[1, 3], maximize=False)


def test_mixed_integer_with_nonpositive_variable():
    # x1 <= 0 integer, x2 >= 0 continuous: optimum at x1 = -3, x2 = 14/3
    cons = [([0, 6], "<=", 28), ([7, 4], "<=", 2)]
    x, obj, status = solve_milp([1, 2], cons, var_types=[2, 1], int_vars=[0])
    assert status == "optimal"
    assert obj == pytest.approx(19 / 3)
    assert x[0] == pytest.approx(-3)


def test_artificial_left_in_basis_after_phase1():
    # Phase I leaves an artificial basic at zero; the old code then returned
    # x2 = 0, violating 3 x1 + 2 x2 >= 15. The only feasible point is (0, 7.5).
    x, obj, status = solve_milp([0, -1], [([1, 2], "<=", 15), ([3, 2], ">=", 15)],
                                var_types=[2, 1], int_vars=[0])
    assert status == "optimal"
    assert np.allclose(x, [0, 7.5])
    assert obj == pytest.approx(-7.5)


def test_integer_infeasible():
    # 1.5 <= x1 + x2 <= 1.75 has no integer solution
    with pytest.raises(ValueError, match="INFEASIBLE"):
        solve_milp([1, 1], [([2, 2], ">=", 3), ([2, 2], "<=", 3.5)], int_vars=[0, 1])


def test_divisibility_check_with_unbounded_variables():
    # 6 x1 + 4 x2 + 2 x3 = 23 with x1 free: no integer point, LP relaxation unbounded
    with pytest.raises(ValueError, match="INFEASIBLE"):
        solve_milp([5, 5, 4], [([6, 4, 2], "=", 23)], var_types=[3, 1, 1],
                   int_vars=[0, 1, 2])


def test_unbounded_milp():
    with pytest.raises(ValueError, match="UNBOUNDED problem"):
        solve_milp([1, 1], [([1, -1], "<=", 1.5)], int_vars=[0, 1])


def test_binary_knapsack():
    w = [12, 7, 11, 8, 9, 6, 5, 14, 3, 10]
    v = [24, 13, 23, 15, 16, 11, 9, 27, 5, 19]
    cons = [(w, "<=", 26)] + [([1 if k == j else 0 for k in range(10)], "<=", 1)
                              for j in range(10)]
    best = max(sum(v[j] for j in range(10) if mask >> j & 1)
               for mask in range(1 << 10)
               if sum(w[j] for j in range(10) if mask >> j & 1) <= 26)
    x, obj, status = solve_milp(v, cons, int_vars=range(10))
    assert status == "optimal"
    assert obj == pytest.approx(best)
    assert feasible(x, cons)


def test_inactive_cut_rows_are_dropped():
    s = GomoryMILP()
    s.load_problem([1, 1], [([2, 2], "<=", 7), ([1, 0], "<=", 3.2)], int_vars=[0, 1])
    s._solve_lp()
    assert s._add_gomory_cuts(s._fractional_rows())
    s.run_dual_simplex()
    s._drop_inactive_cuts()
    assert not any(s.cut_cols[col] for col in s.basis)
    assert s.tableau.shape == (len(s.basis) + 1, len(s.int_cols) + 1)


def test_branching_keeps_tableau_small_on_unbounded_region():
    # 2 x1 - 3 x2 = 0.5 has no integer solution, but its LP relaxation is
    # unbounded, so the search dives forever. Redundant bound rows must be
    # dropped so the tableau stays small until the node limit stops it.
    old = milp_gomory.MAX_NODES
    milp_gomory.MAX_NODES = 300
    try:
        s = GomoryMILP()
        s.load_problem([1, 0], [([2, -3], "=", 0.5)], int_vars=[0, 1])
        with pytest.raises(ValueError, match="undecided"):
            s.optimize()
        assert s.nodes == 300
        assert s.tableau.shape[0] < 20
    finally:
        milp_gomory.MAX_NODES = old


def test_interactive_solve(monkeypatch, capsys):
    answers = iter(["2", "5 4", "yes", "2", "1 1 5", "<=", "10 6 45", "<=", "1", "1", "1"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    GomoryMILP().solve()
    out = capsys.readouterr().out
    assert "OPTIMAL SOLUTION" in out
    assert "Optimal objective value = 23.333333" in out


def test_output_is_ascii(monkeypatch, capsys):
    # Windows consoles/pipes may use cp1252; printed text must not need Unicode.
    answers = iter(["2", "1 1", "yes", "1", "2 2 7", "<=", "1", "1", "1 2"])
    monkeypatch.setattr("builtins.input", lambda _prompt="": next(answers))
    GomoryMILP().solve()
    capsys.readouterr().out.encode("ascii")


def test_matches_scipy_on_random_problems():
    scipy_optimize = pytest.importorskip("scipy.optimize")
    rng = np.random.default_rng(1)
    checked = 0
    for _ in range(200):
        n, m = int(rng.integers(2, 6)), int(rng.integers(2, 6))
        c = rng.integers(-5, 10, n).astype(float)
        # bounded problems: non-negative variables and an upper bound on their sum
        cons = [(rng.integers(-3, 8, n).astype(float), str(rng.choice(["<=", ">=", "="])),
                 float(rng.integers(0, 30))) for _ in range(m)]
        cons.append((np.ones(n), "<=", 25.0))
        ints = [j for j in range(n) if rng.random() < 0.7]
        A = np.array([r for r, _, _ in cons])
        lo = [-np.inf if s == "<=" else b for _, s, b in cons]
        hi = [np.inf if s == ">=" else b for _, s, b in cons]
        ref = scipy_optimize.milp(-c, constraints=scipy_optimize.LinearConstraint(A, lo, hi),
                                  integrality=[1 if j in ints else 0 for j in range(n)],
                                  options={"presolve": False})
        if ref.status == 0:
            x, obj, status = solve_milp(c, cons, int_vars=ints)
            assert status == "optimal"
            assert obj == pytest.approx(-ref.fun, abs=1e-6)
            assert feasible(x, cons, 1e-5)
            checked += 1
        elif ref.status == 2:
            with pytest.raises(ValueError, match="INFEASIBLE"):
                solve_milp(c, cons, int_vars=ints)
    assert checked > 50
