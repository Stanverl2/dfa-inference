from ortools.math_opt.python import mathopt

# https://people.orie.cornell.edu/shmoys/or630/notes-06/lec16.pdf
# https://optimization.cbe.cornell.edu/index.php?title=Column_generation_algorithms

# simplex
# https://www.cs.cornell.edu/courses/cs6820/2017fa/handouts/splex.pdf
# https://optimization.cbe.cornell.edu/index.php?title=Simplex_algorithm

demands = [144, 105, 72, 30, 24]
sizes   = [6.0, 13.5, 15.0, 16.5, 22.5]
board_len  = 33.0

# demands = [6, 4]
# sizes   = [3.0, 2.0]
# board_len  = 7.0

# demands = [25, 20, 15]
# sizes = [5.0, 7.0, 9.0]
# board_len = 20.0

n = len(demands)
epsilon = 1e-6

patterns = []
for i, size in enumerate(sizes):
    assert size < board_len
    max_cuts = int(board_len // size)
    patterns.append([max_cuts if j == i else 0 for j in range(n)])

while True:
    master = mathopt.Model()
    x = [master.add_variable(lb=0.0, name=f"x{p}") for p in range(len(patterns))]
    master.minimize(sum(x))

    demand_constrs = []
    for i in range(n):
        c = master.add_linear_constraint(
            sum(patterns[p][i] * x[p] for p in range(len(patterns))) >= demands[i]
        )
        demand_constrs.append(c)

    lp = mathopt.solve(master, mathopt.SolverType.GLOP)
    assert lp.termination.reason == mathopt.TerminationReason.OPTIMAL
    duals = [lp.dual_values()[c] for c in demand_constrs]

    # print("duals", duals)

    sub = mathopt.Model()
    u = [sub.add_integer_variable(lb=0, name=f"u{i}") for i in range(n)]
    sub.maximize(sum(duals[i] * u[i] for i in range(n)))
    sub.add_linear_constraint(
        sum(sizes[i] * u[i] for i in range(n)) <= board_len
    )
    sp = mathopt.solve(sub, mathopt.SolverType.CP_SAT)
    assert sp.termination.reason == mathopt.TerminationReason.OPTIMAL
    qty = [int(sp.variable_values()[var]) for var in u]
    val = sp.objective_value()
    
    # print("sub problem sol", val, qty)

    if val <= 1 + epsilon:
        break
    patterns.append(qty)

print("all pattern count:", len(patterns))

mip = mathopt.Model()
y = [mip.add_integer_variable(lb=0, name=f"y{p}") for p in range(len(patterns))]
mip.minimize(sum(y))
for i in range(n):
    mip.add_linear_constraint(
        sum(patterns[p][i] * y[p] for p in range(len(patterns))) >= demands[i]
    )
res = mathopt.solve(mip, mathopt.SolverType.CP_SAT)
assert res.termination.reason in (
    mathopt.TerminationReason.OPTIMAL,
    mathopt.TerminationReason.FEASIBLE,
)

print("used patterns:")
total = 0
for p, qty in enumerate(patterns):
    cnt = int(round(res.variable_values()[y[p]]))
    if cnt > 0:
        print(qty, "x", cnt)
        total += cnt
print("total", total)
