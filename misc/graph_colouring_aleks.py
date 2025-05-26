from ortools.math_opt.python import mathopt

# provide edges and number of nodes.
n = 8
edges = [
    (1, 2), (1, 5), (2, 3), (2, 6), (3, 4),
    (3, 7), (4, 8), (5, 6), (6, 7), (7, 8),
    (1, 6), (2, 5)
]
epsilon = 1e-6

columns = []
for i in range(n):
    columns.append([1 if j == i else 0 for j in range(n)])

while True:
    master = mathopt.Model()
    x = [master.add_variable(lb=0.0, name=f"x{p}") for p in range(len(columns))]
    master.minimize(sum(x))
    # min x1 + x2 + x3 + x4

    independent_set_constraints = []
    for i in range(n):
        c = master.add_linear_constraint(
            sum(columns[p][i] * x[p] for p in range(len(columns))) == 1
        )
        independent_set_constraints.append(c)

    lp = mathopt.solve(master, mathopt.SolverType.GLOP)
    assert lp.termination.reason == mathopt.TerminationReason.OPTIMAL
    duals = [lp.dual_values()[c] for c in independent_set_constraints]

    # Pricing Problem:
    sub = mathopt.Model()
    y = [sub.add_integer_variable(lb=0, name=f"u{i}") for i in range(n)]
    sub.minimize(1 - sum(duals[i] * y[i] for i in range(n)))
    for u, v in edges:
        sub.add_linear_constraint(y[u-1] + y[v-1] <= 1)

    sp = mathopt.solve(sub, mathopt.SolverType.CP_SAT)
    assert sp.termination.reason == mathopt.TerminationReason.OPTIMAL
    qty = [int(sp.variable_values()[var]) for var in y]
    val = sp.objective_value()
    
    print(f"New column: {qty}, Reduced cost: {val}")

    if val + epsilon > 0:
        break
    columns.append(qty)

print("all independent set count:", len(columns))

mip = mathopt.Model()
y = [mip.add_integer_variable(lb=0, name=f"y{p}") for p in range(len(columns))]
mip.minimize(sum(y))
for i in range(n):
    mip.add_linear_constraint(
        sum(columns[p][i] * y[p] for p in range(len(columns))) == 1
    )
res = mathopt.solve(mip, mathopt.SolverType.CP_SAT)
assert res.termination.reason in (
    mathopt.TerminationReason.OPTIMAL,
    mathopt.TerminationReason.FEASIBLE,
)

print("used columns:")
total = 0
for p, qty in enumerate(columns):
    cnt = int(round(res.variable_values()[y[p]]))
    if cnt > 0:
        print(qty, "x", cnt)
        total += cnt
print("total", total)
