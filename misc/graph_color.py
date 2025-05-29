from ortools.math_opt.python import mathopt
from ortools.sat.python import cp_model
import random

def apply_branches(graph, branch_ops):
    g = {i: set(neigh) for i, neigh in graph.items()}
    mapping = {i: i for i in g}
    for op, u, v in branch_ops:
        gu, gv = mapping[u], mapping[v]
        if op == 'DIFFER':
            g[gu].add(gv)
            g[gv].add(gu)
        else:
            g[gu] |= g[gv]
            g[gu].discard(gu)
            for w in list(g.keys()):
                if w != gv and gv in g[w]:
                    g[w].remove(gv)
                    g[w].add(gu)
            del g[gv]
            for k in mapping:
                if mapping[k] == gv:
                    mapping[k] = gu
    return g

def solve_lp_column_generation(graph, branch_ops=None):
    if branch_ops is None:
        branch_ops = []
    g = apply_branches(graph, branch_ops)
    nodes = list(g)
    columns = [{i} for i in nodes]
    eps = 1e-6
    while True:
        master = mathopt.Model()
        x = [master.add_variable(lb=0.0, name=f'x{p}') for p in range(len(columns))]
        constrs = []
        for i in nodes:
            c = master.add_linear_constraint(
                sum((1 if i in s else 0) * x[p] for p, s in enumerate(columns)) >= 1
            )
            constrs.append(c)
        master.minimize(sum(x))
        lp = mathopt.solve(master, mathopt.SolverType.GLOP)
        assert lp.termination.reason == mathopt.TerminationReason.OPTIMAL
        duals = [lp.dual_values()[c] for c in constrs]
        sub = mathopt.Model()
        z = {i: sub.add_integer_variable(lb=0, ub=1, name=f'z{i}') for i in nodes}
        for i in nodes:
            for j in g[i]:
                if i < j:
                    sub.add_linear_constraint(z[i] + z[j] <= 1)
        sub.maximize(sum(duals[nodes.index(i)] * z[i] for i in nodes))
        sp = mathopt.solve(sub, mathopt.SolverType.CP_SAT)
        assert sp.termination.reason == mathopt.TerminationReason.OPTIMAL
        val = sp.objective_value()
        if val <= 1 + eps:
            break
        new_s = {i for i in nodes if int(sp.variable_values()[z[i]])}
        columns.append(new_s)
    lp_sol = {idx: lp.variable_values()[x[idx]] for idx in range(len(columns))}
    bound = lp.objective_value()
    return lp_sol, columns, bound

def is_integer_solution(sol):
    return all(abs(v - round(v)) < 1e-6 for v in sol.values())

def pick_fractional_pair(sol, columns):
    for i, s1 in enumerate(columns):
        v1 = sol[i]
        if v1 < 1e-6 or abs(v1 - 1) < 1e-6:
            continue
        for j, s2 in enumerate(columns):
            if j <= i:
                continue
            v2 = sol[j]
            if v2 < 1e-6 or abs(v2 - 1) < 1e-6:
                continue
            inter = s1 & s2
            diff = s1 - s2
            if inter and diff:
                return next(iter(inter)), next(iter(diff))
    raise RuntimeError('No fractional pair')

def extract_coloring(columns, sol):
    coloring = {}
    cid = 0
    for idx, s in enumerate(columns):
        if round(sol[idx]) == 1:
            for i in s:
                if i not in coloring:
                    coloring[i] = cid
            cid += 1
    return coloring

def bap_solve(graph):
    best_value = float('inf')
    best_sol = None
    def recurse(branch_ops):
        nonlocal best_value, best_sol
        sol, cols, bound = solve_lp_column_generation(graph, branch_ops)
        if bound >= best_value - 1e-6:
            return
        if is_integer_solution(sol):
            obj = sum(sol.values())
            if obj < best_value:
                best_value = obj
                best_sol = extract_coloring(cols, sol)
            return
        i, j = pick_fractional_pair(sol, cols)
        recurse(branch_ops + [('DIFFER', i, j)])
        recurse(branch_ops + [('SAME', i, j)])
    recurse([])
    return best_value, best_sol

def naive_solve(graph):
    lp_sol, columns, _ = solve_lp_column_generation(graph)
    model = mathopt.Model()
    y = [model.add_integer_variable(lb=0, ub=1, name=f'y{p}') for p in range(len(columns))]
    for i in graph:
        model.add_linear_constraint(
            sum((1 if i in s else 0) * y[p] for p, s in enumerate(columns)) >= 1
        )
    model.minimize(sum(y))
    res = mathopt.solve(model, mathopt.SolverType.CP_SAT)
    assert res.termination.reason == mathopt.TerminationReason.OPTIMAL
    chromatic = int(res.objective_value())
    coloring = {}
    cid = 0
    for idx, s in enumerate(columns):
        if int(res.variable_values()[y[idx]]):
            for i in s:
                if i not in coloring:
                    coloring[i] = cid
            cid += 1
    return chromatic, coloring

def generate_random_graph(n, p):
    g = {i: set() for i in range(n)}
    for i in range(n):
        for j in range(i + 1, n):
            if random.random() < p:
                g[i].add(j)
                g[j].add(i)
    return g

def check_coloring(graph, coloring):
    for i, neigh in graph.items():
        if i not in coloring:
            return False
        for j in neigh:
            if coloring[i] == coloring.get(j):
                return False
    return True

if __name__ == '__main__':
    # g = {0:{1},1:{2},2:{3},3:{4},4:{0}}
    # val2, col2 = bap_solve(g)
    # print(val2, col2)
    

    # g = {0:{1,2},1:{0,2},2:{0,1,3},3:{2}}
    # val, col = bap_solve(g)
    # print('Chromatic number:', val)
    # print('Coloring:', col)
    rg = {0: {1, 2, 3, 5, 7, 8, 9, 10, 11, 13, 14, 15, 16, 18, 19}, 1: {0, 3, 4, 5, 7, 8, 10, 11, 14, 16, 17, 18}, 2: {0, 3, 7, 8, 10, 11, 12, 13, 14}, 3: {0, 1, 2, 9, 11, 12, 14, 15, 18, 19}, 4: {1, 5, 6, 7, 8, 9, 11, 13, 18}, 5: {0, 1, 4, 6, 7, 10, 12, 13, 14, 16, 19}, 6: {4, 5, 9, 12, 14, 15, 16, 17, 18, 19}, 7: {0, 1, 2, 4, 5, 8, 9, 10, 13, 18}, 8: {0, 1, 2, 4, 7, 14, 17, 19}, 9: {0, 3, 4, 6, 7, 11, 16}, 10: {0, 1, 2, 5, 7, 11, 13, 14}, 11: {0, 1, 2, 3, 4, 9, 10, 12, 14, 15, 17, 19}, 12: {2, 3, 5, 6, 11, 14, 15, 16}, 13: {0, 2, 4, 5, 7, 10, 15, 16}, 14: {0, 1, 2, 3, 5, 6, 8, 10, 11, 12, 15, 18}, 15: {0, 3, 6, 11, 12, 13, 14, 16, 17}, 16: {0, 1, 5, 6, 9, 12, 13, 15, 17, 18, 19}, 17: {1, 6, 8, 11, 15, 16, 18, 19}, 18: {0, 1, 3, 4, 6, 7, 14, 16, 17, 19}, 19: {0, 3, 5, 6, 8, 11, 16, 17, 18}}
    print(rg)
    val2, col2 = bap_solve(rg)
    print(col2, val2)

    # rg = generate_random_graph(50, 0.9)
    # chrom, col3 = naive_solve(rg)
    # print(col3)
    # print('\nNaive (n=50, p=0.9) chromatic number:', chrom)
    # assert check_coloring(rg, col3)
