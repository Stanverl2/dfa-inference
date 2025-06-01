from ortools.math_opt.python import mathopt
import networkx as nx
import copy

import math
from collections import defaultdict

verbose_level = 0

G = nx.Graph()
# G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)])

# G.add_edges_from([(0, 1), (1, 2), (2, 3), (3, 4), (4, 0),
#                   (5, 6), (6, 7), (7, 8), (8, 9), (9, 5),
#                   (0, 5)])

G.add_edges_from([(0, 1), (0, 2), (0, 3), (0, 5), (0, 7), (0, 8), (0, 9), (0, 10), (0, 11), (0, 13), (0, 14), (0, 15), (0, 16), (0, 18), (0, 19), (1, 3), (1, 4), (1, 5), (1, 7), (1, 8), (1, 10), (1, 11), (1, 14), (1, 16), (1, 17), (1, 18), (2, 3), (2, 7), (2, 8), (2, 10), (2, 11), (2, 12), (2, 13), (2, 14), (3, 9), (3, 11), (3, 12), (3, 14), (3, 15), (3, 18), (3, 19), (4, 5), (4, 6), (4, 7), (4, 8), (4, 9), (4, 11), (4, 13), (4, 18), (5, 6), (5, 7), (5, 10), (5, 12), (5, 13), (5, 14), (5, 16), (5, 19), (6, 9), (6, 12), (6, 14), (6, 15), (6, 16), (6, 17), (6, 18), (6, 19), (7, 8), (7, 9), (7, 10), (7, 13), (7, 18), (8, 14), (8, 17), (8, 19), (9, 16), (9, 11), (10, 11), (10, 13), (10, 14), (11, 12), (11, 14), (11, 15), (11, 17), (11, 19), (12, 14), (12, 15), (12, 16), (13, 15), (13, 16), (14, 15), (14, 18), (15, 16), (15, 17), (16, 17), (16, 18), (16, 19), (17, 18), (17, 19), (18, 19)])

n = G.number_of_nodes()
epsilon = 1e-6


def log(name, message, indent_change=0):
    print(f"{'    ' * indent_change}[{name}] {message}")


def print_model(model, name):
    dashes = "----" * 8
    header = f"\n\t\t\t{dashes} {name} MODEL {dashes}"
    print(header)

    obj = model.objective
    sense = 'Maximize' if obj.is_maximize else 'Minimize'
    print(f"\t\t\t{sense} cᵀx + d")

    variables = list(model.variables())
    integer_vars = [v.name for v in variables if v.integer]
    continuous_vars = [v.name for v in variables if not v.integer]

    if continuous_vars:
        print("\t\t\tx ∈ ℝⁿ  where:")
        bounds = defaultdict(list)
        for v in variables:
            if not v.integer:
                bounds[(v.lower_bound, v.upper_bound)].append(v.name)
        for (lb, ub), names in bounds.items():
            name_str = ', '.join(names)
            if lb == 0 and ub == math.inf:
                print(f"\t\t\t  {name_str} ≥ 0")
            elif lb == -math.inf and ub == math.inf:
                print(f"\t\t\t  {name_str} free")
            elif lb != -math.inf and ub == math.inf:
                print(f"\t\t\t  {name_str} ≥ {lb}")
            elif lb == -math.inf and ub != math.inf:
                print(f"\t\t\t  {name_str} ≤ {ub}")
            elif lb == ub:
                print(f"\t\t\t  {name_str} = {lb}")
            else:
                print(f"\t\t\t  {lb} ≤ {name_str} ≤ {ub}")

    if integer_vars:
        print("\t\t\tx ∈ ℤⁿ  where:")
        bounds = defaultdict(list)
        for v in variables:
            if v.integer:
                bounds[(v.lower_bound, v.upper_bound)].append(v.name)
        for (lb, ub), names in bounds.items():
            name_str = ', '.join(names)
            if lb == 0 and ub == math.inf:
                print(f"\t\t\t  {name_str} ≥ 0")
            elif lb == -math.inf and ub == math.inf:
                print(f"\t\t\t  {name_str} free")
            elif lb != -math.inf and ub == math.inf:
                print(f"\t\t\t  {name_str} ≥ {lb}")
            elif lb == -math.inf and ub != math.inf:
                print(f"\t\t\t  {name_str} ≤ {ub}")
            elif lb == ub:
                print(f"\t\t\t  {name_str} = {lb}")
            else:
                print(f"\t\t\t  {lb} ≤ {name_str} ≤ {ub}")

    print("\t\t\tSubject to:")
    for constraint in model.linear_constraints():
        parts = [f"\t\t\t{lt.coefficient}*{lt.variable.name}" for lt in constraint.terms()]
        expr = ' + '.join(parts) if parts else '0'
        lb, ub = constraint.lower_bound, constraint.upper_bound
        if lb == ub:
            print(f"\t\t\t  {expr} = {lb}")
        elif lb <= -math.inf:
            print(f"\t\t\t  {expr} ≤ {ub}")
        elif ub >= math.inf:
            print(f"\t\t\t  {expr} ≥ {lb}")
        else:
            print(f"\t\t\t  {lb} ≤ {expr} ≤ {ub}")

    print('\t\t\t' + '-' * len(header) + "\n")


def solve_lp_master(patterns, branch_constraints):
    name = 'LP Master'
    if verbose_level >= 1:
        log(name, f"Start with {len(patterns)} patterns, constraints={branch_constraints}", indent_change=2)

    master = mathopt.Model()
    x = [master.add_variable(lb=0.0, name=f"x{p}") for p in range(len(patterns))]
    master.minimize(sum(x))

    vertex_constraints = []
    for v in range(n):
        c = master.add_linear_constraint(
            sum(patterns[p][v] * x[p] for p in range(len(patterns))) >= 1,
            name=f"cover_v{v}"
        )
        vertex_constraints.append(c)

    for bc in branch_constraints:
        if bc[0] == 'pattern':
            _, p, include = bc
            master.add_linear_constraint(x[p] == (1.0 if include else 0.0))
        else:
            u, v, together = bc
            if verbose_level >= 1:
                log(name, f"Apply vertex branch: ({u},{v}, together={together})", 2)
            for p, pat in enumerate(patterns):
                if together and not (pat[u] and pat[v]):
                    master.add_linear_constraint(x[p] == 0)
                if not together and pat[u] and pat[v]:
                    master.add_linear_constraint(x[p] == 0)

    if verbose_level >= 2:
        print_model(master, "LP")

    result = mathopt.solve(master, mathopt.SolverType.GLOP)
    if result.termination.reason == mathopt.TerminationReason.OPTIMAL:
        if verbose_level >= 1:
            log(name, f"Objective = {result.objective_value()}", indent_change=2)
            print("\t\t[LP Master] LP relaxation solution: "+", ".join(
                [f"{var.name} = {val}" for var, val in result.variable_values().items() if val != 0]
            ))
    else:
        if verbose_level >= 1:
            log(name, f"Terminated: {result.termination.reason}", indent_change=2)

    return result, x, vertex_constraints


def solve_pricing(duals, graph, branch_constraints, patterns):
    name = 'Pricing'
    if verbose_level >= 1:
        print()
        log(name, "Start pricing sub-problem", indent_change=2)

    modified_G = graph.copy()
    for bc in branch_constraints:
        if bc[0] != 'pattern':
            u, v, together = bc
            if not together:
                modified_G.add_edge(u, v)
                if verbose_level >= 1:
                    log(name, f"Add separation edge ({u},{v})", 2)

    model = mathopt.Model()
    y = {i: model.add_variable(lb=0.0, ub=1.0, is_integer=True, name=f"y_{i}")
         for i in modified_G.nodes()}
    model.maximize(sum(duals[i] * y[i] for i in modified_G.nodes()))

    for u, v in modified_G.edges():
        model.add_linear_constraint(y[u] + y[v] <= 1)

    for bc in branch_constraints:
        if bc[0] != 'pattern':
            u, v, together = bc
            if together:
                model.add_linear_constraint(y[u] == 1)
                model.add_linear_constraint(y[v] == 1)
                if verbose_level >= 1:
                    log(name, f"Force together ({u},{v}) = 1", 2)

    for pat in patterns:
        expr = sum((1 - pat[i]) * y[i] + pat[i] * (1 - y[i]) for i in range(n))
        model.add_linear_constraint(expr >= 1)

    if verbose_level >= 2:
        print_model(model, "MIP")

    result = mathopt.solve(model, mathopt.SolverType.HIGHS)
    if result.termination.reason != mathopt.TerminationReason.OPTIMAL:
        if verbose_level >= 1:
            log(name, f"MIP solve failed: {result.termination.reason}", indent_change=2)
        raise RuntimeError(f"Pricing MIP failed: {result.termination.reason}")

    new_set = [i for i in modified_G.nodes() if result.variable_values()[y[i]] > 0.5]
    rc = 1 - sum(duals[i] for i in new_set)
    if verbose_level >= 1:
        log(name, f"Set {new_set}, reduced cost={rc:.6f}", indent_change=2)
    return new_set, rc


def column_generation_loop(patterns, branch_constraints):
    name = 'COL-GEN'
    if verbose_level >= 1:
        log(name, f"Enter column generation", indent_change=1)
    while True:
        lp_res, x_vars, v_constraints = solve_lp_master(patterns, branch_constraints)
        if lp_res.termination.reason != mathopt.TerminationReason.OPTIMAL:
            if verbose_level >= 1:
                log(name, "LP not optimal, aborting", indent_change=1)
            return None, None, None

        duals = [lp_res.dual_values()[c] for c in v_constraints]
        new_set, rc = solve_pricing(duals, G, branch_constraints, patterns)

        if rc >= -epsilon:
            if verbose_level >= 1:
                log(name, "No negative reduced-cost column, done", indent_change=1)
            break

        patterns.append([1 if i in new_set else 0 for i in range(n)])
        if verbose_level >= 1:
            log(name, f"Added new pattern {patterns[-1]}", 1)

    return lp_res, x_vars, patterns


def branch_and_price(patterns, branch_constraints, level, best_obj=float('inf')):
    name = f'B&P {level}'
    if verbose_level >= 1:
        print("\n")
        log(name, f"Start B&P: patterns={len(patterns)}, constraints={branch_constraints}, best_obj={best_obj}", indent_change=0)

    lp_res, x_vars, patterns = column_generation_loop(patterns, branch_constraints)
    if lp_res is None:
        if verbose_level >= 1:
            log(name, "COL-GEN failed, backtracking", indent_change=-1)
        return best_obj

    bound = lp_res.objective_value()
    if verbose_level >= 1:
        log(name, f"LP relaxation bound={bound}", indent_change=0)

    if bound >= best_obj - epsilon:
        if verbose_level >= 0:
            log(name, f"Prune: bound {bound} >= best_obj {best_obj}", indent_change=-1)
        return best_obj

    x_values = [lp_res.variable_values()[x] for x in x_vars]
    fractional_indices = [i for i, v in enumerate(x_values) if abs(v - round(v)) > epsilon]

    if not fractional_indices:
        obj = sum(x_values)
        if verbose_level >= 0:
            log(name, f"Integer solution found: value={obj}", indent_change=0)
        best_obj = min(best_obj, obj)
        if verbose_level >= 1:
            log(name, f"Update best_obj to {best_obj}", indent_change=0)
        return best_obj

    p = fractional_indices[0]
    if verbose_level >= 1:
        log(name, f"Branch on pattern x{p}", indent_change=0)

    best_obj = branch_and_price(copy.deepcopy(patterns), branch_constraints + [('pattern', p, False)], level + 1, best_obj)
    best_obj = branch_and_price(copy.deepcopy(patterns), branch_constraints + [('pattern', p, True)], level + 1, best_obj)

    if verbose_level >= 1:
        log(name, "Exit B&P", indent_change=-1)
    return best_obj


if __name__ == "__main__":
    initial_patterns = [[1 if i == j else 0 for i in range(n)] for j in range(n)]
    branch_and_price(initial_patterns, [], 1)
