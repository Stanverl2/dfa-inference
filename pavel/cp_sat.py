from ortools.sat.python import cp_model
from collections import defaultdict
from dfa import Node, render_dfa, dfa_to_list

# Exact DFA Identification Using SAT Solvers
# https://www.cs.cmu.edu/~mheule/publications/DFA_ICGI.pdf
def solve_dfa_sat_paper(C, dfa_list, conflict_edges):
    model = cp_model.CpModel()
    N = len(dfa_list)
    color = [model.new_int_var(0, C - 1, f"color_{i}") for i in range(N)]
    conflict_set = {(min(i, j), max(i, j)) for (i, j) in conflict_edges}

    for v, u in conflict_set:
        model.add(color[v] != color[u])

    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in conflict_set:
                continue

            node_i = dfa_list[i]
            node_j = dfa_list[j]

            has_a = node_i.a is not None and node_j.a is not None
            has_b = node_i.b is not None and node_j.b is not None

            if not (has_a or has_b):
                continue

            eq_color_ij = model.new_bool_var(f'eq_color_{i}_{j}')
            model.add(color[i] == color[j]).only_enforce_if(eq_color_ij)
            model.add(color[i] != color[j]).only_enforce_if(~eq_color_ij)

            if has_a:
                a_i = node_i.a.id
                a_j = node_j.a.id
                model.add(color[a_i] == color[a_j]).only_enforce_if(eq_color_ij)
            if has_b:
                b_i = node_i.b.id
                b_j = node_j.b.id
                model.add(color[b_i] == color[b_j]).only_enforce_if(eq_color_ij)

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        return [solver.value(color[i]) for i in range(N)]
    return None


def solve_dfa_sat_edge_first(C, dfa_list, conflict_edges):
    model = cp_model.CpModel()
    N = len(dfa_list)
    conflict_set = {(min(i, j), max(i, j)) for (i, j) in conflict_edges}

    eq = {}
    for i in range(N):
        for j in range(i + 1, N):
            eq[i, j] = model.new_bool_var(f"eq_{i}_{j}")

    for i, j in conflict_set:
        model.add(eq[i, j] == 0)

    for i in range(N):
        for j in range(i + 1, N):
            for k in range(j + 1, N):
                model.add(eq[i, j] + eq[i, k] + eq[j, k] != 2)

    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in conflict_set:
                continue

            node_i = dfa_list[i]
            node_j = dfa_list[j]

            has_a = node_i.a is not None and node_j.a is not None
            has_b = node_i.b is not None and node_j.b is not None

            if not (has_a or has_b):
                continue

            eq_ij = eq[i, j]

            if has_a:
                a_i = node_i.a.id
                a_j = node_j.a.id
                a1, a2 = sorted((a_i, a_j))
                model.add_implication(eq_ij, eq[a1, a2])

            if has_b:
                b_i = node_i.b.id
                b_j = node_j.b.id
                b1, b2 = sorted((b_i, b_j))
                model.add_implication(eq_ij, eq[b1, b2])

    model.maximize(sum(eq[i, j] for i in range(N) for j in range(i + 1, N)))

    solver = cp_model.CpSolver()
    status = solver.solve(model)
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        parent = list(range(N))
        def find(x):
            while parent[x] != x:
                parent[x] = parent[parent[x]]
                x = parent[x]
            return x
        def union(x, y):
            x_root = find(x)
            y_root = find(y)
            if x_root != y_root:
                parent[y_root] = x_root

        for i in range(N):
            for j in range(i + 1, N):
                if solver.value(eq[i, j]):
                    union(i, j)

        roots = {find(i) for i in range(N)}
        root_to_color = {r: idx for idx, r in enumerate(sorted(roots))}
        return [root_to_color[find(i)] for i in range(N)]

    return None # TODO remove, it always succeed


def rebuild_dfa_from_coloring(dfa_list, coloring):
    groups = defaultdict(list)
    for node in dfa_list:
        groups[coloring[node.id]].append(node)

    new_nodes = {}
    for color, nodes in groups.items():
        has_acc = any(n.state == Node.ACC for n in nodes)
        has_rej = any(n.state == Node.REJ for n in nodes)
        assert not (has_acc and has_rej), "conflict state"
        state = Node.ACC if has_acc else Node.REJ if has_rej else None

        names = sorted(n.name for n in nodes)
        name = ",".join(names)

        new_nodes[color] = Node(id=color, state=state, name=name)

    for color, nodes in groups.items():
        sample = nodes[0]
        if sample.a:
            a_color = coloring[sample.a.id]
            new_nodes[color].a = new_nodes[a_color]
        if sample.b:
            b_color = coloring[sample.b.id]
            new_nodes[color].b = new_nodes[b_color]

    root_color = coloring[0]
    return new_nodes[root_color]


if __name__ == "__main__":
    dfa = Node(
        id=0,
        a=Node(
            id=1,
            state=Node.REJ,
            a=Node(id=2, state=Node.ACC),
            b=Node(id=3, state=Node.REJ)
        ),
        b=Node(id=4, state=Node.ACC)
    )

    render_dfa(dfa, path="/tmp/original_dfa", view=True)

    dfa_list = dfa_to_list(dfa)
    C = 2
    edges = [(1, 4), (3, 4), (1, 2), (2, 3)] # TODO generate

    coloring = solve_dfa_sat_paper(C, dfa_list, edges)

    if coloring is not None:
        print("Coloring:", coloring)
        minimized_dfa = rebuild_dfa_from_coloring(dfa_list, coloring)
        render_dfa(minimized_dfa, path="/tmp/minimized_dfa", view=True)
    else:
        print("No solution found.")
