from ortools.sat.python import cp_model
from collections import defaultdict
from dfa import Node, render_dfa, dfa_to_list


def solve_dfa_coloring_cp_sat(C, N, edges, dfa):
    model = cp_model.CpModel()
    color = [model.new_int_var(0, C - 1, f"color_{i}") for i in range(N)]
    conflict_set = {(min(i, j), max(i, j)) for (i, j) in edges}

    for v, u in conflict_set:
        assert v != u, f"invalid edge {v} {u}"
        model.add(color[v] != color[u])

    for i in range(N):
        for j in range(i + 1, N):
            if (i, j) in conflict_set:
                continue

            node_i = dfa[i]
            node_j = dfa[j]

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


def rebuild_dfa_from_coloring(dfa, coloring):
    groups = defaultdict(list)
    for node in dfa:
        groups[coloring[node.id]].append(node)

    new_nodes = {}
    for color, nodes in groups.items():
        states = {n.state for n in nodes}
        state = Node.ACC if Node.ACC in states else Node.REJ
        new_nodes[color] = Node(id=color, state=state)

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

dfa_apta = dfa_to_list(dfa)
C = 2
N = len(dfa_apta)
edges = [(1, 4), (3, 4), (1, 2), (2, 3)]

coloring = solve_dfa_coloring_cp_sat(C, N, edges, dfa_apta)

if coloring is not None:
    print("Coloring:", coloring)
    minimized_dfa = rebuild_dfa_from_coloring(dfa_apta, coloring)
    render_dfa(minimized_dfa, path="/tmp/minimized_dfa", view=True)
else:
    print("No solution found.")