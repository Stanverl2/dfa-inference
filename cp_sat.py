from ortools.sat.python import cp_model
from dfa import Node, render_dfa, dfa_to_list, calc_inequality_edges, rebuild_dfa_from_coloring

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

if __name__ == "__main__":
    # dfa = Node(
    #     id=0,
    #     a=Node(
    #         id=1,
    #         state=Node.REJ,
    #         a=Node(id=2, state=Node.ACC),
    #         b=Node(id=3, state=Node.REJ)
    #     ),
    #     b=Node(id=4, state=Node.ACC)
    # )
    # C = 2

    dfa = Node(
        id=0,
        a=Node(
            id=1,
            state=Node.ACC,
            b=Node(
                id=2,
                a=Node(
                    id=3,
                    a=Node(id=4, state=Node.ACC)
                ),
                b=Node(id=5, state=Node.REJ)
            )
        ),
        b=Node(
            id=6,
            state=Node.REJ,
            b=Node(id=7, state=Node.ACC)
        )
    )
    C = 3

    render_dfa(dfa, path="/tmp/original_dfa", view=True)

    dfa_list = dfa_to_list(dfa)
    edges = calc_inequality_edges(dfa)

    coloring = solve_dfa_sat_paper(C, dfa_list, edges)

    if coloring is not None:
        print("Coloring:", coloring)
        minimized_dfa = rebuild_dfa_from_coloring(dfa_list, coloring)
        render_dfa(minimized_dfa, path="/tmp/minimized_dfa", view=True)
    else:
        print("No solution found.")
