from ortools.math_opt.python import mathopt
from dfa import Node, render_dfa, dfa_to_list, calc_inequality_edges, rebuild_dfa_from_coloring
import math, collections

def apply_branch_constraints(num_nodes, base_conflicts, branch_constraints):
    parent = list(range(num_nodes))

    def find(x):
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[ry] = rx

    for op, u, v in branch_constraints:
        if op == 'SAME':
            union(u, v)

    root_map = {i: find(i) for i in range(num_nodes)}
    super_roots = sorted(set(root_map.values()))
    root_to_index = {r: idx for idx, r in enumerate(super_roots)}
    num_super = len(super_roots)

    # Check for contradictory DIFFER after SAME merging
    for op, u, v in branch_constraints:
        if op == 'DIFFER' and find(u) == find(v):
            return None, None, None
        
    for u, v in base_conflicts:
        if find(u) == find(v):
            return None, None, None

    adj = [set() for _ in range(num_super)]
    for u, v in base_conflicts:
        ru, rv = root_to_index[root_map[u]], root_to_index[root_map[v]]
        if ru != rv:
            adj[ru].add(rv)
            adj[rv].add(ru)

    for op, u, v in branch_constraints:
        if op == 'DIFFER':
            ru, rv = root_to_index[root_map[u]], root_to_index[root_map[v]]
            if ru != rv:
                adj[ru].add(rv)
                adj[rv].add(ru)

    super_edges = []
    for i in range(num_super):
        for j in adj[i]:
            if i < j:
                super_edges.append((i, j))

    node_to_super = {i: root_to_index[root_map[i]] for i in range(num_nodes)}
    return num_super, super_edges, node_to_super

def solve_lp_column_generation(num_nodes, base_conflicts, branch_constraints=None):
    if branch_constraints is None:
        branch_constraints = []
    res = apply_branch_constraints(num_nodes, base_conflicts, branch_constraints)
    if res[0] is None:
        return None, None, float('inf'), None
    num_super, super_edges, node_to_super = res
    super_nodes = list(range(num_super))
    columns = [{i} for i in super_nodes]
    eps = 1e-6

    while True:
        master = mathopt.Model()
        x = [master.add_variable(lb=0.0, name=f"x_{idx}") for idx in range(len(columns))]
        constrs = []
        for i in super_nodes:
            expr = sum((1 if i in col else 0) * x[idx] for idx, col in enumerate(columns))
            constrs.append(master.add_linear_constraint(expr >= 1))
        master.minimize(sum(x))
        lp = mathopt.solve(master, mathopt.SolverType.GLOP)
        assert lp.termination.reason == mathopt.TerminationReason.OPTIMAL
        duals = [lp.dual_values()[c] for c in constrs]

        sub = mathopt.Model()
        z = {i: sub.add_integer_variable(lb=0, ub=1, name=f"z_{i}") for i in super_nodes}
        for u, v in super_edges:
            sub.add_linear_constraint(z[u] + z[v] <= 1)
        sub.maximize(sum(duals[i] * z[i] for i in super_nodes))
        sp = mathopt.solve(sub, mathopt.SolverType.CP_SAT)
        assert sp.termination.reason == mathopt.TerminationReason.OPTIMAL

        if sp.objective_value() <= 1 + eps:
            break

        new_set = {i for i in super_nodes if int(sp.variable_values()[z[i]]) == 1}
        columns.append(new_set)

    lp_solution = {idx: lp.variable_values()[x[idx]] for idx in range(len(columns))}
    lp_bound = lp.objective_value()
    return lp_solution, columns, lp_bound, node_to_super

def is_integer_solution(lp_solution):
    return all(abs(val - round(val)) < 1e-6 for val in lp_solution.values())

def pick_fractional_pair(lp_solution, columns):
    for a_idx, set_a in enumerate(columns):
        v_a = lp_solution[a_idx]
        if v_a < 1e-6 or abs(v_a - 1) < 1e-6:
            continue
        for b_idx, set_b in enumerate(columns):
            if b_idx <= a_idx:
                continue
            v_b = lp_solution[b_idx]
            if v_b < 1e-6 or abs(v_b - 1) < 1e-6:
                continue
            inter = set_a & set_b
            diff = set_a - set_b
            if inter and diff:
                u = next(iter(inter))
                v = next(iter(diff))
                return u, v
    raise RuntimeError("No fractional pair found")

def extract_coloring_from_columns(columns, lp_solution):
    coloring = {}
    color_id = 0
    for idx, col in enumerate(columns):
        if round(lp_solution[idx]) == 1:
            for node in col:
                if node not in coloring:
                    coloring[node] = color_id
            color_id += 1
    return coloring

def propagate_same_constraints(u_id, v_id, dfa_list):
    if u_id == v_id:
        return []

    queue = collections.deque([(u_id, v_id)])
    processed_pairs = set()
    all_same_constraints = []

    while queue:
        n1_id, n2_id = queue.popleft()

        # Create a canonical, sorted tuple for checking if the pair was processed
        canonical_pair = tuple(sorted((n1_id, n2_id)))

        if canonical_pair in processed_pairs:
            continue

        processed_pairs.add(canonical_pair)
        all_same_constraints.append(('SAME', n1_id, n2_id))

        n1 = dfa_list[n1_id]
        n2 = dfa_list[n2_id]

        if n1.a and n2.a:
            queue.append((n1.a.id, n2.a.id))

        if n1.b and n2.b:
            queue.append((n1.b.id, n2.b.id))

    return all_same_constraints

def branch_and_price_dfa_coloring(dfa_list, conflict_edges):
    N = len(dfa_list)
    base_conflicts = [tuple(sorted((u, v))) for (u, v) in conflict_edges]

    best_value = float('inf')
    best_coloring = None
    stack = [[]]

    def check_transitions(coloring):
        violations = []
        for i in range(N):
            for j in range(i + 1, N):
                if coloring[i] != coloring[j]:
                    continue
                ni, nj = dfa_list[i], dfa_list[j]
                if ni.a is not None and nj.a is not None:
                    ai, aj = ni.a.id, nj.a.id
                    if coloring[ai] != coloring[aj]:
                        violations.append((i, j))
                        continue
                if ni.b is not None and nj.b is not None:
                    bi, bj = ni.b.id, nj.b.id
                    if coloring[bi] != coloring[bj]:
                        violations.append((i, j))
                        continue
        return violations

    while stack:
        branch_constraints = stack.pop()
        lp_solution, columns, lp_bound, node_to_super = solve_lp_column_generation(
            N, base_conflicts, branch_constraints
        )
        if lp_solution is None:
            continue
        pruning_bound = math.ceil(lp_bound - 1e-6)
        if pruning_bound >= best_value - 1e-6:
            continue

        # print(best_value, lp_bound, len(columns), len(branch_constraints))

        if is_integer_solution(lp_solution):
            super_col = extract_coloring_from_columns(columns, lp_solution)
            orig_col = {v: super_col[node_to_super[v]] for v in range(N)}

            violations = check_transitions(orig_col)

            if not violations:
                num_colors = max(orig_col.values()) + 1
                # print("found", num_colors)
                if num_colors < best_value:
                    best_value = num_colors
                    best_coloring = orig_col.copy()
                continue

            i, j = violations[0]
            # print("viol", i, j)
            # Branch DIFFER
            stack.append(branch_constraints + [('DIFFER', i, j)])
            # Branch SAME with propagation to children
            same_constraints_to_add = propagate_same_constraints(i, j, dfa_list)
            same_branch = branch_constraints + same_constraints_to_add
            stack.append(same_branch)
        else:
            u_super, v_super = pick_fractional_pair(lp_solution, columns)

            # 2. Now, apply the "reverse trick": find original nodes that
            #    map to these super-nodes. We only need one for each.
            u_orig = next(orig_node for orig_node, sup_node in node_to_super.items() if sup_node == u_super)
            v_orig = next(orig_node for orig_node, sup_node in node_to_super.items() if sup_node == v_super)

            # 3. Create branches using the correctly identified ORIGINAL node IDs.
            # Branch DIFFER
            stack.append(branch_constraints + [('DIFFER', u_orig, v_orig)])
            
            # Branch SAME (with propagation)
            same_constraints_to_add = propagate_same_constraints(u_orig, v_orig, dfa_list)
            same_branch = branch_constraints + same_constraints_to_add
            stack.append(same_branch)

    return best_value, best_coloring

if __name__ == "__main__":
    # from benchmark import parse_prefix_tree
    # dfa_l = parse_prefix_tree("./test_cases_3/size=10/test_00/prefix_tree.txt")
    # dfa = dfa_l[0]
    
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
    
    render_dfa(dfa, path="/tmp/original_dfa", view=True)
    dfa_list = dfa_to_list(dfa)
    edges = calc_inequality_edges(dfa)
    chromatic_bp, coloring_bp = branch_and_price_dfa_coloring(dfa_list, edges)
    print("B&P coloring uses", chromatic_bp, "colors:", coloring_bp)
    if coloring_bp is not None:
        minimized_dfa = rebuild_dfa_from_coloring(dfa_list, coloring_bp)
        render_dfa(minimized_dfa, path="/tmp/minimized_dfa", view=True)
    else:
        print("No B&P solution found.")
