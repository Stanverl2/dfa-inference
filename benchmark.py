from col_solve import branch_and_price_dfa_coloring
from cp_sat import solve_dfa_sat_paper
import time
import csv
import re

from dfa import Node
from misc.graph_color import naive_solve


# STEP 1: Generate the tests with the generate_test_cases_file.
def benchmark():

    nodes_simple, edges, determinization_constraints = parse_test_file("test_cases/size=30/test_00/consistency_graph.txt")
    nodes_simple, edges, determinization_constraints = reindex_graph(nodes_simple, edges, determinization_constraints)
    nodes = parse_prefix_tree("test_cases/size=30/test_00/prefix_tree.txt")
    # nodes also show how nodes are connected in the prefix_tree. This is needed for the detrminization constraint.

    (time_spentSAT, cSAT, outputSAT) = solveSat(nodes, edges)
    print(time_spentSAT)
    (time_spentBP, cBP, outputBP) = solveBranchAndPrice(nodes, edges)
    print(time_spentBP)
    (time_spentNaive, cNaive , outputNaive) = solveNaive(nodes_simple, edges)
    print(time_spentNaive)

    results = [
        ("SAT", time_spentSAT, cSAT),
        # ("BranchAndPrice", time_spentBP, cBP),
        ("Naive", time_spentNaive, cNaive)
    ]

    write_results_to_csv("test_cases/size=30/test_00/results_summary.csv", results)

def solveSat(nodes, edges):
    start_time = time.time()
    low, high = 1, len(nodes)
    best_solution = None
    best_k = None
    while low <= high:
        mid = (low + high) // 2
        sol = solve_dfa_sat_paper(mid, nodes, edges)
        if sol is not None:
            best_k, best_solution = mid, sol
            high = mid - 1
        else:
            low = mid + 1
    return (time.time() - start_time, best_k, best_solution)

def solveBranchAndPrice(nodes, edges):
    start_time = time.time()
    (chromatic_number, output) = branch_and_price_dfa_coloring(nodes, edges)
    end_time = time.time()
    time_spent =  end_time - start_time
    return (time_spent, chromatic_number, output)

def solveNaive(nodes, edges):
    def build_graph(nodes, edges):
        graph = {node: [] for node in nodes}  # Initialize empty adjacency list for each node
        for a, b in edges:
            graph[a].append(b)
            # If the graph is undirected, also add the reverse:
            # graph[b].append(a)
        return graph

    graph = build_graph(nodes, edges)
    start_time = time.time()
    (chromatic_number, output) = naive_solve(graph)
    end_time = time.time()
    time_spent = end_time - start_time
    return (time_spent, chromatic_number, output)


def write_results_to_csv(filename, results):
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["Algorithm", "Time(s)", "ChromaticNumber"])
        for name, time_spent, chromatic in results:
            writer.writerow([name, f"{time_spent:.6f}", chromatic])

def reindex_graph(nodes, edges, implications=None):
    """
    Reindexes node IDs to start at 0, and adjusts edges and implications accordingly.
    Accepts:
      - nodes: list of node IDs
      - edges: list of (a, b) edges
      - implications (optional): list of ((a1, a2), (b1, b2)) pairs

    Returns:
      - new_nodes: list of reindexed node IDs (0..n-1)
      - new_edges: reindexed edge list
      - new_implications: reindexed implications (if provided)
    """
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(nodes))}
    new_nodes = list(range(len(nodes)))
    new_edges = [(id_map[a], id_map[b]) for (a, b) in edges]

    new_implications = None
    if implications is not None:
        new_implications = [
            ((id_map[a1], id_map[a2]), (id_map[b1], id_map[b2]))
            for ((a1, a2), (b1, b2)) in implications
        ]

    return new_nodes, new_edges, new_implications

def parse_prefix_tree(filename):
    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    num_nodes = int(lines[0])
    node_lines = lines[1:num_nodes + 1]
    num_edges = int(lines[num_nodes + 1])
    edge_lines = lines[num_nodes + 2:]

    # First pass: map old IDs to new IDs (0-based)
    original_ids = [int(line.split()[0]) for line in node_lines]
    id_map = {old_id: new_id for new_id, old_id in enumerate(sorted(original_ids))}

    # Second pass: create Node instances with new IDs
    nodes = {}
    for old_id in original_ids:
        new_id = id_map[old_id]
        nodes[old_id] = Node(new_id)

    # Third pass: assign .a and .b to actual Node objects
    for line in edge_lines:
        from_old, to_old, label = line.split()
        from_old = int(from_old)
        to_old = int(to_old)
        label = label.strip('"')

        from_node = nodes[from_old]
        to_node = nodes[to_old]

        if label == '0':
            from_node.a = to_node
        elif label == '1':
            from_node.b = to_node
        else:
            raise ValueError(f"Unexpected label: {label}")

    # Return list of Node instances in reindexed order
    node_list = [None] * len(nodes)
    for old_id, node in nodes.items():
        new_id = id_map[old_id]
        node_list[new_id] = node

    return node_list

def parse_test_file(filename):
    nodes = set()
    edges = []
    implications = []

    with open(filename, 'r') as f:
        lines = [line.strip() for line in f if line.strip()]

    idx = 0

    # Read number of nodes
    num_nodes = int(lines[idx])
    idx += 1

    # Read node lines
    for _ in range(num_nodes):
        node_id = int(lines[idx].split()[0])  # Only the number matters
        nodes.add(node_id)
        idx += 1

    # Read number of edges
    num_edges = int(lines[idx])
    idx += 1

    # Read edge lines
    for _ in range(num_edges):
        a, b = map(int, lines[idx].split()[:2])  # Only the first two parts (ignore label)
        edges.append((a, b))
        nodes.update([a, b])
        idx += 1

    # Read implication lines
    while idx < len(lines):
        line = lines[idx]
        if match := re.match(r'^\((\d+)=(\d+)\)->\((\d+)=(\d+)\)$', line):
            a1, a2, b1, b2 = map(int, match.groups())
            implications.append(((a1, a2), (b1, b2)))
            nodes.update([a1, a2, b1, b2])
        idx += 1

    return sorted(nodes), edges, implications

if __name__ == "__main__":
    benchmark()
