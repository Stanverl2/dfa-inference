from col_solve import branch_and_price_dfa_coloring
from cp_sat import solve_dfa_sat_paper
import time
import csv
import re

from dfa import Node, rebuild_dfa_from_coloring, render_dfa
from misc.graph_color import naive_solve
import os


# This line below is to set up proper rendering for debugging. At least for MacOS. Then u can include redner_DFA stuff.
os.environ["PATH"] += os.pathsep + "/usr/local/bin"

def benchmark():
    """
    The outer for loop goes over folders with different sizes.
    """
    summary_results = []
    for test_size in range(1):
        size = 1  # You can change the size to correspond to the correct folder.
        for i in range(4):  # You can change the range to run all examples in the folder.
            test_name = f"test_cases_4/size={size:02}/test_{i:02}" # change the folder to run the correct one.
            print(f"running test /size={size:02}/test_{i:02}")
            nodes_simple, edges, determinization_constraints = parse_test_file(test_name + "/consistency_graph.txt")
            nodes_simple, edges, determinization_constraints = reindex_graph(nodes_simple, edges, determinization_constraints)
            nodes = parse_prefix_tree(test_name+"/prefix_tree.txt")
            # nodes also show how nodes are connected in the prefix_tree. This is needed for the detrminization constraint.
            # #
            (time_spentSAT, cSAT, outputSAT) = solveSat(nodes, edges)
            print(f"Sat Time: {time_spentSAT}")
            # Commands below can be used for debugging and checking correctness.
            # # minimized_dfa = rebuild_dfa_from_coloring(nodes, outputSAT)
            # # render_dfa(minimized_dfa, path=test_name + "/rendered_dfa_SAT", view=True)

            (time_spentBP, cBP, outputBP) = solveBranchAndPrice(nodes, edges)
            print(f"BP time: {time_spentBP}")

            #
            (time_spentNaive, cNaive , outputNaive) = solveNaive(nodes_simple, edges)
            print(f"Naive Time: {time_spentNaive}")
            # # minimized_dfa = rebuild_dfa_from_coloring(nodes, outputNaive)
            # # render_dfa(minimized_dfa, path=test_name+"/rendered_dfa_NAIVE", view=True)

            results = [
                ("SAT", time_spentSAT, cSAT),
                ("BranchAndPrice", time_spentBP, cBP),
                ("Naive", time_spentNaive, cNaive)
            ]

            for algo, t, c in results:
                summary_results.append((test_name, algo, t, c))

            write_results_to_csv(test_name + "/results_summary.csv", results)

    write_results_to_csv( "test_cases_3/all_results_summary.csv", summary_results, header=True)
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
        return graph

    graph = build_graph(nodes, edges)
    start_time = time.time()
    (chromatic_number, output) = naive_solve(graph)
    end_time = time.time()
    time_spent = end_time - start_time
    return (time_spent, chromatic_number, output)


def write_results_to_csv(filename, results, header=False):
    with open(filename, 'w', newline='') as f:
        writer = csv.writer(f)
        if header:
            writer.writerow(["Test", "Algorithm", "Time(s)", "Chromatic Number"])
        writer.writerows(results)
def reindex_graph(nodes, edges, implications=None):
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

    id_map = {}  # To reindex node IDs to start from 0
    nodes_temp = {}
    next_id = 0

    # Step 1: Parse nodes and assign states
    for line in node_lines:
        node_id_str, label = line.split()
        original_id = int(node_id_str)

        if original_id not in id_map:
            id_map[original_id] = next_id
            next_id += 1

        label = label.strip('"').lower()
        if label == 'accept':
            state = Node.ACC
        elif label == 'reject':
            state = Node.REJ
        else:
            state = None

        new_id = id_map[original_id]
        nodes_temp[original_id] = Node(id=new_id, state=state, name=f"{label}_{new_id}")

    # Step 2: Parse edges and assign transitions
    for line in edge_lines:
        from_node_str, to_node_str, label = line.split()
        from_id = int(from_node_str)
        to_id = int(to_node_str)
        symbol = label.strip('"')

        from_node = nodes_temp[from_id]
        to_node = nodes_temp[to_id]

        if symbol == '0':
            from_node.a = to_node
        elif symbol == '1':
            from_node.b = to_node
        else:
            raise ValueError(f"Unexpected transition symbol: {symbol}")

    # Step 3: Return reindexed nodes in sorted order
    reindexed_nodes = sorted(nodes_temp.values(), key=lambda n: n.id)
    return reindexed_nodes

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
