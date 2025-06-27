import os
import numpy as np
from pathlib import Path

from create_consistency_graph import get_consistency_graph
from dfa import Node, generate_prefix_tree
from dfa_generation import generate_dfa


def generate_random_word(length=5, alphabet=None) -> str:
    if alphabet is None:
        alphabet = ['a', 'b']
    return ''.join(np.array(alphabet)[np.random.randint(0, len(alphabet), size=length)])


def generate_random_word_of_varying_length(max_length=5, alphabet=None) -> str:
    if alphabet is None:
        alphabet = ['a', 'b']

    # Bias toward longer words using a truncated geometric-like distribution
    lengths = np.arange(1, max_length + 1)
    probs = np.linspace(1, max_length, max_length)
    probs = probs / probs.sum()  # Normalize to get probabilities
    # Increase weight for longer lengths
    probs = probs ** 2
    probs /= probs.sum()

    actual_length = np.random.choice(lengths, p=probs)
    return ''.join(np.random.choice(alphabet, size=actual_length))


def save_prefix_tree(root: Node, filename: str):
    visited = {}
    edges = []

    def traverse(node: Node):
        if node.id in visited:
            return
        visited[node.id] = node

        if node.a:
            edges.append((node.id, node.a.id, "0"))
            traverse(node.a)
        if node.b:
            edges.append((node.id, node.b.id, "1"))
            traverse(node.b)

    traverse(root)

    with open(filename, 'w') as f:
        f.write(f"{len(visited)}\n")
        for node_id in sorted(visited):
            node = visited[node_id]
            label = "accept" if node.state == Node.ACC else "reject" if node.state == Node.REJ else "other"
            f.write(f"{node_id + 1} \"{label}\"\n")
        f.write(f"{len(edges)}\n")
        for from_id, to_id, symbol in edges:
            f.write(f"{from_id + 1} {to_id + 1} \"{symbol}\"\n")

# Test cases from the report are in test_cases_3 folder
def main():
    os.makedirs("test_cases", exist_ok=True)

    test_case = 1  # Here we just have sample test cases. Feel free to add your own with your own parameters.

    if test_case == 0:
        n = 3
        word_len = 5
        num_words = 15
    elif test_case == 1:
        n = 2
        word_len = 3
        num_words = 8
    elif test_case == 2:
        n = 3
        word_len = 4
        num_words = 15
    elif test_case == 3:
        n = 3
        word_len = 4
        num_words = 15


    else:
        raise ValueError(f"Unsupported size: {test_case}")

    test_id = 0
    max_tests = 5
    # test cases:
    # high f, low b, low s = deep trees
    # low f, high b, low s  = dense graph
    # low f, high b, high s  = dense and cyclic graph
    # high f, low b, and hig s = deep structure but a lot of self loops
    # low f, low b, low s = sparse, not a lot of transitions.
    f_values = [0.69, 0.31, 0.31, 0.69, 0.31]
    b_values = [0.385, 0.615, 0.615, 0.385, 0.385]
    s_values = [0.2, 0.2, 0.8, 0.8, 0.2]

    while test_id < max_tests:
        dfa_nodes = generate_dfa(n=n, s=s_values[test_id], f=f_values[test_id], b=b_values[test_id])
        dfa_root = dfa_nodes[0]
        words = set()
        while len(words) < num_words:
            words.add(generate_random_word(length=word_len))

        accepted = {w for w in words if dfa_root.accepts(w)}
        rejected = words - accepted

        if not accepted or not rejected:
            continue

        prefix_tree_root = generate_prefix_tree(accepted, rejected)

        case_dir = Path(f"test_cases_4/size={test_case:02d}/test_{test_id:02d}")
        case_dir.mkdir(parents=True, exist_ok=True)

        with open(case_dir / "words.txt", "w") as f:
            for w in words:
                f.write(w + "\n")

        prefix_tree_file = case_dir / "prefix_tree.txt"
        save_prefix_tree(prefix_tree_root, str(prefix_tree_file))


        get_consistency_graph(
            prefix_tree_path=str(prefix_tree_file),
            consistency_graph_path=str(case_dir / "consistency_graph.txt"),
            depth=4000
        )
        print(f"Test case {test_id} completed with size={test_case}, n={n}, word_len={word_len}")
        test_id += 1


if __name__ == "__main__":
    main()
