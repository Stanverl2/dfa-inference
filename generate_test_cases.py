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


def main():
    os.makedirs("test_cases", exist_ok=True)

    size = 17  # 10, 15, 20, 30, etc. (Roughly the amount od nodes)

    if size == 0:
        pass
    elif size == 10:
        n = 3
        word_len = 4
        num_words = 15
    elif size == 15:
        n = 4
        word_len = 6
        num_words = 20
    elif size == 17:
        n = 7
        word_len = 9
        num_words = 20
    elif size == 20:
        n = 10
        word_len = 12
        num_words = 25

    else:
        raise ValueError(f"Unsupported size: {size}")

    test_id = 0
    max_tests = 4
    s_values = [0.2, 0.2, 0.8, 0.8]
    f_values = [0.31, 0.69, 0.31, 0.69]

    while test_id < max_tests:
        dfa_nodes = generate_dfa(n=n, s=s_values[test_id], f=f_values[test_id])
        dfa_root = dfa_nodes[0]
        words = set()
        while len(words) < num_words:
            words.add(generate_random_word(length=word_len))

        print("We added words")

        accepted = {w for w in words if dfa_root.accepts(w)}
        rejected = words - accepted

        if not accepted or not rejected:
            continue

        prefix_tree_root = generate_prefix_tree(accepted, rejected)

        case_dir = Path(f"test_cases_3/size={size}/test_{test_id:02d}")
        case_dir.mkdir(parents=True, exist_ok=True)

        with open(case_dir / "words.txt", "w") as f:
            for w in words:
                f.write(w + "\n")

        prefix_tree_file = case_dir / "prefix_tree.txt"
        save_prefix_tree(prefix_tree_root, str(prefix_tree_file))

        print("We added prefix tree")

        get_consistency_graph(
            prefix_tree_path=str(prefix_tree_file),
            consistency_graph_path=str(case_dir / "consistency_graph.txt"),
            depth=4000
        )
        print(f"Test case {test_id} completed with size={size}, n={n}, word_len={word_len}")
        test_id += 1


if __name__ == "__main__":
    main()
