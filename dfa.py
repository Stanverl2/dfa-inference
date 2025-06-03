import warnings
from collections import defaultdict
from graphviz import Digraph, Graph


class Node:
    """
    Represents a node in a DFA or prefix tree.

    Attributes:
        id (int): Unique identifier for the node.
        state (int | None): Accepting (1), rejecting (0), or undecided (None) state.
        a (Node | None): Transition on symbol 'a'.
        b (Node | None): Transition on symbol 'b'.
        name (str): Label for visualization (defaults to the node ID).
    """
    ACC = 1
    REJ = 0

    def __init__(self, id: int, state=None, a=None, b=None, name: str = None):
        self.id = id
        self.state = state
        self.a = a
        self.b = b
        self.name = name if name is not None else str(id)

    def __repr__(self):
        return f"Node(id={self.id}, state={self.state})"

    def accepts(self, word: str) -> bool:
        if len(word) == 0:
            return bool(self.state)

        if word[0] == 'a' and self.a is not None:
            return self.a.accepts(word[1:])
        elif word[0] == 'b' and self.b is not None:
            return self.b.accepts(word[1:])
        else:
            return False


def render_dfa(root, path="/tmp/dfa", view=True):
    """
    Renders a DFA starting from the given root node using Graphviz.

    Accepting states are drawn with double circles.
    Rejecting states are drawn with black fill and white text.
    Undecided states are drawn normally.

    :param root: The root node of the DFA.
    :param path: Output path (without file extension) for the rendered image.
    :param view: Whether to open the rendered image after generation.
    :return: A rendering of the provided DFA.
    """
    dot = Digraph(name="DFA", format="png", engine="dot",
                  graph_attr={"rankdir": "LR"},
                  node_attr={"fontname": "Monospace"},
                  edge_attr={"fontname": "Monospace"})

    visited = set()

    def add_state(node):
        if node.id in visited:
            return
        visited.add(node.id)

        node_id = str(node.id)

        if node.state is None:
            dot.node(node_id, label=node.name, shape="circle")
        elif node.state == Node.ACC:
            dot.node(
                node_id,
                label=node.name,
                shape="circle",
                peripheries="2",
                style="filled",
                fillcolor="white",
                fontcolor="black"
            )
        elif node.state == Node.REJ:
            dot.node(
                node_id,
                label=node.name,
                shape="circle",
                peripheries="2",
                style="filled",
                fillcolor="black",
                fontcolor="white"
            )
        else:
            assert False, "bad state"

        for sym in ("a", "b"):
            child = getattr(node, sym)
            if child:
                dot.edge(node_id, str(child.id), label=sym)
                add_state(child)

    add_state(root)
    dot.render(path, view=view)

def render_graph(N, edge_list, path="/tmp/graph", view=True):
    dot = Graph(name="Graph", format="png", engine="circo",
                node_attr={"fontname": "Monospace"},
                edge_attr={"fontname": "Monospace"})

    for i in range(N):
        dot.node(str(i), label=str(i), shape="circle")

    for u, v in edge_list:
        dot.edge(str(u), str(v))

    dot.render(path, view=view)

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
        a_targets = set()
        b_targets = set()
        for node in nodes:
            if node.a:
                a_targets.add(coloring[node.a.id])
            if node.b:
                b_targets.add(coloring[node.b.id])

        assert len(a_targets) <= 1, f"inconsistent a transition in group {color}"
        assert len(b_targets) <= 1, f"inconsistent b transition in group {color}"

        if a_targets:
            new_nodes[color].a = new_nodes[a_targets.pop()]
        if b_targets:
            new_nodes[color].b = new_nodes[b_targets.pop()]

    root_color = coloring[0]
    return new_nodes[root_color]

def dfa_to_list(root):
    """
    Traverses the DFA and returns a list of all nodes, sorted by ID. Ensures that node IDs are
    consecutive and start from 0.

    :param root: The root node of the DFA.
    :return: A list of Node instances sorted by ID.
    """
    all_nodes = {}

    def visit(node):
        assert node.id not in all_nodes, "dup id"
        all_nodes[node.id] = node
        if node.a:
            visit(node.a)
        if node.b:
            visit(node.b)

    visit(root)

    ids = sorted(all_nodes)
    assert ids == list(range(len(ids))), "bad id seq"

    return [all_nodes[i] for i in ids]


def _rec_conflict_check(node_a, node_b, conflict_set, visiting_set):
    if node_a is None or node_b is None:
        return False

    id1, id2 = min(node_a.id, node_b.id), max(node_a.id, node_b.id)
    key = (id1, id2)

    if key in conflict_set:
        return True

    if key in visiting_set:
        return False 

    visiting_set.add(key)

    if (node_a.state == Node.ACC and node_b.state == Node.REJ) or \
       (node_a.state == Node.REJ and node_b.state == Node.ACC):
        conflict_set.add(key)
        return True

    if _rec_conflict_check(node_a.a, node_b.a, conflict_set, visiting_set):
        conflict_set.add(key)
        return True

    if _rec_conflict_check(node_a.b, node_b.b, conflict_set, visiting_set):
        conflict_set.add(key)
        return True

    return False

def calc_inequality_edges(root):
    nodes = dfa_to_list(root)
    N = len(nodes)
 
    conflict_set = set()
    visiting_set = set()

    for i in range(N):
        for j in range(i + 1, N):
            node_i = nodes[i]
            node_j = nodes[j]
            
            _rec_conflict_check(node_i, node_j, conflict_set, visiting_set)

    return list(conflict_set)

def generate_prefix_tree(positives: set[str], negatives: set[str]) -> Node:
    """
    Generates a prefix tree generated for a set of positive examples and a set of negative examples.

    :param positives: A set of strings containing examples of words in the language.
    :param negatives: A set of strings containing examples of words not in the language.
    :return: The root node of the prefix tree.
    """
    next_node_id: int = 0

    root: Node = Node(next_node_id)
    next_node_id += 1

    next_node_id = iterate_set(root, next_node_id, positives, 1)
    _ = iterate_set(root, next_node_id, negatives, 0)

    return root


def iterate_set(root: Node, next_node_id: int, examples: set, final_state: int) -> int:
    """
    Helper method for creating prefix trees. The method iterates over a set of examples (either
    positive of negative) and iteratively builds the prefix tree. Skips any word containing symbols
    not in the alphabet {'a', 'b'}. If a word is found in both positive and negative sets, a
    RuntimeError is raised.

    :param root: The staring node of the prefix tree.
    :param next_node_id: The identifier number of the next node id added to the prefix tree.
    :param examples: A set of words represented as strings.
    :param final_state: An integer determining whether it is a set of positive (1) or negative (0) examples.
    :return: An integer that represents the identifier number of the next node_id.
    """
    alphabet = {'a', 'b'}

    for word in examples:
        if not set(word).issubset(alphabet):
            warnings.warn(f"A word ({word}) was encountered that exists of characters not inside the"
                          f"alphabet '{'a, b'}'. This word has been ignored.", RuntimeWarning)
            continue

        curr: Node = root

        for char in word:
            if char == "a":
                if curr.a is None:
                    previous: Node = curr

                    curr = Node(next_node_id)
                    next_node_id += 1

                    previous.a = curr
                else:
                    curr = curr.a
            elif char == "b":
                if curr.b is None:
                    previous: Node = curr

                    curr = Node(next_node_id)
                    next_node_id += 1

                    previous.b = curr
                else:
                    curr = curr.b
            else:
                raise ValueError(f"A word ({word}) was encountered that exists of characters not "
                                 f"inside the alphabet '{'a, b'}'. This word has been ignored.")

        if curr.state is not None and curr.state != final_state:
            raise RuntimeError(f"Both the positive and negative set of examples have the word '{word}' in it")

        curr.state = final_state

    return next_node_id
