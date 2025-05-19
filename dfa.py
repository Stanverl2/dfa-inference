import warnings

from graphviz import Digraph


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
                dot.edge(node_id, str(child.id), label=sym, weight="2")
                add_state(child)

    add_state(root)
    dot.render(path, view=view)


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
