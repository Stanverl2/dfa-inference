from graphviz import Digraph

class Node:
    ACC = 1
    REJ = 0

    def __init__(self, id, state=None, a=None, b=None):
        self.id = id
        self.state = state
        self.a = a
        self.b = b

    def __repr__(self):
        return f"Node(id={self.id}, state={self.state})"


def render_dfa(root, path="/tmp/dfa", view=True):
    dot = Digraph(name="DFA", format="png", engine="dot",
                  graph_attr={"rankdir": "LR"},
                  node_attr={"fontname": "Monospace", "fixedsize": "true"},
                  edge_attr={"fontname": "Monospace"})

    visited = set()

    def add_state(node):
        if node.id in visited:
            return
        visited.add(node.id)

        id_str = str(node.id)
        if node.state is None:
            dot.node(id_str, shape="circle")
        elif node.state == Node.ACC:
            dot.node(id_str, shape="circle", peripheries="2", style="filled", fillcolor="white", fontcolor="black")
        else:
            dot.node(id_str, shape="circle", peripheries="2", style="filled", fillcolor="black", fontcolor="white")

        for sym in ("a", "b"):
            child = getattr(node, sym)
            if child:
                dot.edge(id_str, str(child.id), label=sym, weight="2")
                add_state(child)

    add_state(root)
    dot.render(path, view=view)


def dfa_to_list(root):
    all_nodes = {}

    def visit(node):
        if node.id in all_nodes:
            raise ValueError(f"Duplicate node ID: {node.id}")
        all_nodes[node.id] = node
        if node.a:
            visit(node.a)
        if node.b:
            visit(node.b)

    visit(root)

    ids = sorted(all_nodes)
    if ids != list(range(len(ids))):
        raise ValueError(f"IDs must be unique and continuous from 0 to N-1. Found: {ids}")

    return [all_nodes[i] for i in ids]