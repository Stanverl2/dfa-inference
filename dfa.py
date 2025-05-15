from graphviz import Digraph

class Node:
    ACC = 1
    REJ = 0

    def __init__(self, id, state=None, a=None, b=None, name=None):
        self.id = id
        self.state = state
        self.a = a
        self.b = b
        self.name = name if name is not None else str(id)

    def __repr__(self):
        return f"Node(id={self.id}, state={self.state})"


def render_dfa(root, path="/tmp/dfa", view=True):
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
            dot.node(node_id, label=node.name, shape="circle", peripheries="2", style="filled", fillcolor="white", fontcolor="black")
        elif node.state == Node.REJ:
            dot.node(node_id, label=node.name, shape="circle", peripheries="2", style="filled", fillcolor="black", fontcolor="white")
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