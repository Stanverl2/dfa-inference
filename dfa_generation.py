import random
import numpy as np

from dfa import Node, render_dfa


# Step 2 of the algorithm , creating geometric distribution.
def geometric(mean):
    p = 1 / (mean + 1)
    return np.random.geometric(p)


def add_edge(fr: Node, to: Node, char):
    match char:
        case 'a' if fr.a is None:
            fr.a = to
        case 'b' if fr.b is None:
            fr.b = to


def add_undefined_edge(fr: Node, to: Node):
    if fr.a is None and fr.b is None:
        if random.random() < 0.5:
            fr.a = to
        else:
            fr.b = to
    elif fr.b is None:
        fr.b = to
    elif fr.a is None:
        fr.a = to


def is_fully_saturated(node: Node):
    return len(unused_symbols(node)) == 0


def unused_symbols(node: Node):
    res = []
    if node.a is None:
        res.append('a')
    if node.b is None:
        res.append('b')
    return res


def get_neighbours(node: Node):
    res = []
    if node.a is not None:
        res.append(node.a)
    if node.b is not None:
        res.append(node.b)
    return res


def get_incoming(target: Node, nodes):
    return [n for n in nodes if n.a == target or n.b == target]


# input parameters:
# n - the number of vertices
# a: alphabet (a, b in our case) - not used, but if we change the Node structure then we can use it.
# forward-burning probability f = 0.31
# backward-burning ratio b =  0.385
# s: self loop probability. s = 0.2
# SOURCES: https://www.cs.le.ac.uk/people/nw91/Files/stamina_journal.pdf, https://dl.acm.org/doi/10.1145/1217299.1217301
# returns Nodes. Nodes id = 0 is the start node.
def generate_dfa(n: int, a=None, f=0.31, b=0.385, s=0.2) -> list[Node]:
    nodes = []
    globally_visited = set()

    def append_node(node_v: Node, locally_visited: set):
        # Step 1: Choose random ambassador and create an edge.
        if node_v not in nodes:
            nodes.append(node_v)
        locally_visited.add(node_v)

        candidates = [node for node in nodes if node.id != node_v.id and node not in locally_visited]
        if not candidates:
            return

        ambassador = random.choice(candidates)
        locally_visited.add(ambassador)
        available_symbols = unused_symbols(ambassador)
        if not available_symbols:
            locally_visited.add(ambassador)
            append_node(node_v, locally_visited)
            globally_visited.add(ambassador)
            # Everytime a new state is considered, there should be something pointing towards it. If not, try again.
            return

        chosen_symbol = random.choice(available_symbols)
        add_edge(ambassador, node_v, chosen_symbol)

        # Add self loops
        if random.random() < s:
            available_symbols = unused_symbols(ambassador)
            if available_symbols:
                chosen_symbol = random.choice(available_symbols)
                add_edge(ambassador, ambassador, chosen_symbol)
            available_symbols = unused_symbols(ambassador)
            if len(available_symbols) == 0:
                globally_visited.add(ambassador)

        # Step 2: generate x,y
        x = geometric(f / (1 - f))
        y = geometric(f*b / (1 - f*b))

        # Step 3
        visited = locally_visited | globally_visited
        unvisited_out = [n for n in get_neighbours(ambassador) if n not in visited]
        unvisited_in = [n for n in get_incoming(ambassador, nodes) if n not in visited]

        out_neigh = random.sample(unvisited_out, min(x, len(unvisited_out)))
        in_neigh = random.sample(unvisited_in, min(y, len(unvisited_in)))
        all_neigh = out_neigh + in_neigh

        for neigh in all_neigh:
            locally_visited.add(neigh)          # local nodes cannot be revisited
            globally_visited.add(ambassador)     # add to global, cause otherwise we have separate components
            add_undefined_edge(node_v, neigh)
            append_node(neigh, locally_visited)

    # We start building our DFA from here. Start node is i=0.
    for i in range(n):
        new_node = Node(i, state=random.random() < 0.5)
        append_node(new_node, set())

    # Fill in missing edges.
    for node in nodes:
        unused = unused_symbols(node)
        for sym in unused:
            target = random.choice(nodes)
            add_edge(node, target, sym)

    return nodes
