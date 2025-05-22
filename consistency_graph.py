import networkx as nx
from typing import Any, List, Union
import copy


class MergedNode:
    def __init__(self, ids: List[int], states: List[Any]):
        self.ids = sorted(set(ids))
        self.state = states  

    @property
    def name(self) -> str:
        return ",".join(str(i) for i in self.ids)

    def __repr__(self):
        return f"MergedNode(ids={self.ids}, state={self.state})"

    def copy(self):
        pass


class DirectedMultiGraph:
    def __init__(self):
        self.graph = nx.MultiDiGraph()

    def add_node(self, node_id: int, state: Any) -> MergedNode:
        node = MergedNode([node_id], [state])
        self.graph.add_node(node.name, data=node)
        return node

    def add_edge(self, from_node: MergedNode, to_node: MergedNode, label=None):
        self.graph.add_edge(from_node.name, to_node.name, label=label)

    def node_exists(self, node_id: int) -> MergedNode:
        for node_data in self.graph.nodes.values():
            merged_node = node_data['data']
            if node_id in merged_node.ids:
                return True
        return False

    def get_node(self, node_id: int) -> MergedNode:
        for node_data in self.graph.nodes.values():
            merged_node = node_data['data']
            if node_id in merged_node.ids:
                return merged_node
        raise ValueError(f"Node with ID {node_id} not found.")

    def get_node_from_name(self, node_name: str) -> MergedNode:
        target_ids = set(int(i) for i in node_name.split(",") if i.strip())
        for node_data in self.graph.nodes.values():
            merged_node = node_data['data']
            if target_ids.issubset(set(merged_node.ids)):
                return merged_node
        raise ValueError(f"No node found containing IDs: {sorted(target_ids)}")

    def get_edge_pairs(self, node: MergedNode):
        print("=========== IN E P ==========")
        edge_pairs = []

        print(f"IDs in just-now merged node: {node.ids}")
        outgoing_edges = []
        for id_ in node.ids:
            if self.node_exists(id_):
                outgoing_edges.extend(self.graph.out_edges(self.get_node(id_).name, data=True))

        for i in range(len(outgoing_edges)):
            for j in range(i + 1, len(outgoing_edges)):
                edge1, edge2 = outgoing_edges[i], outgoing_edges[j]
                if edge1[2]['label'] == edge2[2]['label']:
                    if (edge1, edge2) not in edge_pairs and edge1 != edge2:
                        edge_pairs.append((edge1, edge2))

        print(f"Edge pairs: ")
        for ep in edge_pairs:
            print(ep)
        print("=========== OUT E P ==========")
        return edge_pairs

    def merge_nodes(self, node1: MergedNode, node2: MergedNode) -> MergedNode:
        merged_ids = sorted(set(node1.ids + node2.ids))
        merged_state = node1.state + node2.state
        merged_node = MergedNode(merged_ids, merged_state)
        merged_name = merged_node.name

        self.graph.add_node(merged_name, data=merged_node)

        for pred, _, key, data in list(self.graph.in_edges(node1.name, keys=True, data=True)):
            self.graph.add_edge(pred, merged_name, **data)
        for pred, _, key, data in list(self.graph.in_edges(node2.name, keys=True, data=True)):
            self.graph.add_edge(pred, merged_name, **data)

        for _, succ, key, data in list(self.graph.out_edges(node1.name, keys=True, data=True)):
            self.graph.add_edge(merged_name, succ, **data)
        for _, succ, key, data in list(self.graph.out_edges(node2.name, keys=True, data=True)):
            self.graph.add_edge(merged_name, succ, **data)

        if node1.name != node2.name:
            self.graph.remove_node(node1.name)
        self.graph.remove_node(node2.name)

        return merged_node

    def display(self):
        print("Nodes:")
        for name, node_data in self.graph.nodes(data=True):
            node = node_data['data']
            print(f"  {node.name}: {node}")
        print("Edges:")
        for u, v, k, d in self.graph.edges(keys=True, data=True):
            print(f"  {u} -> {v} (key={k}, label={d.get('label')})")

    def load_from_file(self, filepath: str):
        with open(filepath, 'r') as f:
            lines = [line.strip() for line in f if line.strip()]

        num_nodes = int(lines[0])
        node_lines = lines[1:num_nodes + 1]
        num_edges = int(lines[num_nodes + 1])
        edge_lines = lines[num_nodes + 2:]

        id_to_node = {}

        for line in node_lines:
            parts = line.split()
            node_id = int(parts[0])
            node_type = parts[1].strip('"')
            node = self.add_node(node_id, node_type)
            id_to_node[node_id] = node

        for line in edge_lines:
            parts = line.split()
            from_id = int(parts[0])
            to_id = int(parts[1])
            label = parts[2].strip('"')
            self.add_edge(id_to_node[from_id], id_to_node[to_id], label=label)

    def try_merge_util(self, node1: MergedNode, node2: MergedNode) -> bool:
        print("v~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~v")

        self.display()
        if ("reject" in node1.state and "accept" in node2.state) or ("reject" in node2.state and "accept" in node1.state):
            print("False")
            return False

        merged = self.merge_nodes(node1, node2)

        for e1, e2 in self.get_edge_pairs(merged):
            print(e1)
            print(e2)
            tgt1 = self.get_node_from_name(e1[1])
            tgt2 = self.get_node_from_name(e2[1])
            # tgt1 = self.get_node(int(e1[1]))
            # tgt2 = self.get_node(int(e2[1]))
            if not self.try_merge_util(tgt1, tgt2):
                return False

        return True

    def try_merge(self, node1: MergedNode, node2: MergedNode) -> bool:
        print("a")
        copy_self = copy.deepcopy(self)
        copy_n1 = copy.deepcopy(node1)
        copy_n2 = copy.deepcopy(node2)
        print("b")
        ret = copy_self.try_merge_util(copy_n1, copy_n2)
        return ret

    def get_inequality_constraints(self, filepath: str):
        incompat_graph = nx.Graph()
        nodes = list(self.graph.nodes(data=True))

        for name, data in nodes:
            incompat_graph.add_node(name)

        for i in range(len(nodes)):
            for j in range(i + 1, len(nodes)):
                node1 = nodes[i][1]['data']
                node2 = nodes[j][1]['data']
                if not self.try_merge(node1, node2):
                    incompat_graph.add_edge(node1.name, node2.name)

        with open(filepath, 'w') as f:
            f.write(f"{len(incompat_graph.nodes)}\n")
            for node in incompat_graph.nodes:
                f.write(f"{node}\n")
            f.write(f"{len(incompat_graph.edges)}\n")
            for u, v in incompat_graph.edges:
                f.write(f"{u} {v}\n")

    def get_equality_constraints(self, filepath: str):
        all_edges = list(self.graph.edges(keys=True, data=True))
        with open(filepath, 'a') as f:
            for i in range(len(all_edges)):
                u1, v1, _, d1 = all_edges[i]
                for j in range(i + 1, len(all_edges)):
                    u2, v2, _, d2 = all_edges[j]
                    if d1.get('label') == d2.get('label'):
                        f.write(f"({u1}={v1})->({u2}={v2})\n")

    def get_constraints(self, filepath: str):
        self.get_inequality_constraints(filepath)
        self.get_equality_constraints(filepath)


if __name__ == "__main__":
    G = DirectedMultiGraph()
    G.load_from_file("prefix_tree_paper.txt")

    G.get_constraints("consistency_graph_paper.txt")
