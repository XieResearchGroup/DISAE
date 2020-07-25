from rdkit.Chem import MolFromSmiles
from fingerprint.features import atom_features, bond_features
import warnings
import collections
from copy import copy, deepcopy
import json

degrees = [0, 1, 2, 3, 4, 5]

def node_id(smiles, idx):
    return "/".join([smiles, str(idx)])

def load_from_smiles(smiles):
    """ Load a single molecule graph from its SMIELS string. """
    graph = Molecule()
    mol = MolFromSmiles(smiles)
    if not mol:
        raise ValueError("Could not parse SMILES string:", smiles)
    for atom in mol.GetAtoms():
        atom_node = Node('atom', node_id(smiles, atom.GetIdx()), atom_features(atom))
        graph.add_node(atom_node)

    for bond in mol.GetBonds():
        src_node = graph.get_node('atom', node_id(smiles, bond.GetBeginAtom().GetIdx()))
        tgt_node = graph.get_node('atom', node_id(smiles, bond.GetEndAtom().GetIdx()))
        bond_node = Node('bond', node_id(smiles, bond.GetIdx()), bond_features(bond))
        graph.add_node(bond_node)
        bond_node.add_neighbors([src_node, tgt_node])
        src_node.add_neighbors([bond_node, tgt_node])
        tgt_node.add_neighbors([bond_node, src_node])

    mol_node = Node('molecule', smiles)
    graph.add_node(mol_node)
    atom_nodes = graph.get_node_list('atom')
    mol_node.add_neighbors(atom_nodes)

    graph.sort_by_degree('atom')

    return graph

def load_from_mol(mol):
    """ Load a single molecule graph from its RDKit mol object. """
    graph = Molecule()
    smiles=mol[0]
    mol=mol[1]
    for atom in mol.GetAtoms():
        atom_node = Node('atom', node_id(smiles, atom.GetIdx()), atom_features(atom))
        graph.add_node(atom_node)

    for bond in mol.GetBonds():
        src_node = graph.get_node('atom', node_id(smiles, bond.GetBeginAtom().GetIdx()))
        tgt_node = graph.get_node('atom', node_id(smiles, bond.GetEndAtom().GetIdx()))
        bond_node = Node('bond', node_id(smiles, bond.GetIdx()), bond_features(bond))
        graph.add_node(bond_node)
        bond_node.add_neighbors([src_node, tgt_node])
        src_node.add_neighbors([bond_node, tgt_node])
        tgt_node.add_neighbors([bond_node, src_node])

    mol_node = Node('molecule', smiles)
    graph.add_node(mol_node)
    atom_nodes = graph.get_node_list('atom')
    mol_node.add_neighbors(atom_nodes)

    graph.sort_by_degree('atom')

    return graph
def load_from_smiles_tuple(smiles_tuple):
    """ Load a composite graph, each subgraph is a molecule. """
    graph_list = [load_from_smiles(s) for s in smiles_tuple]
    big_graph = Molecule()
    for idx, subgraph in enumerate(graph_list):
        big_graph.add_subgraph(subgraph, str(idx))

    # Sort by degree
    # So that the order of getting nodes by increasing degree
    # is the same as the order of getting node list
    big_graph.sort_by_degree('atom')

    return big_graph

def load_from_mol_tuple(mol_tuple):
    """ Load a composite graph, each subgraph is a molecule. """
    graph_list = [load_from_mol(m) for m in mol_tuple]
    big_graph = Molecule()
    for idx, subgraph in enumerate(graph_list):
        big_graph.add_subgraph(subgraph, str(idx))

    # Sort by degree
    # So that the order of getting nodes by increasing degree
    # is the same as the order of getting node list
    big_graph.sort_by_degree('atom')

    return big_graph
def _check_type(ntype, store):
    if ntype not in store:
        raise KeyError("Type {0} does not exist.".format(ntype))

class Graph(object):
    """ Base class represent graph structure.
    Attributes:
        _node_index (dict of dict of Node): Graph nodes index.
            The first level key is node type, and the second level key
            is node external id.
        _type_info (dict of dict): Other information of node types.
    """

    def __init__(self):
        self._node_index = dict()
        self._type_list = dict()

    def get_types(self):
        return list(self._node_index.keys())

    def add_node(self, node):
        """ Add a new node to the graph if not already existed. """
        index = self._node_index.setdefault(node.ntype, dict())
        if node.ext_id not in index:
            index.setdefault(node.ext_id, node)
            self._type_list.setdefault(node.ntype, list()).append(node)

    def get_node(self, ntype, nid):
        """ Get node by type and external id. """
        _check_type(ntype, self._node_index)
        if nid not in self._node_index[ntype]:
            raise KeyError("Node with id {0} does not exist.".format(nid))
        return self._node_index[ntype][nid]

    def has_node(self, ntype, nid):
        """ Check if node exists in the graph. """
        return ntype in self._node_index and nid in self._node_index[ntype]

    def add_node_list(self, ntype, node_list):
        for node in node_list:
            self.add_node(node)

    def get_node_list(self, ntype):
        _check_type(ntype, self._type_list)
        return self._type_list[ntype]

class Node(object):
    """ Class represent graph node.
    Args:
        ntype (string): Node type
        ext_id (string): External identifier
        data: Node payload (default None)
    """

    def __init__(self, ntype, ext_id, data=None):
        self.ntype = ntype
        self.ext_id = ext_id
        self.data = data
        self.neighbors = set()
        self.custom_neighbors = set()

    def __str__(self):
        return ":".join([self.ext_id, self.ntype])

    def __lt__(self, other):
        return self.ntype < other.ntype or (self.ntype == other.ntype and self.ext_id < other.ext_id)

    def set_data(self, data):
        self.data = data

    def _add_neighbor(self, neighbors, new_neighbors):
        """ Add neighbor(s) for the node.
        Args:
            neighbors (Node or an iterable of Node): Old neighbor(s).
            new_neighbors (Node or an iterable of Node): Neighbor(s) to add.
            undirected (bool): If the edge is undirected (default False).
        """
        if isinstance(new_neighbors, Node):
            new_neighbors = [new_neighbors]
        if isinstance(new_neighbors, collections.Iterable) and \
                all([isinstance(node, Node) for node in new_neighbors]):
            neighbors.update(new_neighbors)
        else:
            raise ValueError("`neighbors` has to be either a Node object \
                    or an iterable of Node objects!")

    def add_neighbors(self, new_neighbors):
        self._add_neighbor(self.neighbors, new_neighbors)

    def add_custom_neighbors(self, new_neighbors):
        self._add_neighbor(self.custom_neighbors, new_neighbors)

    def get_neighbors(self):
        return sorted(list(self.neighbors))

    def has_neighbor(self, node):
        return node in self.neighbors

    def clear_neighbors(self):
        self.neighbors = set()

    def get_custom_neighbors(self):
        return list(self.custom_neighbors)

    def has_custom_neighbor(self, node):
        return node in self.custom_neighbors

    def clear_custom_neighbors(self):
        self.custom_neighbors = set()

class Molecule(Graph):
    """ Graph sub-class that represents a molecule. """

    def __init__(self):
        super(Molecule, self).__init__()
        self.type_degree_nodelist = dict()

    def sort_by_degree(self, ntype):
        degree_nodelist = self.type_degree_nodelist.setdefault(ntype, dict())
        node_list = self.get_node_list(ntype)

        # bucket sorting
        nodes_by_degree = {d: [] for d in degrees}
        for node in node_list:
            neighbor_num = len([n for n in node.get_neighbors() if n.ntype == ntype])
            nodes_by_degree[neighbor_num].append(node)

        sorted_nodes = []
        for degree in degrees:
            degree_nodelist[degree] = nodes_by_degree[degree]
            sorted_nodes.extend(nodes_by_degree[degree])

        self._type_list[ntype] = sorted_nodes

    def add_subgraph(self, subgraph, prefix):
        """ Add a sub-graph to the current graph. """
        for ntype in subgraph.get_types():
            new_nodes = subgraph.get_node_list(ntype)
            for node in new_nodes:
                node.ext_id = node_id(prefix, node.ext_id)
            self.add_node_list(ntype, new_nodes)

    def get_neighbor_idx_by_degree(self, ntype, neighbor_type, degree):
        node_idx = {node.ext_id: idx for idx, node in enumerate(self.get_node_list(neighbor_type))}

        neighbor_idx = []
        for node in self.type_degree_nodelist[ntype][degree]:
            neighbor_idx.append([node_idx[n.ext_id] for n in node.get_neighbors() if n.ntype == neighbor_type])

        return neighbor_idx

    def get_neighbor_idx(self, ntype, neighbor_type):
        node_idx = {node.ext_id: idx for idx, node in enumerate(self.get_node_list(neighbor_type))}

        neighbor_idx = []
        for node in self._type_list[ntype]:
            neighbor_idx.append([node_idx[n.ext_id] for n in node.get_neighbors() if n.ntype == neighbor_type])

        return neighbor_idx
