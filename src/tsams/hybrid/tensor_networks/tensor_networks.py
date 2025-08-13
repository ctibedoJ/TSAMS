"""
TIBEDO Tensor Network Base Module

This module implements the core data structures and algorithms for tensor networks,
providing the foundation for more specialized tensor network architectures.

Key components:
1. Tensor: Core tensor data structure with arbitrary rank
2. TensorNode: Node in a tensor network, containing a tensor and its connections
3. TensorLink: Connection between tensor nodes
4. TensorNetwork: Graph structure representing a tensor network
5. TensorContractor: Algorithms for contracting tensor networks
"""

import numpy as np
import torch
import opt_einsum
from typing import List, Dict, Tuple, Any, Optional, Union, Callable, Set
import logging
import networkx as nx
from dataclasses import dataclass
import string
import itertools

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Tensor:
    """
    Core tensor data structure with arbitrary rank.
    
    This class encapsulates a multi-dimensional array with named dimensions,
    supporting various tensor operations and transformations.
    """
    
    def __init__(self, 
                 data: Union[np.ndarray, torch.Tensor], 
                 indices: List[str],
                 use_gpu: bool = False):
        """
        Initialize a tensor with data and named indices.
        
        Args:
            data: Tensor data as numpy array or PyTorch tensor
            indices: List of index names, one for each dimension
            use_gpu: Whether to store the tensor on GPU
        """
        # Validate inputs
        if len(indices) != len(data.shape):
            raise ValueError(f"Number of indices ({len(indices)}) must match tensor rank ({len(data.shape)})")
        
        # Check for duplicate indices
        if len(indices) != len(set(indices)):
            raise ValueError("Duplicate indices are not allowed")
        
        # Store the indices
        self.indices = indices
        
        # Convert data to PyTorch tensor if needed
        if isinstance(data, np.ndarray):
            self.data = torch.from_numpy(data)
        else:
            self.data = data
        
        # Move to GPU if requested
        self.use_gpu = use_gpu
        if use_gpu and torch.cuda.is_available():
            self.data = self.data.cuda()
        
        logger.debug(f"Created tensor with shape {self.data.shape} and indices {indices}")
    
    @property
    def shape(self) -> Tuple[int, ...]:
        """Get the shape of the tensor."""
        return self.data.shape
    
    @property
    def rank(self) -> int:
        """Get the rank (number of dimensions) of the tensor."""
        return len(self.shape)
    
    def to_numpy(self) -> np.ndarray:
        """Convert the tensor to a numpy array."""
        if self.use_gpu:
            return self.data.cpu().numpy()
        return self.data.numpy()
    
    def to_gpu(self) -> 'Tensor':
        """Move the tensor to GPU."""
        if not self.use_gpu and torch.cuda.is_available():
            self.data = self.data.cuda()
            self.use_gpu = True
        return self
    
    def to_cpu(self) -> 'Tensor':
        """Move the tensor to CPU."""
        if self.use_gpu:
            self.data = self.data.cpu()
            self.use_gpu = False
        return self
    
    def rename_indices(self, old_to_new: Dict[str, str]) -> 'Tensor':
        """
        Rename tensor indices.
        
        Args:
            old_to_new: Dictionary mapping old index names to new index names
            
        Returns:
            Tensor with renamed indices
        """
        new_indices = [old_to_new.get(idx, idx) for idx in self.indices]
        return Tensor(self.data, new_indices, self.use_gpu)
    
    def transpose(self, new_order: List[str]) -> 'Tensor':
        """
        Transpose the tensor to a new index order.
        
        Args:
            new_order: List of indices in the desired order
            
        Returns:
            Transposed tensor
        """
        # Check that all indices are present
        if set(new_order) != set(self.indices):
            raise ValueError("New order must contain exactly the same indices as the tensor")
        
        # Get the permutation
        permutation = [self.indices.index(idx) for idx in new_order]
        
        # Transpose the data
        new_data = self.data.permute(*permutation)
        
        return Tensor(new_data, new_order, self.use_gpu)
    
    def contract(self, other: 'Tensor', optimize: str = 'auto') -> 'Tensor':
        """
        Contract this tensor with another tensor along shared indices.
        
        Args:
            other: Tensor to contract with
            optimize: Optimization strategy for contraction
            
        Returns:
            Contracted tensor
        """
        # Find shared indices
        shared_indices = set(self.indices) & set(other.indices)
        if not shared_indices:
            raise ValueError("No shared indices to contract")
        
        # Determine output indices
        output_indices = [idx for idx in self.indices if idx not in shared_indices] + \
                         [idx for idx in other.indices if idx not in shared_indices]
        
        # Prepare for einsum
        all_indices = self.indices + [idx for idx in other.indices if idx not in self.indices]
        equation = ''.join(self.indices) + ',' + ''.join(other.indices) + '->' + ''.join(output_indices)
        
        # Ensure both tensors are on the same device
        if self.use_gpu != other.use_gpu:
            if self.use_gpu:
                other_data = other.data.cuda()
            else:
                other_data = other.data.cpu()
        else:
            other_data = other.data
        
        # Perform the contraction
        if self.use_gpu or other.use_gpu:
            # Use PyTorch's einsum for GPU tensors
            result_data = torch.einsum(equation, self.data, other_data)
        else:
            # Use opt_einsum for optimized contraction
            result_data = opt_einsum.contract(equation, self.data, other_data, optimize=optimize)
        
        return Tensor(result_data, output_indices, self.use_gpu or other.use_gpu)
    
    def svd(self, left_indices: List[str], right_indices: List[str], max_singular_values: Optional[int] = None) -> Tuple['Tensor', 'Tensor', 'Tensor']:
        """
        Perform a singular value decomposition (SVD) of the tensor.
        
        Args:
            left_indices: Indices to keep in the left tensor
            right_indices: Indices to keep in the right tensor
            max_singular_values: Maximum number of singular values to keep
            
        Returns:
            Tuple of (U, S, V) tensors
        """
        # Validate indices
        if not set(left_indices + right_indices) == set(self.indices):
            raise ValueError("Left and right indices must exactly cover all tensor indices")
        
        # Reshape the tensor into a matrix
        left_dims = [self.shape[self.indices.index(idx)] for idx in left_indices]
        right_dims = [self.shape[self.indices.index(idx)] for idx in right_indices]
        
        left_size = np.prod(left_dims)
        right_size = np.prod(right_dims)
        
        # Transpose to get left and right indices grouped
        all_indices = left_indices + right_indices
        tensor_transposed = self.transpose(all_indices)
        
        # Reshape to a matrix
        matrix = tensor_transposed.data.reshape(left_size, right_size)
        
        # Perform SVD
        U, S, V = torch.svd(matrix)
        
        # Truncate if requested
        if max_singular_values is not None and max_singular_values < len(S):
            U = U[:, :max_singular_values]
            S = S[:max_singular_values]
            V = V[:, :max_singular_values]
        
        # Create bond index
        bond_index = 'b' + ''.join(sorted(left_indices + right_indices))
        
        # Reshape U and V back to tensors
        u_shape = left_dims + [S.shape[0]]
        v_shape = [S.shape[0]] + right_dims
        
        u_tensor = Tensor(U.reshape(*u_shape), left_indices + [bond_index], self.use_gpu)
        s_tensor = Tensor(torch.diag(S), [bond_index, bond_index + '*'], self.use_gpu)
        v_tensor = Tensor(V.reshape(*v_shape), [bond_index + '*'] + right_indices, self.use_gpu)
        
        return u_tensor, s_tensor, v_tensor
    
    def __repr__(self) -> str:
        """String representation of the tensor."""
        return f"Tensor(shape={self.shape}, indices={self.indices}, device={'gpu' if self.use_gpu else 'cpu'})"


@dataclass
class TensorLink:
    """
    Connection between tensor nodes in a tensor network.
    
    This class represents an edge in the tensor network graph, connecting
    two tensor nodes through specific indices.
    """
    source_node: 'TensorNode'
    target_node: 'TensorNode'
    source_index: str
    target_index: str
    bond_dimension: int
    
    def __repr__(self) -> str:
        """String representation of the tensor link."""
        return f"TensorLink({self.source_node.name}[{self.source_index}] -> {self.target_node.name}[{self.target_index}], dim={self.bond_dimension})"


class TensorNode:
    """
    Node in a tensor network, containing a tensor and its connections.
    
    This class represents a vertex in the tensor network graph, containing
    a tensor and its connections to other nodes.
    """
    
    def __init__(self, 
                 name: str,
                 tensor: Tensor):
        """
        Initialize a tensor node.
        
        Args:
            name: Name of the node
            tensor: Tensor contained in the node
        """
        self.name = name
        self.tensor = tensor
        self.links: Dict[str, TensorLink] = {}  # Maps index name to link
        
        logger.debug(f"Created tensor node {name} with tensor shape {tensor.shape}")
    
    def add_link(self, link: TensorLink) -> None:
        """
        Add a link to the node.
        
        Args:
            link: Link to add
        """
        if link.source_node == self:
            self.links[link.source_index] = link
        elif link.target_node == self:
            self.links[link.target_index] = link
        else:
            raise ValueError(f"Link does not connect to node {self.name}")
    
    def remove_link(self, index: str) -> None:
        """
        Remove a link from the node.
        
        Args:
            index: Index name of the link to remove
        """
        if index in self.links:
            del self.links[index]
    
    def get_neighbors(self) -> Set['TensorNode']:
        """
        Get all neighboring nodes.
        
        Returns:
            Set of neighboring nodes
        """
        neighbors = set()
        for link in self.links.values():
            if link.source_node == self:
                neighbors.add(link.target_node)
            else:
                neighbors.add(link.source_node)
        return neighbors
    
    def __repr__(self) -> str:
        """String representation of the tensor node."""
        return f"TensorNode({self.name}, tensor_shape={self.tensor.shape}, indices={self.tensor.indices})"


class TensorNetwork:
    """
    Graph structure representing a tensor network.
    
    This class provides methods for building, manipulating, and contracting
    tensor networks.
    """
    
    def __init__(self, name: str = "TensorNetwork"):
        """
        Initialize an empty tensor network.
        
        Args:
            name: Name of the tensor network
        """
        self.name = name
        self.nodes: Dict[str, TensorNode] = {}
        self.links: List[TensorLink] = []
        self.graph = nx.Graph()
        
        logger.info(f"Created tensor network {name}")
    
    def add_node(self, node: TensorNode) -> None:
        """
        Add a node to the tensor network.
        
        Args:
            node: Node to add
        """
        if node.name in self.nodes:
            raise ValueError(f"Node with name {node.name} already exists")
        
        self.nodes[node.name] = node
        self.graph.add_node(node.name, tensor_node=node)
        
        logger.debug(f"Added node {node.name} to tensor network {self.name}")
    
    def add_link(self, 
                source_node: Union[str, TensorNode], 
                target_node: Union[str, TensorNode],
                source_index: str,
                target_index: str) -> TensorLink:
        """
        Add a link between two nodes in the tensor network.
        
        Args:
            source_node: Source node or its name
            target_node: Target node or its name
            source_index: Index name in the source node
            target_index: Index name in the target node
            
        Returns:
            Created tensor link
        """
        # Get the actual nodes
        if isinstance(source_node, str):
            source_node = self.nodes[source_node]
        if isinstance(target_node, str):
            target_node = self.nodes[target_node]
        
        # Validate indices
        if source_index not in source_node.tensor.indices:
            raise ValueError(f"Index {source_index} not found in source node {source_node.name}")
        if target_index not in target_node.tensor.indices:
            raise ValueError(f"Index {target_index} not found in target node {target_node.name}")
        
        # Get bond dimension
        source_dim = source_node.tensor.shape[source_node.tensor.indices.index(source_index)]
        target_dim = target_node.tensor.shape[target_node.tensor.indices.index(target_index)]
        
        if source_dim != target_dim:
            raise ValueError(f"Incompatible dimensions: {source_dim} vs {target_dim}")
        
        # Create the link
        link = TensorLink(
            source_node=source_node,
            target_node=target_node,
            source_index=source_index,
            target_index=target_index,
            bond_dimension=source_dim
        )
        
        # Add the link to the nodes
        source_node.add_link(link)
        target_node.add_link(link)
        
        # Add the link to the network
        self.links.append(link)
        
        # Update the graph
        self.graph.add_edge(source_node.name, target_node.name, link=link)
        
        logger.debug(f"Added link {source_node.name}[{source_index}] -> {target_node.name}[{target_index}]")
        
        return link
    
    def remove_node(self, node_name: str) -> None:
        """
        Remove a node from the tensor network.
        
        Args:
            node_name: Name of the node to remove
        """
        if node_name not in self.nodes:
            raise ValueError(f"Node {node_name} not found")
        
        node = self.nodes[node_name]
        
        # Remove all links connected to this node
        links_to_remove = list(node.links.values())
        for link in links_to_remove:
            self.remove_link(link)
        
        # Remove the node
        del self.nodes[node_name]
        self.graph.remove_node(node_name)
        
        logger.debug(f"Removed node {node_name} from tensor network {self.name}")
    
    def remove_link(self, link: TensorLink) -> None:
        """
        Remove a link from the tensor network.
        
        Args:
            link: Link to remove
        """
        # Remove the link from the nodes
        source_node = link.source_node
        target_node = link.target_node
        
        source_node.remove_link(link.source_index)
        target_node.remove_link(link.target_index)
        
        # Remove the link from the network
        self.links.remove(link)
        
        # Update the graph
        self.graph.remove_edge(source_node.name, target_node.name)
        
        logger.debug(f"Removed link {source_node.name}[{link.source_index}] -> {target_node.name}[{link.target_index}]")
    
    def get_subnetwork(self, node_names: List[str]) -> 'TensorNetwork':
        """
        Extract a subnetwork containing only the specified nodes.
        
        Args:
            node_names: Names of nodes to include in the subnetwork
            
        Returns:
            Extracted subnetwork
        """
        # Create a new tensor network
        subnetwork = TensorNetwork(f"{self.name}_sub")
        
        # Add the nodes
        for name in node_names:
            if name in self.nodes:
                subnetwork.add_node(self.nodes[name])
        
        # Add the links between these nodes
        for link in self.links:
            source_name = link.source_node.name
            target_name = link.target_node.name
            if source_name in node_names and target_name in node_names:
                subnetwork.add_link(
                    source_node=link.source_node,
                    target_node=link.target_node,
                    source_index=link.source_index,
                    target_index=link.target_index
                )
        
        return subnetwork
    
    def find_contraction_path(self, output_node: Optional[str] = None) -> List[Tuple[int, int]]:
        """
        Find an optimal contraction path for the tensor network.
        
        Args:
            output_node: Name of the node to designate as output (will be contracted last)
            
        Returns:
            List of node pairs to contract
        """
        # Create a list of tensors and their indices
        tensors = []
        indices = []
        node_to_idx = {}
        
        for i, (name, node) in enumerate(self.nodes.items()):
            tensors.append(node.tensor.data)
            indices.append(node.tensor.indices)
            node_to_idx[name] = i
        
        # Find the optimal contraction path
        path, _ = opt_einsum.contract_path(*indices, tensors, optimize='auto')
        
        # Convert the path to node pairs
        node_pairs = []
        for i, j in path:
            # Find the corresponding nodes
            node_i = list(self.nodes.keys())[i]
            node_j = list(self.nodes.keys())[j]
            node_pairs.append((node_i, node_j))
        
        return node_pairs
    
    def visualize(self, filename: Optional[str] = None) -> None:
        """
        Visualize the tensor network.
        
        Args:
            filename: If provided, save the visualization to this file
        """
        import matplotlib.pyplot as plt
        
        # Create a position layout
        pos = nx.spring_layout(self.graph)
        
        # Draw the graph
        plt.figure(figsize=(10, 8))
        nx.draw(self.graph, pos, with_labels=True, node_color='lightblue', 
                node_size=1500, font_size=10, font_weight='bold')
        
        # Draw edge labels
        edge_labels = {}
        for link in self.links:
            edge_labels[(link.source_node.name, link.target_node.name)] = f"{link.source_index}-{link.target_index}"
        
        nx.draw_networkx_edge_labels(self.graph, pos, edge_labels=edge_labels, font_size=8)
        
        # Save or show
        if filename:
            plt.savefig(filename)
        else:
            plt.show()
    
    def __repr__(self) -> str:
        """String representation of the tensor network."""
        return f"TensorNetwork({self.name}, nodes={len(self.nodes)}, links={len(self.links)})"


class TensorContractor:
    """
    Algorithms for contracting tensor networks.
    
    This class provides various methods for contracting tensor networks,
    with different optimization strategies and approximation techniques.
    """
    
    @staticmethod
    def contract_pair(network: TensorNetwork, node1_name: str, node2_name: str) -> TensorNode:
        """
        Contract a pair of nodes in the tensor network.
        
        Args:
            network: Tensor network
            node1_name: Name of the first node
            node2_name: Name of the second node
            
        Returns:
            New node resulting from the contraction
        """
        # Get the nodes
        node1 = network.nodes[node1_name]
        node2 = network.nodes[node2_name]
        
        # Find shared indices
        shared_links = []
        for idx1, link1 in node1.links.items():
            for idx2, link2 in node2.links.items():
                if link1 == link2:
                    shared_links.append(link1)
        
        # Contract the tensors
        result_tensor = node1.tensor.contract(node2.tensor)
        
        # Create a new node
        new_node_name = f"{node1_name}_{node2_name}"
        new_node = TensorNode(new_node_name, result_tensor)
        
        # Remove the old nodes and add the new one
        network.remove_node(node1_name)
        network.remove_node(node2_name)
        network.add_node(new_node)
        
        # Reconnect the links
        for link in shared_links:
            if link.source_node == node1 or link.source_node == node2:
                other_node = link.target_node
                other_index = link.target_index
            else:
                other_node = link.source_node
                other_index = link.source_index
            
            # Skip if the other node was one of the contracted nodes
            if other_node == node1 or other_node == node2:
                continue
            
            # Find the corresponding index in the new tensor
            if link.source_node == node1:
                new_index = link.source_index
            elif link.source_node == node2:
                new_index = link.source_index
            elif link.target_node == node1:
                new_index = link.target_index
            else:  # link.target_node == node2
                new_index = link.target_index
            
            # Add a new link
            network.add_link(
                source_node=new_node,
                target_node=other_node,
                source_index=new_index,
                target_index=other_index
            )
        
        return new_node
    
    @staticmethod
    def contract_network(network: TensorNetwork, contraction_path: Optional[List[Tuple[str, str]]] = None) -> Tensor:
        """
        Contract the entire tensor network.
        
        Args:
            network: Tensor network to contract
            contraction_path: Sequence of node pairs to contract
            
        Returns:
            Final contracted tensor
        """
        # Make a copy of the network to avoid modifying the original
        network_copy = TensorNetwork(f"{network.name}_copy")
        for node_name, node in network.nodes.items():
            network_copy.add_node(node)
        for link in network.links:
            network_copy.add_link(
                source_node=link.source_node,
                target_node=link.target_node,
                source_index=link.source_index,
                target_index=link.target_index
            )
        
        # Find a contraction path if not provided
        if contraction_path is None:
            path_pairs = network.find_contraction_path()
            contraction_path = [(list(network.nodes.keys())[i], list(network.nodes.keys())[j]) for i, j in path_pairs]
        
        # Contract the network
        while len(network_copy.nodes) > 1:
            # Get the next pair to contract
            if contraction_path:
                node1_name, node2_name = contraction_path.pop(0)
            else:
                # If we run out of path (e.g., due to approximations changing the network),
                # just contract the first two nodes
                node1_name, node2_name = list(network_copy.nodes.keys())[:2]
            
            # Contract the pair
            TensorContractor.contract_pair(network_copy, node1_name, node2_name)
        
        # Return the final tensor
        return list(network_copy.nodes.values())[0].tensor
    
    @staticmethod
    def approximate_contraction(network: TensorNetwork, 
                               max_bond_dimension: int, 
                               contraction_path: Optional[List[Tuple[str, str]]] = None) -> Tensor:
        """
        Contract the tensor network with bond dimension truncation.
        
        Args:
            network: Tensor network to contract
            max_bond_dimension: Maximum bond dimension to keep during contraction
            contraction_path: Sequence of node pairs to contract
            
        Returns:
            Approximated contracted tensor
        """
        # Make a copy of the network to avoid modifying the original
        network_copy = TensorNetwork(f"{network.name}_approx")
        for node_name, node in network.nodes.items():
            network_copy.add_node(node)
        for link in network.links:
            network_copy.add_link(
                source_node=link.source_node,
                target_node=link.target_node,
                source_index=link.source_index,
                target_index=link.target_index
            )
        
        # Find a contraction path if not provided
        if contraction_path is None:
            path_pairs = network.find_contraction_path()
            contraction_path = [(list(network.nodes.keys())[i], list(network.nodes.keys())[j]) for i, j in path_pairs]
        
        # Contract the network with approximations
        while len(network_copy.nodes) > 1:
            # Get the next pair to contract
            if contraction_path:
                node1_name, node2_name = contraction_path.pop(0)
            else:
                # If we run out of path, just contract the first two nodes
                node1_name, node2_name = list(network_copy.nodes.keys())[:2]
            
            # Contract the pair
            new_node = TensorContractor.contract_pair(network_copy, node1_name, node2_name)
            
            # Apply SVD truncation to the new node if its rank is high
            if new_node.tensor.rank > 4:  # Arbitrary threshold
                # Identify external indices (those connected to other nodes)
                external_indices = set()
                for idx, link in new_node.links.items():
                    external_indices.add(idx)
                
                # Identify internal indices (those not connected to other nodes)
                internal_indices = set(new_node.tensor.indices) - external_indices
                
                # If there are internal indices, we can try to reduce the tensor
                if internal_indices:
                    # Split the tensor into two parts and truncate the bond
                    left_indices = list(external_indices)[:len(external_indices)//2]
                    right_indices = list(external_indices)[len(external_indices)//2:] + list(internal_indices)
                    
                    # Perform SVD and truncate
                    u_tensor, s_tensor, v_tensor = new_node.tensor.svd(
                        left_indices=left_indices,
                        right_indices=right_indices,
                        max_singular_values=max_bond_dimension
                    )
                    
                    # Contract s with v
                    sv_tensor = s_tensor.contract(v_tensor)
                    
                    # Update the node's tensor
                    new_node.tensor = u_tensor.contract(sv_tensor)
        
        # Return the final tensor
        return list(network_copy.nodes.values())[0].tensor
"""