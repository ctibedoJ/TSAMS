"""
Protein Folding with Linear Scaling

This module implements protein folding prediction tools using the Tibedo Framework,
achieving linear scaling with system size.
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from scipy.spatial.distance import pdist, squareform
import time

# Import Tibedo components
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from tibedo.core.spinor.reduction_chain import ReductionChain
from tibedo.core.prime_indexed.prime_indexed_structure import PrimeIndexedStructure
from tibedo.core.advanced.protein_simulator import ProteinSimulator
from tibedo.ml.neural_networks.tibedo_neural_network import TibedoNeuralNetwork


class AminoAcidEncoder:
    """
    Encoder for amino acid sequences.
    
    This class provides methods for encoding amino acid sequences into
    numerical representations suitable for machine learning.
    """
    
    def __init__(self):
        """
        Initialize the AminoAcidEncoder.
        """
        # Define amino acid properties
        self.amino_acids = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'K', 'L',
            'M', 'N', 'P', 'Q', 'R', 'S', 'T', 'V', 'W', 'Y'
        ]
        
        # Hydrophobicity scale (Kyte-Doolittle)
        self.hydrophobicity = {
            'A': 1.8, 'C': 2.5, 'D': -3.5, 'E': -3.5, 'F': 2.8,
            'G': -0.4, 'H': -3.2, 'I': 4.5, 'K': -3.9, 'L': 3.8,
            'M': 1.9, 'N': -3.5, 'P': -1.6, 'Q': -3.5, 'R': -4.5,
            'S': -0.8, 'T': -0.7, 'V': 4.2, 'W': -0.9, 'Y': -1.3
        }
        
        # Volume (Å³)
        self.volume = {
            'A': 88.6, 'C': 108.5, 'D': 111.1, 'E': 138.4, 'F': 189.9,
            'G': 60.1, 'H': 153.2, 'I': 166.7, 'K': 168.6, 'L': 166.7,
            'M': 162.9, 'N': 114.1, 'P': 112.7, 'Q': 143.8, 'R': 173.4,
            'S': 89.0, 'T': 116.1, 'V': 140.0, 'W': 227.8, 'Y': 193.6
        }
        
        # Charge at pH 7
        self.charge = {
            'A': 0, 'C': 0, 'D': -1, 'E': -1, 'F': 0,
            'G': 0, 'H': 0.1, 'I': 0, 'K': 1, 'L': 0,
            'M': 0, 'N': 0, 'P': 0, 'Q': 0, 'R': 1,
            'S': 0, 'T': 0, 'V': 0, 'W': 0, 'Y': 0
        }
        
        # One-hot encoding matrix
        self.one_hot = np.eye(len(self.amino_acids))
        
        # Create amino acid to index mapping
        self.aa_to_idx = {aa: i for i, aa in enumerate(self.amino_acids)}
    
    def encode_sequence(self, sequence, encoding_type='one_hot', window_size=1):
        """
        Encode an amino acid sequence.
        
        Args:
            sequence (str): Amino acid sequence
            encoding_type (str): Type of encoding ('one_hot', 'properties', or 'combined')
            window_size (int): Window size for context
            
        Returns:
            np.ndarray: Encoded sequence
        """
        if encoding_type == 'one_hot':
            # One-hot encoding
            encoded = np.zeros((len(sequence), len(self.amino_acids)))
            
            for i, aa in enumerate(sequence):
                if aa in self.aa_to_idx:
                    encoded[i, self.aa_to_idx[aa]] = 1
            
            return encoded
            
        elif encoding_type == 'properties':
            # Property-based encoding
            encoded = np.zeros((len(sequence), 3))
            
            for i, aa in enumerate(sequence):
                if aa in self.hydrophobicity:
                    encoded[i, 0] = self.hydrophobicity[aa]
                    encoded[i, 1] = self.volume[aa]
                    encoded[i, 2] = self.charge[aa]
            
            # Normalize properties
            encoded[:, 0] = (encoded[:, 0] - np.min(list(self.hydrophobicity.values()))) / \
                           (np.max(list(self.hydrophobicity.values())) - np.min(list(self.hydrophobicity.values())))
            
            encoded[:, 1] = (encoded[:, 1] - np.min(list(self.volume.values()))) / \
                           (np.max(list(self.volume.values())) - np.min(list(self.volume.values())))
            
            encoded[:, 2] = (encoded[:, 2] - np.min(list(self.charge.values()))) / \
                           (np.max(list(self.charge.values())) - np.min(list(self.charge.values())))
            
            return encoded
            
        elif encoding_type == 'combined':
            # Combined encoding (one-hot + properties)
            one_hot_encoded = self.encode_sequence(sequence, 'one_hot')
            properties_encoded = self.encode_sequence(sequence, 'properties')
            
            encoded = np.concatenate([one_hot_encoded, properties_encoded], axis=1)
            
            return encoded
            
        else:
            raise ValueError(f"Unknown encoding type: {encoding_type}")
    
    def encode_with_context(self, sequence, window_size=3, encoding_type='combined'):
        """
        Encode an amino acid sequence with context.
        
        Args:
            sequence (str): Amino acid sequence
            window_size (int): Window size for context
            encoding_type (str): Type of encoding ('one_hot', 'properties', or 'combined')
            
        Returns:
            np.ndarray: Encoded sequence with context
        """
        # Encode the sequence
        encoded = self.encode_sequence(sequence, encoding_type)
        
        # Add context
        context_encoded = []
        
        for i in range(len(sequence)):
            # Extract window centered at position i
            start = max(0, i - window_size // 2)
            end = min(len(sequence), i + window_size // 2 + 1)
            
            # Pad if needed
            pad_left = max(0, window_size // 2 - i)
            pad_right = max(0, i + window_size // 2 + 1 - len(sequence))
            
            # Extract window
            window = encoded[start:end]
            
            # Pad window
            if pad_left > 0:
                window = np.vstack([np.zeros((pad_left, encoded.shape[1])), window])
            
            if pad_right > 0:
                window = np.vstack([window, np.zeros((pad_right, encoded.shape[1]))])
            
            # Flatten window
            flat_window = window.flatten()
            
            context_encoded.append(flat_window)
        
        return np.array(context_encoded)


class ProteinStructureDataset(Dataset):
    """
    Dataset for protein structure prediction.
    
    This class provides a PyTorch dataset for protein structure prediction,
    mapping amino acid sequences to 3D coordinates.
    """
    
    def __init__(self, sequences, structures, encoder, window_size=3):
        """
        Initialize the ProteinStructureDataset.
        
        Args:
            sequences (list): List of amino acid sequences
            structures (list): List of 3D structures (N, 3)
            encoder (AminoAcidEncoder): Amino acid encoder
            window_size (int): Window size for context
        """
        self.sequences = sequences
        self.structures = structures
        self.encoder = encoder
        self.window_size = window_size
        
        # Encode sequences
        self.encoded_sequences = []
        for seq in sequences:
            encoded = encoder.encode_with_context(seq, window_size)
            self.encoded_sequences.append(encoded)
    
    def __len__(self):
        """
        Get the number of samples in the dataset.
        
        Returns:
            int: Number of samples
        """
        return len(self.sequences)
    
    def __getitem__(self, idx):
        """
        Get a sample from the dataset.
        
        Args:
            idx (int): Index
            
        Returns:
            tuple: (encoded_sequence, structure)
        """
        return torch.tensor(self.encoded_sequences[idx], dtype=torch.float32), \
               torch.tensor(self.structures[idx], dtype=torch.float32)


class TibedoProteinFolder:
    """
    Protein folding prediction using the Tibedo Framework.
    
    This class implements protein folding prediction tools using the Tibedo Framework,
    achieving linear scaling with system size.
    """
    
    def __init__(self, hidden_dims=[256, 128, 64], window_size=3):
        """
        Initialize the TibedoProteinFolder.
        
        Args:
            hidden_dims (list): List of hidden layer dimensions
            window_size (int): Window size for context
        """
        self.hidden_dims = hidden_dims
        self.window_size = window_size
        
        # Create amino acid encoder
        self.encoder = AminoAcidEncoder()
        
        # Calculate input dimension
        encoding_dim = len(self.encoder.amino_acids) + 3  # one-hot + properties
        self.input_dim = encoding_dim * (2 * (window_size // 2) + 1)
        
        # Create model
        self.model = None
        
        # Create spinor reduction chain
        self.reduction_chain = ReductionChain(
            initial_dimension=16,
            chain_length=5
        )
        
        # Create prime-indexed structure
        self.prime_structure = PrimeIndexedStructure(max_index=100)
        
        # Create protein simulator
        self.protein_simulator = ProteinSimulator()
        
        # Training history
        self.train_losses = []
        self.val_losses = []
    
    def build_model(self):
        """
        Build the protein folding model.
        """
        # Create model
        self.model = TibedoNeuralNetwork(
            input_dim=self.input_dim,
            hidden_dims=self.hidden_dims,
            output_dim=3  # 3D coordinates
        )
        
        # Create optimizer
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.001)
        
        # Loss function
        self.criterion = nn.MSELoss()
    
    def train(self, train_sequences, train_structures, val_sequences=None, val_structures=None,
              epochs=100, batch_size=32):
        """
        Train the protein folding model.
        
        Args:
            train_sequences (list): List of training amino acid sequences
            train_structures (list): List of training 3D structures
            val_sequences (list, optional): List of validation amino acid sequences
            val_structures (list, optional): List of validation 3D structures
            epochs (int): Number of training epochs
            batch_size (int): Batch size
        """
        # Build model if not already built
        if self.model is None:
            self.build_model()
        
        # Create datasets
        train_dataset = ProteinStructureDataset(
            train_sequences, train_structures, self.encoder, self.window_size
        )
        
        train_loader = DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )
        
        if val_sequences is not None and val_structures is not None:
            val_dataset = ProteinStructureDataset(
                val_sequences, val_structures, self.encoder, self.window_size
            )
            
            val_loader = DataLoader(
                val_dataset, batch_size=batch_size, shuffle=False
            )
        else:
            val_loader = None
        
        # Train model
        for epoch in range(epochs):
            # Train
            self.model.train()
            epoch_loss = 0.0
            
            for batch_x, batch_y in train_loader:
                # Zero gradients
                self.optimizer.zero_grad()
                
                # Forward pass
                outputs = self.model(batch_x)
                
                # Compute loss
                loss = self.criterion(outputs, batch_y)
                
                # Backward pass and optimize
                loss.backward()
                self.optimizer.step()
                
                epoch_loss += loss.item() * len(batch_x)
            
            # Average loss
            epoch_loss /= len(train_dataset)
            self.train_losses.append(epoch_loss)
            
            # Validate
            if val_loader is not None:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}, Val Loss: {val_loss:.6f}')
            else:
                if (epoch + 1) % 10 == 0:
                    print(f'Epoch {epoch+1}/{epochs}, Train Loss: {epoch_loss:.6f}')
    
    def validate(self, val_loader):
        """
        Validate the protein folding model.
        
        Args:
            val_loader (DataLoader): Validation data loader
            
        Returns:
            float: Validation loss
        """
        self.model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for batch_x, batch_y in val_loader:
                # Forward pass
                outputs = self.model(batch_x)
                
                # Compute loss
                loss = self.criterion(outputs, batch_y)
                
                val_loss += loss.item() * len(batch_x)
        
        # Average loss
        val_loss /= len(val_loader.dataset)
        
        return val_loss
    
    def predict_structure(self, sequence):
        """
        Predict the 3D structure of a protein.
        
        Args:
            sequence (str): Amino acid sequence
            
        Returns:
            np.ndarray: Predicted 3D structure (N, 3)
        """
        # Encode sequence
        encoded = self.encoder.encode_with_context(sequence, self.window_size)
        encoded = torch.tensor(encoded, dtype=torch.float32)
        
        # Predict structure
        self.model.eval()
        with torch.no_grad():
            predicted = self.model(encoded)
        
        return predicted.numpy()
    
    def refine_structure(self, sequence, initial_structure):
        """
        Refine a protein structure using the Tibedo Framework.
        
        Args:
            sequence (str): Amino acid sequence
            initial_structure (np.ndarray): Initial 3D structure (N, 3)
            
        Returns:
            np.ndarray: Refined 3D structure (N, 3)
        """
        # Use protein simulator to refine structure
        refined_structure = self.protein_simulator.refine_structure(
            sequence, initial_structure
        )
        
        return refined_structure
    
    def calculate_rmsd(self, structure1, structure2):
        """
        Calculate the root-mean-square deviation (RMSD) between two structures.
        
        Args:
            structure1 (np.ndarray): First structure (N, 3)
            structure2 (np.ndarray): Second structure (N, 3)
            
        Returns:
            float: RMSD
        """
        # Check if structures have the same shape
        if structure1.shape != structure2.shape:
            raise ValueError("Structures must have the same shape")
        
        # Calculate squared differences
        squared_diff = np.sum((structure1 - structure2)**2, axis=1)
        
        # Calculate RMSD
        rmsd = np.sqrt(np.mean(squared_diff))
        
        return rmsd
    
    def calculate_contact_map(self, structure, threshold=8.0):
        """
        Calculate the contact map of a protein structure.
        
        Args:
            structure (np.ndarray): 3D structure (N, 3)
            threshold (float): Distance threshold for contacts
            
        Returns:
            np.ndarray: Contact map (N, N)
        """
        # Calculate pairwise distances
        distances = squareform(pdist(structure))
        
        # Create contact map
        contact_map = distances < threshold
        
        return contact_map
    
    def save_model(self, path):
        """
        Save the protein folding model.
        
        Args:
            path (str): Path to save the model
        """
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'input_dim': self.input_dim,
            'hidden_dims': self.hidden_dims,
            'window_size': self.window_size
        }, path)
    
    def load_model(self, path):
        """
        Load the protein folding model.
        
        Args:
            path (str): Path to load the model from
        """
        checkpoint = torch.load(path)
        
        # Update parameters
        self.input_dim = checkpoint['input_dim']
        self.hidden_dims = checkpoint['hidden_dims']
        self.window_size = checkpoint['window_size']
        
        # Build model
        self.build_model()
        
        # Load state dictionaries
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def visualize_structure(self, structure, sequence=None):
        """
        Visualize a protein structure.
        
        Args:
            structure (np.ndarray): 3D structure (N, 3)
            sequence (str, optional): Amino acid sequence
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig = plt.figure(figsize=(10, 8))
        ax = fig.add_subplot(111, projection='3d')
        
        # Plot backbone
        ax.plot(structure[:, 0], structure[:, 1], structure[:, 2], 'k-', alpha=0.7)
        
        # Plot atoms
        if sequence is not None:
            # Define colors for different amino acids
            hydrophobic = ['A', 'C', 'F', 'I', 'L', 'M', 'V', 'W', 'Y']
            polar = ['N', 'Q', 'S', 'T']
            positive = ['H', 'K', 'R']
            negative = ['D', 'E']
            special = ['G', 'P']
            
            colors = []
            for aa in sequence:
                if aa in hydrophobic:
                    colors.append('red')
                elif aa in polar:
                    colors.append('blue')
                elif aa in positive:
                    colors.append('green')
                elif aa in negative:
                    colors.append('purple')
                elif aa in special:
                    colors.append('orange')
                else:
                    colors.append('gray')
            
            # Plot atoms with colors
            ax.scatter(structure[:, 0], structure[:, 1], structure[:, 2], c=colors, s=100)
        else:
            # Plot atoms without colors
            ax.scatter(structure[:, 0], structure[:, 1], structure[:, 2], c='blue', s=100)
        
        # Set labels
        ax.set_xlabel('X (Å)')
        ax.set_ylabel('Y (Å)')
        ax.set_zlabel('Z (Å)')
        ax.set_title('Protein Structure')
        
        # Set equal aspect ratio
        ax.set_box_aspect([1, 1, 1])
        
        return fig
    
    def visualize_contact_map(self, contact_map):
        """
        Visualize a protein contact map.
        
        Args:
            contact_map (np.ndarray): Contact map (N, N)
            
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot contact map
        ax.imshow(contact_map, cmap='binary', interpolation='none')
        
        # Set labels
        ax.set_xlabel('Residue Index')
        ax.set_ylabel('Residue Index')
        ax.set_title('Contact Map')
        
        return fig
    
    def visualize_learning_curve(self):
        """
        Visualize the learning curve.
        
        Returns:
            matplotlib.figure.Figure: The visualization figure
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Plot training loss
        ax.plot(range(1, len(self.train_losses) + 1), self.train_losses, label='Training Loss')
        
        # Plot validation loss if available
        if self.val_losses:
            ax.plot(range(1, len(self.val_losses) + 1), self.val_losses, label='Validation Loss')
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        ax.set_title('Learning Curve')
        ax.legend()
        ax.grid(True)
        
        return fig