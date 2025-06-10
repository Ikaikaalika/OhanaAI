"""
Graph Neural Network model implementation using MLX.
Implements Graph Attention Network (GAT) with edge-type awareness for genealogical relationship modeling.
"""

import mlx.core as mx
import mlx.nn as nn
import mlx.optimizers as optim
from typing import List, Tuple, Optional
import math
import logging

logger = logging.getLogger(__name__)

class GraphAttentionLayer(nn.Module):
    """Graph Attention Layer with edge type support."""
    
    def __init__(self, in_features: int, out_features: int, num_heads: int = 1, 
                 num_edge_types: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.dropout = dropout
        
        # Ensure out_features is divisible by num_heads
        assert out_features % num_heads == 0
        self.head_dim = out_features // num_heads
        
        # Linear transformations for each head
        self.W_query = nn.Linear(in_features, out_features, bias=False)
        self.W_key = nn.Linear(in_features, out_features, bias=False)
        self.W_value = nn.Linear(in_features, out_features, bias=False)
        
        # Edge type embeddings
        self.edge_type_embedding = nn.Embedding(num_edge_types, self.head_dim)
        
        # Attention mechanism
        self.attention = nn.MultiHeadAttention(out_features, num_heads, bias=False)
        
        # Output projection
        self.out_proj = nn.Linear(out_features, out_features)
        
        # Normalization and dropout
        self.layer_norm = nn.LayerNorm(out_features)
        self.dropout_layer = nn.Dropout(dropout)
        
    def __call__(self, x: mx.array, edge_index: mx.array, edge_types: mx.array) -> mx.array:
        """
        Forward pass of GAT layer.
        
        Args:
            x: Node features [num_nodes, in_features]
            edge_index: Edge indices [2, num_edges]  
            edge_types: Edge types [num_edges]
        
        Returns:
            Updated node features [num_nodes, out_features]
        """
        num_nodes = x.shape[0]
        
        if edge_index.shape[1] == 0:
            # Handle empty graph case
            transformed = self.W_value(x)
            return self.layer_norm(transformed + x if x.shape[-1] == self.out_features else transformed)
        
        # Transform node features
        queries = self.W_query(x)  # [num_nodes, out_features]
        keys = self.W_key(x)       # [num_nodes, out_features]
        values = self.W_value(x)   # [num_nodes, out_features]
        
        # Reshape for multi-head attention
        queries = queries.reshape(num_nodes, self.num_heads, self.head_dim)
        keys = keys.reshape(num_nodes, self.num_heads, self.head_dim)
        values = values.reshape(num_nodes, self.num_heads, self.head_dim)
        
        # Get edge type embeddings
        edge_embeds = self.edge_type_embedding(edge_types)  # [num_edges, head_dim]
        edge_embeds = mx.expand_dims(edge_embeds, axis=1)   # [num_edges, 1, head_dim]
        edge_embeds = mx.broadcast_to(edge_embeds, (edge_embeds.shape[0], self.num_heads, self.head_dim))
        
        # Compute attention for each edge
        src_nodes = edge_index[0]  # Source nodes
        dst_nodes = edge_index[1]  # Destination nodes
        
        # Get features for source and destination nodes
        src_queries = queries[dst_nodes]  # [num_edges, num_heads, head_dim]
        src_keys = keys[src_nodes]        # [num_edges, num_heads, head_dim]
        src_values = values[src_nodes]    # [num_edges, num_heads, head_dim]
        
        # Incorporate edge type information into attention
        # Modify keys with edge type embeddings
        modified_keys = src_keys + edge_embeds
        
        # Compute attention scores
        attention_scores = mx.sum(src_queries * modified_keys, axis=-1)  # [num_edges, num_heads]
        attention_scores = attention_scores / math.sqrt(self.head_dim)
        
        # Apply softmax per destination node
        attention_weights = self._softmax_per_node(attention_scores, dst_nodes, num_nodes)
        
        # Apply dropout
        attention_weights = self.dropout_layer(attention_weights)
        
        # Aggregate messages
        attention_weights = mx.expand_dims(attention_weights, axis=-1)  # [num_edges, num_heads, 1]
        messages = attention_weights * src_values  # [num_edges, num_heads, head_dim]
        
        # Aggregate messages per destination node
        aggregated = self._aggregate_messages(messages, dst_nodes, num_nodes)
        
        # Reshape and project
        aggregated = aggregated.reshape(num_nodes, self.out_features)
        output = self.out_proj(aggregated)
        
        # Apply residual connection and layer norm
        if x.shape[-1] == self.out_features:
            output = self.layer_norm(output + x)
        else:
            output = self.layer_norm(output)
        
        return output
    
    def _softmax_per_node(self, scores: mx.array, dst_nodes: mx.array, num_nodes: int) -> mx.array:
        """Apply softmax normalization per destination node."""
        # Create a large negative value for masking
        large_neg = -1e9
        
        # Initialize attention matrix
        attention_matrix = mx.full((num_nodes, scores.shape[1], scores.shape[0]), large_neg)
        
        # Fill in the actual scores
        for i in range(scores.shape[0]):
            dst_node = dst_nodes[i]
            attention_matrix = attention_matrix.at[dst_node, :, i].set(scores[i])
        
        # Apply softmax along the edge dimension for each node
        attention_weights = mx.softmax(attention_matrix, axis=-1)
        
        # Extract the relevant attention weights
        result = mx.zeros_like(scores)
        for i in range(scores.shape[0]):
            dst_node = dst_nodes[i]
            result = result.at[i].set(attention_weights[dst_node, :, i])
        
        return result
    
    def _aggregate_messages(self, messages: mx.array, dst_nodes: mx.array, num_nodes: int) -> mx.array:
        """Aggregate messages per destination node."""
        # Initialize output
        output = mx.zeros((num_nodes, self.num_heads, self.head_dim))
        
        # Sum messages for each destination node
        for i in range(messages.shape[0]):
            dst_node = dst_nodes[i]
            output = output.at[dst_node].add(messages[i])
        
        return output

class OhanaAIModel(nn.Module):
    """Complete OhanaAI model for genealogical parent prediction."""
    
    def __init__(self, input_features: int, hidden_dim: int = 256, num_heads: int = 4,
                 num_layers: int = 3, num_edge_types: int = 4, dropout: float = 0.1):
        super().__init__()
        
        self.input_features = input_features
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.dropout = dropout
        
        # Input projection
        self.input_proj = nn.Linear(input_features, hidden_dim)
        
        # GAT layers
        self.gat_layers = []
        for i in range(num_layers):
            layer = GraphAttentionLayer(
                in_features=hidden_dim,
                out_features=hidden_dim,
                num_heads=num_heads,
                num_edge_types=num_edge_types,
                dropout=dropout
            )
            self.gat_layers.append(layer)
        
        # Node embedding projection
        self.node_embedding = nn.Linear(hidden_dim, hidden_dim)
        
        # Parent prediction head
        self.parent_predictor = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize model weights."""
        def init_linear(m):
            if isinstance(m, nn.Linear):
                std = math.sqrt(2.0 / (m.weight.shape[0] + m.weight.shape[1]))
                m.weight = mx.random.normal(m.weight.shape) * std
                if hasattr(m, 'bias') and m.bias is not None:
                    m.bias = mx.zeros_like(m.bias)
        
        # Apply initialization to all linear layers
        self.apply(init_linear)
    
    def forward(self, x: mx.array, edge_index: mx.array, edge_types: mx.array) -> mx.array:
        """
        Forward pass through the model.
        
        Args:
            x: Node features [num_nodes, input_features]
            edge_index: Edge indices [2, num_edges]
            edge_types: Edge types [num_edges]
        
        Returns:
            Node embeddings [num_nodes, hidden_dim]
        """
        # Input projection
        h = self.input_proj(x)
        h = nn.relu(h)
        
        # Apply GAT layers
        for gat_layer in self.gat_layers:
            h = gat_layer(h, edge_index, edge_types)
        
        # Final node embeddings
        node_embeddings = self.node_embedding(h)
        
        return node_embeddings
    
    def predict_parents(self, node_embeddings: mx.array, 
                       candidate_pairs: mx.array) -> mx.array:
        """
        Predict parent-child relationships.
        
        Args:
            node_embeddings: Node embeddings [num_nodes, hidden_dim]
            candidate_pairs: Candidate parent-child pairs [num_pairs, 2]
        
        Returns:
            Prediction scores [num_pairs]
        """
        if candidate_pairs.shape[0] == 0:
            return mx.array([])
        
        # Get embeddings for parent and child candidates
        parent_embeddings = node_embeddings[candidate_pairs[:, 0]]
        child_embeddings = node_embeddings[candidate_pairs[:, 1]]
        
        # Concatenate embeddings
        pair_embeddings = mx.concatenate([parent_embeddings, child_embeddings], axis=1)
        
        # Predict relationship scores
        scores = self.parent_predictor(pair_embeddings)
        scores = mx.squeeze(scores, axis=-1)  # Remove last dimension
        
        return scores
    
    def __call__(self, x: mx.array, edge_index: mx.array, edge_types: mx.array,
                 candidate_pairs: Optional[mx.array] = None) -> mx.array:
        """
        Full forward pass.
        
        Args:
            x: Node features
            edge_index: Edge indices
            edge_types: Edge types
            candidate_pairs: Optional candidate pairs for prediction
        
        Returns:
            Node embeddings if candidate_pairs is None, else prediction scores
        """
        node_embeddings = self.forward(x, edge_index, edge_types)
        
        if candidate_pairs is not None:
            return self.predict_parents(node_embeddings, candidate_pairs)
        
        return node_embeddings

class ContrastiveLoss(nn.Module):
    """Contrastive loss for parent prediction training."""
    
    def __init__(self, margin: float = 1.0, temperature: float = 0.1):
        super().__init__()
        self.margin = margin
        self.temperature = temperature
    
    def __call__(self, positive_scores: mx.array, negative_scores: mx.array) -> mx.array:
        """
        Compute contrastive loss.
        
        Args:
            positive_scores: Scores for positive (true parent-child) pairs
            negative_scores: Scores for negative (non-parent-child) pairs
        
        Returns:
            Contrastive loss value
        """
        # Sigmoid activation for scores
        pos_probs = mx.sigmoid(positive_scores / self.temperature)
        neg_probs = mx.sigmoid(negative_scores / self.temperature)
        
        # Binary cross-entropy loss
        pos_loss = -mx.log(pos_probs + 1e-8)
        neg_loss = -mx.log(1 - neg_probs + 1e-8)
        
        # Combine losses
        total_loss = mx.mean(pos_loss) + mx.mean(neg_loss)
        
        return total_loss

def create_model(config: dict) -> OhanaAIModel:
    """Create OhanaAI model from configuration."""
    return OhanaAIModel(
        input_features=config['model']['node_features'],
        hidden_dim=config['model']['hidden_dim'],
        num_heads=config['model']['num_heads'],
        num_layers=config['model']['num_layers'],
        num_edge_types=config['model']['edge_types'],
        dropout=config['model']['dropout']
    )