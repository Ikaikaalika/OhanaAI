"""
OhanaAI - Graph Neural Network for Missing Relative Prediction

This module implements a Graph Attention Network (GAT) based model for predicting
missing relatives in family trees. The model:
1. Processes node features through attention-based message passing
2. Uses multi-task learning to predict different types of missing relatives
3. Generates candidate rankings for potential relatives
"""

import mlx.core as mx
import mlx.nn as nn
from typing import Tuple, List, Optional, Dict
import math


# ============================================================================
# Model Configuration
# ============================================================================

class ModelConfig:
    """Configuration for the GNN model"""
    # Input dimensions (must match feature extraction)
    node_feature_dim: int = 176
    edge_feature_dim: int = 8
    global_feature_dim: int = 7

    # GNN architecture
    hidden_dim: int = 128
    num_gnn_layers: int = 3
    num_attention_heads: int = 4
    dropout_rate: float = 0.1

    # Prediction heads
    num_relation_types: int = 5  # father, mother, spouse, child, sibling
    candidate_embedding_dim: int = 64

    # Training
    use_edge_features: bool = True
    use_global_features: bool = True


# ============================================================================
# Graph Attention Layer
# ============================================================================

class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT) with edge features.

    Computes attention-weighted message passing between nodes.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        edge_dim: int = 8,
        use_edge_features: bool = True,
        concat: bool = True,
        dropout: float = 0.1
    ):
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.edge_dim = edge_dim
        self.use_edge_features = use_edge_features
        self.concat = concat
        self.head_dim = out_features // num_heads

        # Linear transformations for queries, keys, values
        self.W_q = nn.Linear(in_features, out_features)
        self.W_k = nn.Linear(in_features, out_features)
        self.W_v = nn.Linear(in_features, out_features)

        # Edge feature projection
        if use_edge_features:
            self.W_e = nn.Linear(edge_dim, num_heads)

        # Attention parameters
        self.a = mx.random.normal((num_heads, 2 * self.head_dim))

        # Output projection
        if concat:
            self.W_o = nn.Linear(out_features, out_features)
        else:
            self.W_o = nn.Linear(self.head_dim, out_features)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(out_features)

    def __call__(
        self,
        x: mx.array,           # [num_nodes, in_features]
        edge_index: mx.array,  # [2, num_edges]
        edge_features: Optional[mx.array] = None  # [num_edges, edge_dim]
    ) -> mx.array:

        num_nodes = x.shape[0]
        num_edges = edge_index.shape[1]

        # Compute Q, K, V for all nodes
        Q = self.W_q(x)  # [num_nodes, out_features]
        K = self.W_k(x)
        V = self.W_v(x)

        # Reshape for multi-head attention
        Q = Q.reshape(num_nodes, self.num_heads, self.head_dim)
        K = K.reshape(num_nodes, self.num_heads, self.head_dim)
        V = V.reshape(num_nodes, self.num_heads, self.head_dim)

        # Get source and target indices
        src_idx = edge_index[0]  # [num_edges]
        tgt_idx = edge_index[1]  # [num_edges]

        # Gather Q, K, V for edges
        Q_src = Q[src_idx]  # [num_edges, num_heads, head_dim]
        K_tgt = K[tgt_idx]  # [num_edges, num_heads, head_dim]
        V_tgt = V[tgt_idx]  # [num_edges, num_heads, head_dim]

        # Compute attention scores
        # Concatenate Q and K, then dot with attention vector
        QK_cat = mx.concatenate([Q_src, K_tgt], axis=-1)  # [num_edges, num_heads, 2*head_dim]
        attn_scores = mx.sum(QK_cat * self.a, axis=-1)  # [num_edges, num_heads]

        # Add edge feature bias if available
        if self.use_edge_features and edge_features is not None:
            edge_bias = self.W_e(edge_features)  # [num_edges, num_heads]
            attn_scores = attn_scores + edge_bias

        # Apply LeakyReLU and scale
        attn_scores = nn.leaky_relu(attn_scores, negative_slope=0.2)
        attn_scores = attn_scores / math.sqrt(self.head_dim)

        # Softmax over incoming edges per target node
        # We need to normalize per target node
        attn_weights = self._sparse_softmax(attn_scores, tgt_idx, num_nodes)
        attn_weights = self.dropout(attn_weights)

        # Weighted sum of values
        weighted_V = V_tgt * attn_weights[:, :, None]  # [num_edges, num_heads, head_dim]

        # Aggregate to target nodes
        out = mx.zeros((num_nodes, self.num_heads, self.head_dim))
        # Scatter-add weighted values to target nodes
        out = self._scatter_add(weighted_V, tgt_idx, num_nodes)

        # Reshape and project
        if self.concat:
            out = out.reshape(num_nodes, -1)  # [num_nodes, out_features]
        else:
            out = mx.mean(out, axis=1)  # [num_nodes, head_dim]

        out = self.W_o(out)

        # Residual connection and layer norm
        if x.shape[-1] == out.shape[-1]:
            out = self.layer_norm(out + x)
        else:
            out = self.layer_norm(out)

        return out

    def _sparse_softmax(
        self,
        scores: mx.array,  # [num_edges, num_heads]
        target_indices: mx.array,  # [num_edges]
        num_nodes: int
    ) -> mx.array:
        """Compute softmax over edges grouped by target node"""
        # For each target node, compute softmax over all incoming edges

        # Get max score per target for numerical stability
        max_scores = mx.zeros((num_nodes, scores.shape[1]))
        for i in range(scores.shape[0]):
            tgt = int(target_indices[i])
            max_scores = max_scores.at[tgt].maximum(scores[i])

        # Subtract max and exponentiate
        scores_shifted = scores - max_scores[target_indices]
        exp_scores = mx.exp(scores_shifted)

        # Sum exp scores per target
        sum_exp = mx.zeros((num_nodes, scores.shape[1]))
        for i in range(scores.shape[0]):
            tgt = int(target_indices[i])
            sum_exp = sum_exp.at[tgt].add(exp_scores[i])

        # Normalize
        return exp_scores / (sum_exp[target_indices] + 1e-10)

    def _scatter_add(
        self,
        src: mx.array,  # [num_edges, num_heads, head_dim]
        index: mx.array,  # [num_edges]
        num_nodes: int
    ) -> mx.array:
        """Scatter-add source values to indexed positions"""
        out = mx.zeros((num_nodes, src.shape[1], src.shape[2]))
        for i in range(src.shape[0]):
            tgt = int(index[i])
            out = out.at[tgt].add(src[i])
        return out


# ============================================================================
# Graph Neural Network Encoder
# ============================================================================

class GNNEncoder(nn.Module):
    """
    Multi-layer Graph Neural Network encoder.

    Takes node features and graph structure, produces node embeddings
    that capture both local and global graph context.
    """

    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config

        # Input projection
        self.input_proj = nn.Linear(config.node_feature_dim, config.hidden_dim)

        # GNN layers
        self.gnn_layers = []
        for i in range(config.num_gnn_layers):
            layer = GraphAttentionLayer(
                in_features=config.hidden_dim,
                out_features=config.hidden_dim,
                num_heads=config.num_attention_heads,
                edge_dim=config.edge_feature_dim,
                use_edge_features=config.use_edge_features,
                concat=True,
                dropout=config.dropout_rate
            )
            self.gnn_layers.append(layer)

        # Global feature integration
        if config.use_global_features:
            self.global_proj = nn.Linear(
                config.global_feature_dim,
                config.hidden_dim
            )

        # Final layer norm
        self.final_norm = nn.LayerNorm(config.hidden_dim)

    def __call__(
        self,
        node_features: mx.array,
        edge_index: mx.array,
        edge_features: Optional[mx.array] = None,
        global_features: Optional[mx.array] = None
    ) -> mx.array:

        # Project input features
        x = self.input_proj(node_features)
        x = nn.relu(x)

        # Apply GNN layers
        for layer in self.gnn_layers:
            x = layer(x, edge_index, edge_features)
            x = nn.relu(x)

        # Integrate global features
        if self.config.use_global_features and global_features is not None:
            global_emb = self.global_proj(global_features)  # [global_feature_dim] -> [hidden_dim]
            # Broadcast and add to all nodes
            x = x + global_emb

        x = self.final_norm(x)

        return x


# ============================================================================
# Multi-Task Prediction Heads
# ============================================================================

class MissingRelativePredictionHead(nn.Module):
    """
    Predicts whether each node is missing a specific type of relative.

    Output: probability for each node that they're missing this relation type.
    """

    def __init__(self, hidden_dim: int, dropout: float = 0.1):
        super().__init__()

        self.fc1 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc2 = nn.Linear(hidden_dim // 2, 1)
        self.dropout = nn.Dropout(dropout)

    def __call__(self, node_embeddings: mx.array) -> mx.array:
        x = self.fc1(node_embeddings)
        x = nn.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return nn.sigmoid(x).squeeze(-1)


class CandidateRankingHead(nn.Module):
    """
    Ranks candidate nodes as potential relatives.

    For a given query node missing a relation, scores all other nodes
    as potential matches.
    """

    def __init__(self, hidden_dim: int, candidate_dim: int):
        super().__init__()

        # Project query node
        self.query_proj = nn.Linear(hidden_dim, candidate_dim)

        # Project candidate nodes
        self.candidate_proj = nn.Linear(hidden_dim, candidate_dim)

        # Bilinear scoring
        self.W_score = mx.random.normal((candidate_dim, candidate_dim)) * 0.1

    def __call__(
        self,
        query_embedding: mx.array,     # [hidden_dim] or [batch, hidden_dim]
        candidate_embeddings: mx.array  # [num_candidates, hidden_dim]
    ) -> mx.array:
        """
        Returns scores for each candidate being the missing relative.
        """
        # Project
        q = self.query_proj(query_embedding)  # [candidate_dim] or [batch, candidate_dim]
        c = self.candidate_proj(candidate_embeddings)  # [num_candidates, candidate_dim]

        # Bilinear score: q^T W c
        # For batched queries
        if len(q.shape) == 1:
            q = q[None, :]  # [1, candidate_dim]

        scores = q @ self.W_score @ c.T  # [batch, num_candidates]

        return scores.squeeze(0)


# ============================================================================
# Full Model
# ============================================================================

class FamilyTreeGNN(nn.Module):
    """
    Complete model for predicting missing relatives in family trees.

    Multi-task outputs:
    1. Missing father prediction (binary per node)
    2. Missing mother prediction (binary per node)
    3. Missing spouse prediction (binary per node)
    4. Missing children prediction (binary per node)
    5. Missing siblings prediction (binary per node)
    6. Candidate ranking for each relation type
    """

    def __init__(self, config: Optional[ModelConfig] = None):
        super().__init__()

        self.config = config or ModelConfig()

        # GNN encoder
        self.encoder = GNNEncoder(self.config)

        # Prediction heads for each relation type
        self.missing_father_head = MissingRelativePredictionHead(
            self.config.hidden_dim, self.config.dropout_rate
        )
        self.missing_mother_head = MissingRelativePredictionHead(
            self.config.hidden_dim, self.config.dropout_rate
        )
        self.missing_spouse_head = MissingRelativePredictionHead(
            self.config.hidden_dim, self.config.dropout_rate
        )
        self.missing_children_head = MissingRelativePredictionHead(
            self.config.hidden_dim, self.config.dropout_rate
        )
        self.missing_siblings_head = MissingRelativePredictionHead(
            self.config.hidden_dim, self.config.dropout_rate
        )

        # Candidate ranking heads
        self.father_ranking_head = CandidateRankingHead(
            self.config.hidden_dim, self.config.candidate_embedding_dim
        )
        self.mother_ranking_head = CandidateRankingHead(
            self.config.hidden_dim, self.config.candidate_embedding_dim
        )
        self.spouse_ranking_head = CandidateRankingHead(
            self.config.hidden_dim, self.config.candidate_embedding_dim
        )

    def encode(
        self,
        node_features: mx.array,
        edge_index: mx.array,
        edge_features: Optional[mx.array] = None,
        global_features: Optional[mx.array] = None
    ) -> mx.array:
        """Encode the graph to get node embeddings."""
        return self.encoder(node_features, edge_index, edge_features, global_features)

    def predict_missing(
        self,
        node_embeddings: mx.array
    ) -> Dict[str, mx.array]:
        """Predict missing relative probabilities for all nodes."""
        return {
            'missing_father': self.missing_father_head(node_embeddings),
            'missing_mother': self.missing_mother_head(node_embeddings),
            'missing_spouse': self.missing_spouse_head(node_embeddings),
            'missing_children': self.missing_children_head(node_embeddings),
            'missing_siblings': self.missing_siblings_head(node_embeddings)
        }

    def rank_candidates(
        self,
        query_embedding: mx.array,
        candidate_embeddings: mx.array,
        relation_type: str
    ) -> mx.array:
        """Rank candidates for a specific relation type."""
        if relation_type == 'father':
            return self.father_ranking_head(query_embedding, candidate_embeddings)
        elif relation_type == 'mother':
            return self.mother_ranking_head(query_embedding, candidate_embeddings)
        elif relation_type == 'spouse':
            return self.spouse_ranking_head(query_embedding, candidate_embeddings)
        else:
            raise ValueError(f"Unknown relation type: {relation_type}")

    def __call__(
        self,
        node_features: mx.array,
        edge_index: mx.array,
        edge_features: Optional[mx.array] = None,
        global_features: Optional[mx.array] = None
    ) -> Tuple[mx.array, Dict[str, mx.array]]:
        """
        Full forward pass.

        Returns:
            node_embeddings: [num_nodes, hidden_dim]
            predictions: Dict with missing predictions for each relation type
        """
        # Encode
        node_embeddings = self.encode(
            node_features, edge_index, edge_features, global_features
        )

        # Predict missing relations
        predictions = self.predict_missing(node_embeddings)

        return node_embeddings, predictions


# ============================================================================
# Loss Functions
# ============================================================================

def binary_cross_entropy(pred: mx.array, target: mx.array) -> mx.array:
    """Binary cross-entropy loss with numerical stability."""
    eps = 1e-7
    pred = mx.clip(pred, eps, 1 - eps)
    return -mx.mean(target * mx.log(pred) + (1 - target) * mx.log(1 - pred))


def focal_loss(pred: mx.array, target: mx.array, gamma: float = 2.0, alpha: float = 0.25) -> mx.array:
    """
    Focal loss for handling class imbalance.
    Helps when most nodes have complete information.
    """
    eps = 1e-7
    pred = mx.clip(pred, eps, 1 - eps)

    # Compute focal weights
    p_t = target * pred + (1 - target) * (1 - pred)
    focal_weight = (1 - p_t) ** gamma

    # Compute alpha weights
    alpha_t = target * alpha + (1 - target) * (1 - alpha)

    # Compute loss
    ce_loss = -target * mx.log(pred) - (1 - target) * mx.log(1 - pred)
    loss = alpha_t * focal_weight * ce_loss

    return mx.mean(loss)


def multi_task_loss(
    predictions: Dict[str, mx.array],
    labels: Dict[str, mx.array],
    weights: Optional[Dict[str, float]] = None,
    use_focal: bool = True
) -> Tuple[mx.array, Dict[str, mx.array]]:
    """
    Compute combined loss for all prediction tasks.

    Args:
        predictions: Dict of predicted probabilities
        labels: Dict of ground truth labels
        weights: Optional task weights
        use_focal: Whether to use focal loss

    Returns:
        total_loss: Combined weighted loss
        task_losses: Individual task losses
    """
    if weights is None:
        weights = {
            'missing_father': 1.0,
            'missing_mother': 1.0,
            'missing_spouse': 0.8,
            'missing_children': 0.5,
            'missing_siblings': 0.5
        }

    loss_fn = focal_loss if use_focal else binary_cross_entropy

    task_losses = {}
    total_loss = mx.array(0.0)

    for task_name, pred in predictions.items():
        if task_name in labels:
            target = mx.array(labels[task_name])
            task_loss = loss_fn(pred, target)
            task_losses[task_name] = task_loss
            total_loss = total_loss + weights.get(task_name, 1.0) * task_loss

    return total_loss, task_losses


def ranking_loss(
    scores: mx.array,
    positive_idx: int,
    negative_indices: List[int],
    margin: float = 1.0
) -> mx.array:
    """
    Margin ranking loss for candidate ranking.

    Encourages positive candidate to score higher than negatives.
    """
    pos_score = scores[positive_idx]
    neg_scores = scores[mx.array(negative_indices)]

    # Hinge loss: max(0, margin - (pos_score - neg_score))
    losses = mx.maximum(mx.array(0.0), margin - (pos_score - neg_scores))

    return mx.mean(losses)


# ============================================================================
# Utility Functions
# ============================================================================

def create_model(config: Optional[ModelConfig] = None) -> FamilyTreeGNN:
    """Create and initialize the model."""
    return FamilyTreeGNN(config)


def count_parameters(model: nn.Module) -> int:
    """Count the number of trainable parameters."""
    def count_array(arr):
        if isinstance(arr, mx.array):
            return arr.size
        return 0

    total = 0
    for name, param in model.parameters().items():
        if isinstance(param, dict):
            for k, v in param.items():
                total += count_array(v)
        else:
            total += count_array(param)
    return total
