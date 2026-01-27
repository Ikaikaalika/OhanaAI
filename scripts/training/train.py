#!/usr/bin/env python3
"""
OhanaAI - Training Pipeline for Missing Relative Prediction

This script handles:
1. Loading and preprocessing GEDCOM training data
2. Data augmentation for genealogical graphs
3. Training the GNN model with multi-task learning
4. Validation and metrics tracking
5. Model checkpointing and export
"""

import argparse
import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import random

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW, Lion
import numpy as np
from sklearn.model_selection import train_test_split

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent))
from model import (
    FamilyTreeGNN, ModelConfig, create_model, count_parameters,
    multi_task_loss, binary_cross_entropy, focal_loss, attribute_loss
)


# ============================================================================
# Data Loading
# ============================================================================

class GedcomDataset:
    """Dataset for GEDCOM training data."""

    def __init__(self, data_dir: Path, config: ModelConfig):
        self.data_dir = data_dir
        self.config = config
        self.examples: List[Dict[str, Any]] = []

        self._load_data()

    def _load_data(self):
        """Load all training batch files."""
        files = sorted(self.data_dir.glob('training_*.json'))
        if not files:
            files = sorted(self.data_dir.glob('*.json'))

        print(f"Found {len(files)} training data files")

        for file_path in files:
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)

                # Handle different data formats
                if 'data' in data:
                    # New format: {"metadata": {...}, "data": [{nodeFeatures, ...}]}
                    for item in data['data']:
                        if 'nodeFeatures' in item:
                            # Direct format with nodeFeatures
                            example = self._process_direct_example(item)
                        elif 'graphData' in item:
                            # Old format with graphData
                            example = self._process_example(item)
                        else:
                            continue
                        if example is not None:
                            self.examples.append(example)
                elif 'nodeFeatures' in data:
                    # Direct training example format
                    example = self._process_direct_example(data)
                    if example is not None:
                        self.examples.append(example)

            except Exception as e:
                print(f"Error loading {file_path}: {e}")
                import traceback
                traceback.print_exc()

        print(f"Loaded {len(self.examples)} training examples")

    def _process_example(self, item: Dict) -> Optional[Dict]:
        """Process a single training example from batch format."""
        graph_data = item.get('graphData', {})
        labels_data = item.get('labels', [])

        nodes = graph_data.get('nodes', [])
        edges = graph_data.get('edges', [])

        if len(nodes) < 3:
            return None

        # Build node features matrix
        node_id_to_idx = {node['id']: i for i, node in enumerate(nodes)}
        node_features = []

        for node in nodes:
            features = node.get('features', [])
            # Pad/truncate to expected dimension
            features = self._pad_features(features, self.config.node_feature_dim)
            node_features.append(features)

        # Build edge index and features
        edge_index = [[], []]
        edge_features = []
        edge_types = []

        type_to_idx = {'parent': 0, 'child': 1, 'spouse': 2, 'sibling': 3}

        for edge in edges:
            src_idx = node_id_to_idx.get(edge.get('source'))
            tgt_idx = node_id_to_idx.get(edge.get('target'))

            if src_idx is not None and tgt_idx is not None:
                edge_index[0].append(src_idx)
                edge_index[1].append(tgt_idx)

                # Create edge features
                edge_type = edge.get('type', 'parent')
                weight = edge.get('weight', 1.0)
                edge_feat = [0] * 4
                edge_feat[type_to_idx.get(edge_type, 0)] = 1
                edge_feat.extend([weight, 0, 0, 0])  # Pad to 8 dims
                edge_features.append(edge_feat[:self.config.edge_feature_dim])
                edge_types.append(type_to_idx.get(edge_type, 0))

        # Build labels
        labels = {
            'missing_father': [],
            'missing_mother': [],
            'missing_spouse': [],
            'missing_children': [],
            'missing_siblings': []
        }

        for label in labels_data:
            person_idx = node_id_to_idx.get(label.get('personId'))
            if person_idx is None:
                continue

            has_missing = label.get('hasMissingParent', False)
            missing_type = label.get('missingParentType')

            missing_father = 1 if missing_type in ['father', 'both'] else (1 if has_missing and missing_type is None else 0)
            missing_mother = 1 if missing_type in ['mother', 'both'] else 0

            # Infer other missing relations from attributes
            attrs = label.get('attributes', {})
            missing_attrs = label.get('missingAttributes', {})

            has_spouse = len(attrs.get('spouses', [])) > 0
            missing_spouse = 0 if has_spouse else 1

            # Assume if they have no children recorded, they might have children
            # This is a simplification - in practice we'd need more data
            missing_children = 1 if missing_attrs.get('spouses', True) else 0
            missing_siblings = 1  # Almost always possible to have siblings

            labels['missing_father'].append(missing_father)
            labels['missing_mother'].append(missing_mother)
            labels['missing_spouse'].append(missing_spouse)
            labels['missing_children'].append(missing_children)
            labels['missing_siblings'].append(missing_siblings)

        # Ensure labels match node count
        num_nodes = len(nodes)
        for key in labels:
            while len(labels[key]) < num_nodes:
                labels[key].append(0)
            labels[key] = labels[key][:num_nodes]

        return {
            'node_features': np.array(node_features, dtype=np.float32),
            'edge_index': np.array(edge_index, dtype=np.int32),
            'edge_features': np.array(edge_features, dtype=np.float32) if edge_features else np.zeros((0, self.config.edge_feature_dim), dtype=np.float32),
            'edge_types': np.array(edge_types, dtype=np.int32),
            'labels': {k: np.array(v, dtype=np.float32) for k, v in labels.items()},
            'node_ids': [node['id'] for node in nodes],
            'global_features': np.zeros(self.config.global_feature_dim, dtype=np.float32)
        }

    def _process_direct_example(self, data: Dict) -> Optional[Dict]:
        """Process a direct training example format."""
        node_features = np.array(data['nodeFeatures'], dtype=np.float32)

        # Handle feature dimension mismatch (old 176 vs new 224)
        if node_features.shape[1] < self.config.node_feature_dim:
            padding = np.zeros((node_features.shape[0], self.config.node_feature_dim - node_features.shape[1]), dtype=np.float32)
            node_features = np.concatenate([node_features, padding], axis=1)
        elif node_features.shape[1] > self.config.node_feature_dim:
            node_features = node_features[:, :self.config.node_feature_dim]

        edge_index = np.array(data['edgeIndex'], dtype=np.int32)
        edge_features = np.array(data.get('edgeFeatures', []), dtype=np.float32)
        edge_types = np.array(data.get('edgeTypes', []), dtype=np.int32)

        labels = {}
        for key in ['missingFather', 'missingMother', 'missingSpouse', 'missingChildren', 'missingSiblings']:
            snake_key = ''.join(['_' + c.lower() if c.isupper() else c for c in key]).lstrip('_')
            labels[snake_key] = np.array(data['labels'].get(key, []), dtype=np.float32)

        # Process attribute labels if present
        attribute_labels = {}
        if 'attributeLabels' in data:
            attr_data = data['attributeLabels']
            for key in ['fatherBirthYear', 'motherBirthYear']:
                if key in attr_data:
                    attribute_labels[key] = np.array(attr_data[key], dtype=np.float32)
            for key in ['fatherEthnicOrigin', 'motherEthnicOrigin', 'parentLocation']:
                if key in attr_data:
                    attribute_labels[key] = np.array(attr_data[key], dtype=np.float32)

        # Process pattern stats if present
        pattern_stats = data.get('patternStats', {})

        return {
            'node_features': node_features,
            'edge_index': edge_index,
            'edge_features': edge_features if edge_features.size > 0 else np.zeros((edge_index.shape[1], self.config.edge_feature_dim), dtype=np.float32),
            'edge_types': edge_types,
            'labels': labels,
            'attribute_labels': attribute_labels,
            'pattern_stats': pattern_stats,
            'node_ids': data.get('nodeIds', []),
            'global_features': np.array(data.get('globalFeatures', [0] * self.config.global_feature_dim), dtype=np.float32)
        }

    def _pad_features(self, features: List[float], target_dim: int) -> List[float]:
        """Pad or truncate features to target dimension."""
        features = list(features)
        if len(features) < target_dim:
            features.extend([0.0] * (target_dim - len(features)))
        return features[:target_dim]

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> Dict:
        return self.examples[idx]


# ============================================================================
# Data Augmentation
# ============================================================================

class GenealogyAugmenter:
    """Data augmentation for genealogical graphs."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def augment(self, example: Dict, augmentation_prob: float = 0.3) -> Dict:
        """Apply random augmentations to a training example."""
        if random.random() > augmentation_prob:
            return example

        augmented = {
            'node_features': example['node_features'].copy(),
            'edge_index': example['edge_index'].copy(),
            'edge_features': example['edge_features'].copy(),
            'edge_types': example['edge_types'].copy(),
            'labels': {k: v.copy() for k, v in example['labels'].items()},
            'node_ids': example['node_ids'].copy(),
            'global_features': example['global_features'].copy()
        }

        # Apply augmentations
        augmentations = [
            self._drop_edges,
            self._mask_features,
            self._add_noise,
            self._simulate_missing_data
        ]

        # Apply 1-2 random augmentations
        num_augs = random.randint(1, 2)
        selected_augs = random.sample(augmentations, num_augs)

        for aug_fn in selected_augs:
            augmented = aug_fn(augmented)

        return augmented

    def _drop_edges(self, example: Dict, drop_rate: float = 0.1) -> Dict:
        """Randomly drop some edges."""
        edge_index = example['edge_index']
        edge_features = example['edge_features']
        edge_types = example['edge_types']

        num_edges = edge_index.shape[1]
        if num_edges == 0:
            return example

        # Keep edges with probability (1 - drop_rate)
        keep_mask = np.random.random(num_edges) > drop_rate
        keep_indices = np.where(keep_mask)[0]

        if len(keep_indices) < 2:
            return example  # Don't drop too many edges

        example['edge_index'] = edge_index[:, keep_indices]
        example['edge_features'] = edge_features[keep_indices]
        example['edge_types'] = edge_types[keep_indices]

        return example

    def _mask_features(self, example: Dict, mask_rate: float = 0.15) -> Dict:
        """Randomly mask some node features."""
        node_features = example['node_features']

        mask = np.random.random(node_features.shape) < mask_rate
        node_features[mask] = 0

        example['node_features'] = node_features
        return example

    def _add_noise(self, example: Dict, noise_scale: float = 0.05) -> Dict:
        """Add Gaussian noise to continuous features."""
        node_features = example['node_features']

        # Only add noise to non-binary features
        # Identify binary features (values are 0 or 1)
        is_binary = np.all((node_features == 0) | (node_features == 1), axis=0)
        continuous_mask = ~is_binary

        noise = np.random.normal(0, noise_scale, node_features.shape)
        noise[:, ~continuous_mask] = 0

        node_features = node_features + noise
        node_features = np.clip(node_features, 0, 1)  # Keep in valid range

        example['node_features'] = node_features.astype(np.float32)
        return example

    def _simulate_missing_data(self, example: Dict, missing_rate: float = 0.1) -> Dict:
        """Simulate additional missing data to improve robustness."""
        node_features = example['node_features']
        labels = example['labels']

        num_nodes = node_features.shape[0]
        nodes_to_modify = np.random.random(num_nodes) < missing_rate

        for i in np.where(nodes_to_modify)[0]:
            # Randomly decide which parent to "remove"
            if random.random() < 0.5:
                # Simulate missing father
                labels['missing_father'][i] = 1
            else:
                # Simulate missing mother
                labels['missing_mother'][i] = 1

        example['labels'] = labels
        return example


# ============================================================================
# Training Loop
# ============================================================================

class Trainer:
    """Training orchestrator for the GNN model."""

    def __init__(
        self,
        model: FamilyTreeGNN,
        train_dataset: GedcomDataset,
        val_dataset: GedcomDataset,
        config: Dict[str, Any]
    ):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.config = config

        # Optimizer
        if config.get('optimizer', 'adamw') == 'lion':
            self.optimizer = Lion(learning_rate=config['learning_rate'])
        else:
            self.optimizer = AdamW(
                learning_rate=config['learning_rate'],
                weight_decay=config.get('weight_decay', 0.01)
            )

        # Augmenter
        self.augmenter = GenealogyAugmenter(model.config)

        # Metrics tracking
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        self.best_val_loss = float('inf')
        self.epochs_without_improvement = 0

        # Loss function configuration
        self.use_focal_loss = config.get('use_focal_loss', True)
        self.task_weights = config.get('task_weights', {
            'missing_father': 1.0,
            'missing_mother': 1.0,
            'missing_spouse': 0.8,
            'missing_children': 0.5,
            'missing_siblings': 0.3
        })

        # Attribute prediction configuration
        self.train_attributes = config.get('train_attributes', True)
        self.attribute_weights = config.get('attribute_weights', {
            'birth_year': 1.0,
            'ethnic_origin': 0.5,
            'location': 0.3
        })
        self.attribute_loss_weight = config.get('attribute_loss_weight', 0.5)

    def train_epoch(self, epoch: int) -> float:
        """Train for one epoch."""
        total_loss = 0.0
        num_batches = 0

        # Shuffle examples
        indices = list(range(len(self.train_dataset)))
        random.shuffle(indices)

        # Define loss function that captures self
        def compute_loss(model, example):
            # Unpack example
            node_features = mx.array(example['node_features'])
            edge_index = mx.array(example['edge_index'])
            edge_features = mx.array(example['edge_features'])
            global_features = mx.array(example['global_features'])

            # Forward pass with attribute prediction if enabled
            predict_attrs = self.train_attributes and 'attribute_labels' in example and len(example['attribute_labels']) > 0
            _, predictions, attr_predictions = model(
                node_features, edge_index, edge_features, global_features,
                predict_attrs=predict_attrs
            )

            # Compute multi-task loss for missing relative prediction
            loss, _ = multi_task_loss(
                predictions,
                example['labels'],
                weights=self.task_weights,
                use_focal=self.use_focal_loss
            )

            # Add attribute loss if training attributes
            if predict_attrs and attr_predictions is not None:
                attr_loss, _ = attribute_loss(
                    attr_predictions,
                    example['attribute_labels'],
                    example['labels'],  # Use missing labels as mask
                    weights=self.attribute_weights
                )
                loss = loss + self.attribute_loss_weight * attr_loss

            return loss

        # Get loss and gradient function
        loss_and_grad = mx.value_and_grad(compute_loss)

        for idx in indices:
            example = self.train_dataset[idx]

            # Apply augmentation
            if self.config.get('augmentation', True):
                example = self.augmenter.augment(example)

            # Skip if graph is too small
            if example['node_features'].shape[0] < 3:
                continue

            # Compute loss and gradients
            loss, grads = loss_and_grad(self.model, example)

            # Update model
            self.optimizer.update(self.model, grads)
            mx.eval(self.model.parameters(), self.optimizer.state)

            total_loss += float(loss)
            num_batches += 1

        avg_loss = total_loss / max(num_batches, 1)
        self.train_losses.append(avg_loss)
        return avg_loss

    def validate(self) -> Tuple[float, Dict[str, float]]:
        """Run validation."""
        total_loss = 0.0
        task_losses = {
            'missing_father': 0.0,
            'missing_mother': 0.0,
            'missing_spouse': 0.0,
            'missing_children': 0.0,
            'missing_siblings': 0.0,
            'attribute_total': 0.0
        }
        num_examples = 0

        for example in self.val_dataset.examples:
            if example['node_features'].shape[0] < 3:
                continue

            node_features = mx.array(example['node_features'])
            edge_index = mx.array(example['edge_index'])
            edge_features = mx.array(example['edge_features'])
            global_features = mx.array(example['global_features'])

            predict_attrs = self.train_attributes and 'attribute_labels' in example and len(example.get('attribute_labels', {})) > 0
            _, predictions, attr_predictions = self.model(
                node_features, edge_index, edge_features, global_features,
                predict_attrs=predict_attrs
            )

            loss, batch_task_losses = multi_task_loss(
                predictions,
                example['labels'],
                weights=self.task_weights,
                use_focal=self.use_focal_loss
            )

            # Add attribute loss
            if predict_attrs and attr_predictions is not None:
                attr_loss, attr_task_losses = attribute_loss(
                    attr_predictions,
                    example['attribute_labels'],
                    example['labels'],
                    weights=self.attribute_weights
                )
                mx.eval(attr_loss)
                task_losses['attribute_total'] += float(attr_loss)
                loss = loss + self.attribute_loss_weight * attr_loss

            mx.eval(loss)
            total_loss += float(loss)

            for key, val in batch_task_losses.items():
                mx.eval(val)
                task_losses[key] += float(val)

            num_examples += 1

        avg_loss = total_loss / max(num_examples, 1)
        avg_task_losses = {k: v / max(num_examples, 1) for k, v in task_losses.items()}

        self.val_losses.append(avg_loss)
        return avg_loss, avg_task_losses

    def train(self, num_epochs: int, save_dir: Path):
        """Full training loop."""
        save_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nStarting training for {num_epochs} epochs")
        print(f"Model parameters: {count_parameters(self.model):,}")
        print(f"Training examples: {len(self.train_dataset)}")
        print(f"Validation examples: {len(self.val_dataset)}")
        print()

        for epoch in range(num_epochs):
            # Train
            train_loss = self.train_epoch(epoch)

            # Validate
            val_loss, task_losses = self.validate()

            # Print progress
            print(f"Epoch {epoch + 1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_loss:.4f}")
            print(f"  Task Losses: father={task_losses['missing_father']:.4f}, "
                  f"mother={task_losses['missing_mother']:.4f}, "
                  f"spouse={task_losses['missing_spouse']:.4f}")

            # Check for improvement
            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.epochs_without_improvement = 0

                # Save best model
                self._save_checkpoint(save_dir / 'best_model.safetensors', epoch, val_loss)
                print(f"  → New best model saved!")
            else:
                self.epochs_without_improvement += 1

            # Early stopping
            patience = self.config.get('patience', 10)
            if self.epochs_without_improvement >= patience:
                print(f"\nEarly stopping after {patience} epochs without improvement")
                break

            # Learning rate scheduling
            if (epoch + 1) % self.config.get('lr_decay_epochs', 20) == 0:
                current_lr = self.optimizer.learning_rate
                new_lr = current_lr * self.config.get('lr_decay_factor', 0.5)
                self.optimizer.learning_rate = new_lr
                print(f"  → Learning rate reduced to {new_lr:.6f}")

        # Save final model
        self._save_checkpoint(save_dir / 'final_model.safetensors', num_epochs, val_loss)
        print(f"\nTraining complete. Best validation loss: {self.best_val_loss:.4f}")

        # Save training history
        self._save_history(save_dir / 'training_history.json')

    def _save_checkpoint(self, path: Path, epoch: int, val_loss: float):
        """Save model checkpoint."""
        # Save model weights
        mx.savez(
            str(path).replace('.safetensors', '.npz'),
            **{k: np.array(v) for k, v in self._flatten_params(self.model.parameters()).items()}
        )

        # Save metadata
        metadata = {
            'epoch': epoch,
            'val_loss': val_loss,
            'config': {
                'node_feature_dim': self.model.config.node_feature_dim,
                'edge_feature_dim': self.model.config.edge_feature_dim,
                'hidden_dim': self.model.config.hidden_dim,
                'num_gnn_layers': self.model.config.num_gnn_layers,
                'num_attention_heads': self.model.config.num_attention_heads
            },
            'timestamp': datetime.now().isoformat()
        }
        with open(str(path).replace('.safetensors', '_metadata.json'), 'w') as f:
            json.dump(metadata, f, indent=2)

    def _flatten_params(self, params: Dict, prefix: str = '') -> Dict:
        """Flatten nested parameter dictionary."""
        flat = {}
        for key, value in params.items():
            full_key = f"{prefix}.{key}" if prefix else key
            if isinstance(value, dict):
                flat.update(self._flatten_params(value, full_key))
            elif isinstance(value, mx.array):
                flat[full_key] = value
        return flat

    def _save_history(self, path: Path):
        """Save training history."""
        history = {
            'train_losses': self.train_losses,
            'val_losses': self.val_losses,
            'best_val_loss': self.best_val_loss,
            'config': self.config
        }
        with open(path, 'w') as f:
            json.dump(history, f, indent=2)


# ============================================================================
# Main Entry Point
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train OhanaAI GNN model')
    parser.add_argument('--data-dir', type=str, default='training_data',
                        help='Directory containing training data')
    parser.add_argument('--output-dir', type=str, default='models/family_tree_gnn',
                        help='Directory to save model')
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--lr', type=float, default=1e-3,
                        help='Learning rate')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension size')
    parser.add_argument('--num-layers', type=int, default=3,
                        help='Number of GNN layers')
    parser.add_argument('--num-heads', type=int, default=4,
                        help='Number of attention heads')
    parser.add_argument('--dropout', type=float, default=0.1,
                        help='Dropout rate')
    parser.add_argument('--val-split', type=float, default=0.2,
                        help='Validation split ratio')
    parser.add_argument('--patience', type=int, default=15,
                        help='Early stopping patience')
    parser.add_argument('--no-augmentation', action='store_true',
                        help='Disable data augmentation')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')

    args = parser.parse_args()

    # Set random seeds
    random.seed(args.seed)
    np.random.seed(args.seed)

    # Create model configuration
    model_config = ModelConfig()
    model_config.hidden_dim = args.hidden_dim
    model_config.num_gnn_layers = args.num_layers
    model_config.num_attention_heads = args.num_heads
    model_config.dropout_rate = args.dropout

    # Load data
    data_dir = Path(args.data_dir)
    if not data_dir.exists():
        print(f"Error: Data directory {data_dir} does not exist")
        sys.exit(1)

    print("Loading training data...")
    full_dataset = GedcomDataset(data_dir, model_config)

    if len(full_dataset) == 0:
        print("Error: No valid training examples found")
        sys.exit(1)

    # Split into train/val
    # If we only have one example, use it for both train and val
    if len(full_dataset) == 1:
        print("Single file mode: using same data for train and validation")
        train_dataset = full_dataset
        val_dataset = full_dataset
    else:
        indices = list(range(len(full_dataset)))
        train_indices, val_indices = train_test_split(
            indices, test_size=args.val_split, random_state=args.seed
        )

        # Create train and val datasets
        train_dataset = GedcomDataset.__new__(GedcomDataset)
        train_dataset.config = model_config
        train_dataset.examples = [full_dataset[i] for i in train_indices]

        val_dataset = GedcomDataset.__new__(GedcomDataset)
        val_dataset.config = model_config
        val_dataset.examples = [full_dataset[i] for i in val_indices]

    # Create model
    print("\nInitializing model...")
    model = create_model(model_config)
    print(f"Model architecture: {args.num_layers} GNN layers, "
          f"{args.hidden_dim} hidden dim, {args.num_heads} attention heads")

    # Training configuration
    train_config = {
        'learning_rate': args.lr,
        'weight_decay': 0.01,
        'patience': args.patience,
        'augmentation': not args.no_augmentation,
        'use_focal_loss': True,
        'lr_decay_epochs': 20,
        'lr_decay_factor': 0.5,
        'task_weights': {
            'missing_father': 1.0,
            'missing_mother': 1.0,
            'missing_spouse': 0.8,
            'missing_children': 0.5,
            'missing_siblings': 0.3
        },
        # Attribute generation training
        'train_attributes': True,
        'attribute_loss_weight': 0.5,
        'attribute_weights': {
            'birth_year': 1.0,
            'ethnic_origin': 0.5,
            'location': 0.3
        }
    }

    # Create trainer and run training
    trainer = Trainer(model, train_dataset, val_dataset, train_config)
    trainer.train(args.epochs, Path(args.output_dir))

    print("\nDone!")


if __name__ == '__main__':
    main()
