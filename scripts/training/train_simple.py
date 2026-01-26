#!/usr/bin/env python3
"""
OhanaAI - Simple MLP Training for Missing Relative Prediction

A simpler training approach that processes individual nodes without
the full graph attention mechanism, suitable for larger datasets.
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any
import random

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
import numpy as np


# ============================================================================
# Simple MLP Model
# ============================================================================

class SimpleMLP(nn.Module):
    """Simple MLP for per-node prediction."""

    def __init__(self, input_dim: int = 176, hidden_dim: int = 128, num_outputs: int = 5):
        super().__init__()

        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim // 2)
        self.fc3 = nn.Linear(hidden_dim // 2, hidden_dim // 4)
        self.out = nn.Linear(hidden_dim // 4, num_outputs)

        self.dropout = nn.Dropout(0.1)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = self.dropout(x)
        x = nn.relu(self.fc2(x))
        x = self.dropout(x)
        x = nn.relu(self.fc3(x))
        x = nn.sigmoid(self.out(x))
        return x


# ============================================================================
# Data Loading
# ============================================================================

def load_training_data(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load training data and return features and labels."""

    files = sorted(data_dir.glob('training_*.json'))
    if not files:
        files = sorted(data_dir.glob('*.json'))

    print(f"Found {len(files)} training data files")

    all_features = []
    all_labels = []

    for file_path in files:
        try:
            with open(file_path, 'r') as f:
                data = json.load(f)

            # Handle nested format
            if 'data' in data:
                items = data['data']
            else:
                items = [data]

            for item in items:
                if 'nodeFeatures' not in item:
                    continue

                node_features = np.array(item['nodeFeatures'], dtype=np.float32)
                labels = item.get('labels', {})

                # Build label matrix [num_nodes, 5]
                num_nodes = node_features.shape[0]
                label_matrix = np.zeros((num_nodes, 5), dtype=np.float32)

                label_keys = ['missingFather', 'missingMother', 'missingSpouse',
                              'missingChildren', 'missingSiblings']

                for i, key in enumerate(label_keys):
                    if key in labels:
                        label_matrix[:, i] = np.array(labels[key], dtype=np.float32)[:num_nodes]

                all_features.append(node_features)
                all_labels.append(label_matrix)

        except Exception as e:
            print(f"Error loading {file_path}: {e}")

    if not all_features:
        raise ValueError("No valid training data found")

    # Concatenate all examples
    X = np.concatenate(all_features, axis=0)
    y = np.concatenate(all_labels, axis=0)

    print(f"Loaded {X.shape[0]} nodes with {X.shape[1]} features")
    print(f"Label distribution:")
    for i, name in enumerate(['Father', 'Mother', 'Spouse', 'Children', 'Siblings']):
        missing = np.sum(y[:, i])
        print(f"  Missing {name}: {int(missing)} ({100*missing/len(y):.1f}%)")

    return X, y


# ============================================================================
# Training
# ============================================================================

def focal_loss(pred: mx.array, target: mx.array, gamma: float = 2.0, alpha: float = 0.25) -> mx.array:
    """Focal loss for handling class imbalance."""
    eps = 1e-7
    pred = mx.clip(pred, eps, 1 - eps)

    p_t = target * pred + (1 - target) * (1 - pred)
    focal_weight = (1 - p_t) ** gamma
    alpha_t = target * alpha + (1 - target) * (1 - alpha)

    ce_loss = -target * mx.log(pred) - (1 - target) * mx.log(1 - pred)
    loss = alpha_t * focal_weight * ce_loss

    return mx.mean(loss)


def train(args):
    # Load data
    print("Loading training data...")
    X, y = load_training_data(Path(args.data_dir))

    # Shuffle and split
    indices = np.random.permutation(len(X))
    split_idx = int(len(X) * 0.8)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]

    print(f"\nTrain: {len(X_train)}, Validation: {len(X_val)}")

    # Create model
    model = SimpleMLP(input_dim=X.shape[1], hidden_dim=args.hidden_dim)
    optimizer = AdamW(learning_rate=args.lr, weight_decay=0.01)

    def compute_loss(model, batch_x, batch_y):
        pred = model(batch_x)
        return focal_loss(pred, batch_y)

    loss_and_grad = mx.value_and_grad(compute_loss)

    # Training loop
    batch_size = args.batch_size
    best_val_loss = float('inf')
    patience_counter = 0

    print(f"\nTraining for {args.epochs} epochs...")
    print("-" * 60)

    for epoch in range(args.epochs):
        # Shuffle training data
        perm = np.random.permutation(len(X_train))
        epoch_loss = 0.0
        num_batches = 0

        for i in range(0, len(X_train), batch_size):
            batch_idx = perm[i:i + batch_size]
            batch_x = mx.array(X_train[batch_idx])
            batch_y = mx.array(y_train[batch_idx])

            loss, grads = loss_and_grad(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)

            epoch_loss += float(loss)
            num_batches += 1

        avg_train_loss = epoch_loss / num_batches

        # Validation
        val_pred = model(mx.array(X_val))
        val_loss = float(focal_loss(val_pred, mx.array(y_val)))

        # Accuracy
        mx.eval(val_pred)
        val_pred_np = np.array(val_pred)
        val_acc = np.mean((val_pred_np > 0.5) == y_val)

        print(f"Epoch {epoch+1:3d}/{args.epochs} | "
              f"Train Loss: {avg_train_loss:.4f} | "
              f"Val Loss: {val_loss:.4f} | "
              f"Val Acc: {val_acc:.4f}")

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0

            # Save best model
            save_model(model, Path(args.output_dir), 'best_model', args)
        else:
            patience_counter += 1
            if patience_counter >= args.patience:
                print(f"\nEarly stopping after {epoch+1} epochs")
                break

        # Learning rate decay
        if (epoch + 1) % 20 == 0:
            optimizer.learning_rate = optimizer.learning_rate * 0.5
            print(f"  → LR reduced to {optimizer.learning_rate:.6f}")

    print("-" * 60)
    print(f"Training complete. Best validation loss: {best_val_loss:.4f}")

    # Save final model
    save_model(model, Path(args.output_dir), 'final_model', args)

    # Export to ONNX
    export_onnx(model, Path(args.output_dir) / 'model.onnx', args)


def save_model(model: SimpleMLP, output_dir: Path, name: str, args):
    """Save model weights."""
    output_dir.mkdir(parents=True, exist_ok=True)

    # Save weights
    weights = {}
    for layer_name in ['fc1', 'fc2', 'fc3', 'out']:
        layer = getattr(model, layer_name)
        weights[f'{layer_name}_weight'] = np.array(layer.weight)
        weights[f'{layer_name}_bias'] = np.array(layer.bias)

    np.savez(output_dir / f'{name}.npz', **weights)

    # Save config
    config = {
        'input_dim': 176,
        'hidden_dim': args.hidden_dim,
        'num_outputs': 5,
        'output_labels': ['missing_father', 'missing_mother', 'missing_spouse',
                          'missing_children', 'missing_siblings'],
        'saved_at': datetime.now().isoformat()
    }
    with open(output_dir / f'{name}_config.json', 'w') as f:
        json.dump(config, f, indent=2)


def export_onnx(model: SimpleMLP, output_path: Path, args):
    """Export model to ONNX format."""
    try:
        import onnx
        from onnx import helper, TensorProto, numpy_helper
    except ImportError:
        print("ONNX not installed, skipping export")
        return

    print(f"\nExporting to ONNX: {output_path}")

    input_dim = 176
    hidden_dim = args.hidden_dim

    # Get weights
    w1 = np.array(model.fc1.weight).T.astype(np.float32)
    b1 = np.array(model.fc1.bias).astype(np.float32)
    w2 = np.array(model.fc2.weight).T.astype(np.float32)
    b2 = np.array(model.fc2.bias).astype(np.float32)
    w3 = np.array(model.fc3.weight).T.astype(np.float32)
    b3 = np.array(model.fc3.bias).astype(np.float32)
    w4 = np.array(model.out.weight).T.astype(np.float32)
    b4 = np.array(model.out.bias).astype(np.float32)

    # Build ONNX graph
    inputs = [helper.make_tensor_value_info('features', TensorProto.FLOAT, ['batch', input_dim])]
    outputs = [helper.make_tensor_value_info('predictions', TensorProto.FLOAT, ['batch', 5])]

    initializers = [
        numpy_helper.from_array(w1, 'W1'),
        numpy_helper.from_array(b1, 'b1'),
        numpy_helper.from_array(w2, 'W2'),
        numpy_helper.from_array(b2, 'b2'),
        numpy_helper.from_array(w3, 'W3'),
        numpy_helper.from_array(b3, 'b3'),
        numpy_helper.from_array(w4, 'W4'),
        numpy_helper.from_array(b4, 'b4'),
    ]

    nodes = [
        helper.make_node('MatMul', ['features', 'W1'], ['mm1']),
        helper.make_node('Add', ['mm1', 'b1'], ['add1']),
        helper.make_node('Relu', ['add1'], ['relu1']),

        helper.make_node('MatMul', ['relu1', 'W2'], ['mm2']),
        helper.make_node('Add', ['mm2', 'b2'], ['add2']),
        helper.make_node('Relu', ['add2'], ['relu2']),

        helper.make_node('MatMul', ['relu2', 'W3'], ['mm3']),
        helper.make_node('Add', ['mm3', 'b3'], ['add3']),
        helper.make_node('Relu', ['add3'], ['relu3']),

        helper.make_node('MatMul', ['relu3', 'W4'], ['mm4']),
        helper.make_node('Add', ['mm4', 'b4'], ['logits']),
        helper.make_node('Sigmoid', ['logits'], ['predictions']),
    ]

    graph = helper.make_graph(nodes, 'OhanaAI_MissingRelativePredictor',
                              inputs, outputs, initializer=initializers)

    model_def = helper.make_model(graph, producer_name='OhanaAI',
                                  opset_imports=[helper.make_operatorsetid('', 13)])
    model_def.ir_version = 8

    # Add metadata
    from onnx import StringStringEntryProto
    model_def.metadata_props.extend([
        StringStringEntryProto(key='model_version', value='1.0.0'),
        StringStringEntryProto(key='created_at', value=datetime.now().isoformat()),
        StringStringEntryProto(key='input_dim', value=str(input_dim)),
        StringStringEntryProto(key='output_labels',
            value='missing_father,missing_mother,missing_spouse,missing_children,missing_siblings')
    ])

    onnx.checker.check_model(model_def)
    onnx.save(model_def, str(output_path))

    print(f"ONNX model saved: {output_path}")


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Train simple MLP for missing relative prediction')
    parser.add_argument('--data-dir', type=str, default='training_data')
    parser.add_argument('--output-dir', type=str, default='models/family_tree_gnn')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=256)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('--hidden-dim', type=int, default=128)
    parser.add_argument('--patience', type=int, default=10)
    parser.add_argument('--seed', type=int, default=42)
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    train(args)


if __name__ == '__main__':
    main()
