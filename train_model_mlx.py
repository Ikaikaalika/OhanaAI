#!/usr/bin/env python3
"""
Ohana AI - Parent Prediction Training (MLX)
Trains a simple MLP classifier on exported training_data batches
and writes an ONNX model compatible with the Next.js inference layer.
"""

import argparse
import glob
import json
import os
from pathlib import Path
from typing import List, Tuple

import mlx.core as mx
import mlx.nn as nn
from mlx.optimizers import AdamW
import numpy as np
from sklearn.model_selection import train_test_split
import onnx
from onnx import helper, TensorProto

INPUT_DIM = 12
HIDDEN_DIMS = [64, 32]
OUTPUT_DIM = 3


def load_training_examples(data_dir: Path) -> Tuple[np.ndarray, np.ndarray]:
    files = sorted(data_dir.glob('training_batch_*.json'))
    if not files:
        raise RuntimeError(f'No training batches found in {data_dir}')

    features: List[np.ndarray] = []
    labels: List[np.ndarray] = []

    for file_path in files:
        with open(file_path, 'r') as f:
            payload = json.load(f)
        for item in payload.get('data', []):
            graph = item.get('graphData', {})
            graph_nodes = {node['id']: node for node in graph.get('nodes', [])}
            for label in item.get('labels', []):
                node = graph_nodes.get(label.get('personId'))
                if not node:
                    continue
                feats = np.array(node.get('features', []), dtype=np.float32)
                if feats.size == 0:
                    continue
                feats = np.pad(feats, (0, max(0, INPUT_DIM - feats.shape[0])))[:INPUT_DIM]
                target = np.zeros(OUTPUT_DIM, dtype=np.float32)
                target[0] = 1.0 if label.get('hasMissingParent') else 0.0
                missing_type = label.get('missingParentType')
                if missing_type == 'father':
                    target[1] = 1.0
                elif missing_type == 'mother':
                    target[2] = 1.0
                elif missing_type == 'both':
                    target[1] = 1.0
                    target[2] = 1.0
                features.append(feats)
                labels.append(target)

    if not features:
        raise RuntimeError('No usable training samples were found in the exported data.')

    X = np.stack(features)
    y = np.stack(labels)
    return X, y


class ParentPredictor(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(INPUT_DIM, HIDDEN_DIMS[0])
        self.fc2 = nn.Linear(HIDDEN_DIMS[0], HIDDEN_DIMS[1])
        self.out = nn.Linear(HIDDEN_DIMS[1], OUTPUT_DIM)

    def __call__(self, x):
        x = nn.relu(self.fc1(x))
        x = nn.relu(self.fc2(x))
        x = nn.sigmoid(self.out(x))
        return x


def bce_loss(pred, target):
    eps = 1e-7
    return -mx.mean(target * mx.log(pred + eps) + (1 - target) * mx.log(1 - pred + eps))


def evaluate(model, X, y):
    preds = model(mx.array(X))
    mx.eval(preds)
    preds = np.array(preds)
    preds = preds.reshape(y.shape)
    loss = -np.mean(y * np.log(preds + 1e-7) + (1 - y) * np.log(1 - preds + 1e-7))
    acc = np.mean((preds > 0.5) == (y > 0.5))
    return loss, acc


def export_onnx(model: ParentPredictor, output_path: Path):
    inputs = helper.make_tensor_value_info('features', TensorProto.FLOAT, ['batch', INPUT_DIM])
    outputs = helper.make_tensor_value_info('predictions', TensorProto.FLOAT, ['batch', OUTPUT_DIM])

    nodes = []
    initializers = []

    def add_linear(name_prefix, weight, bias, input_name, output_name, activation=None):
        weight_name = f'{name_prefix}_W'
        bias_name = f'{name_prefix}_B'
        initializers.append(helper.make_tensor(weight_name, TensorProto.FLOAT, weight.shape, weight.flatten()))
        initializers.append(helper.make_tensor(bias_name, TensorProto.FLOAT, bias.shape, bias.flatten()))
        matmul_out = f'{name_prefix}_MatMul'
        nodes.append(helper.make_node('MatMul', [input_name, weight_name], [matmul_out]))
        add_out = output_name
        nodes.append(helper.make_node('Add', [matmul_out, bias_name], [add_out]))
        if activation == 'Relu':
            act_out = f'{name_prefix}_Relu'
            nodes.append(helper.make_node('Relu', [add_out], [act_out]))
            return act_out
        if activation == 'Sigmoid':
            act_out = f'{name_prefix}_Sigmoid'
            nodes.append(helper.make_node('Sigmoid', [add_out], [act_out]))
            return act_out
        return add_out

    x = 'features'
    w1 = np.array(model.fc1.weight)
    b1 = np.array(model.fc1.bias)
    x = add_linear('FC1', w1.T, b1, x, 'fc1_out', activation='Relu')
    w2 = np.array(model.fc2.weight)
    b2 = np.array(model.fc2.bias)
    x = add_linear('FC2', w2.T, b2, x, 'fc2_out', activation='Relu')
    w3 = np.array(model.out.weight)
    b3 = np.array(model.out.bias)
    x = add_linear('FC3', w3.T, b3, x, 'fc3_out', activation='Sigmoid')
    nodes.append(helper.make_node('Identity', [x], ['predictions']))

    graph = helper.make_graph(nodes, 'OhanaParentPredictor', [inputs], [outputs], initializer=initializers)
    model_proto = helper.make_model(graph, producer_name='OhanaAI', opset_imports=[helper.make_operatorsetid('', 11)])
    model_proto.ir_version = 7
    onnx.save(model_proto, output_path)


def train(args):
    data_dir = Path(args.data_dir)
    model_dir = Path(args.output_dir)
    model_dir.mkdir(parents=True, exist_ok=True)

    X, y = load_training_examples(data_dir)
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

    model = ParentPredictor()
    optimizer = AdamW(learning_rate=args.lr)

    def loss_fn(m, batch_x, batch_y):
        preds = m(batch_x)
        return bce_loss(preds, batch_y)

    loss_and_grad = mx.value_and_grad(loss_fn)

    batch_size = args.batch_size
    num_batches = int(np.ceil(X_train.shape[0] / batch_size))

    for epoch in range(args.epochs):
        perm = np.random.permutation(X_train.shape[0])
        epoch_loss = 0.0
        for b in range(num_batches):
            idx = perm[b * batch_size:(b + 1) * batch_size]
            batch_x = mx.array(X_train[idx])
            batch_y = mx.array(y_train[idx])

            loss, grads = loss_and_grad(model, batch_x, batch_y)
            optimizer.update(model, grads)
            mx.eval(model.parameters(), optimizer.state)
            epoch_loss += float(loss)

        val_loss, val_acc = evaluate(model, X_val, y_val)
        print(f'Epoch {epoch + 1}/{args.epochs} - loss: {epoch_loss / num_batches:.4f} - val_loss: {val_loss:.4f} - val_acc: {val_acc:.4f}')

    onnx_path = model_dir / 'model.onnx'
    export_onnx(model, onnx_path)
    print(f'Saved ONNX model to {onnx_path}')


def main():
    parser = argparse.ArgumentParser(description='Train MLX parent predictor model')
    parser.add_argument('--data-dir', default='training_data', help='Directory containing exported training batches')
    parser.add_argument('--output-dir', default='models/parent_predictor', help='Directory to save ONNX model')
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--lr', type=float, default=1e-3)
    args = parser.parse_args()
    train(args)


if __name__ == '__main__':
    main()
