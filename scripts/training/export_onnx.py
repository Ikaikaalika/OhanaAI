#!/usr/bin/env python3
"""
OhanaAI - ONNX Export for Model Deployment

Exports the trained MLX model to ONNX format for:
1. Cross-platform inference (Node.js, browser, etc.)
2. Optimized inference with ONNX Runtime
3. Model versioning and deployment
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import onnx
from onnx import helper, TensorProto, numpy_helper

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))
from model import ModelConfig


# ============================================================================
# ONNX Model Builder
# ============================================================================

class ONNXExporter:
    """Exports trained model weights to ONNX format."""

    def __init__(self, config: ModelConfig):
        self.config = config

    def load_weights(self, checkpoint_path: Path) -> Dict[str, np.ndarray]:
        """Load weights from MLX checkpoint."""
        # Load .npz file
        weights = {}
        with np.load(checkpoint_path, allow_pickle=True) as data:
            for key in data.files:
                weights[key] = data[key]
        return weights

    def export(
        self,
        weights: Dict[str, np.ndarray],
        output_path: Path,
        model_version: str = "1.0.0"
    ):
        """
        Export model to ONNX format.

        The exported model has:
        - Input: node_features [batch, num_nodes, feature_dim]
        - Output: predictions dictionary with missing relative probabilities
        """
        # For ONNX, we need to simplify the GNN to a feedforward network
        # since ONNX doesn't natively support graph operations well.
        # We'll create a model that processes pre-aggregated node features.

        # Input definitions
        inputs = [
            helper.make_tensor_value_info(
                'node_features',
                TensorProto.FLOAT,
                ['batch', self.config.node_feature_dim]
            )
        ]

        # Output definitions
        outputs = [
            helper.make_tensor_value_info(
                'missing_father',
                TensorProto.FLOAT,
                ['batch', 1]
            ),
            helper.make_tensor_value_info(
                'missing_mother',
                TensorProto.FLOAT,
                ['batch', 1]
            ),
            helper.make_tensor_value_info(
                'missing_spouse',
                TensorProto.FLOAT,
                ['batch', 1]
            ),
            helper.make_tensor_value_info(
                'missing_children',
                TensorProto.FLOAT,
                ['batch', 1]
            ),
            helper.make_tensor_value_info(
                'missing_siblings',
                TensorProto.FLOAT,
                ['batch', 1]
            )
        ]

        nodes = []
        initializers = []

        # Build simplified MLP version of the model
        # Input projection
        nodes, initializers = self._add_mlp_layer(
            nodes, initializers, weights,
            'encoder.input_proj',
            'node_features',
            'input_proj_out',
            activation='relu'
        )

        # Simplified GNN layers (as MLPs for single-node inference)
        current_input = 'input_proj_out'
        for i in range(self.config.num_gnn_layers):
            layer_name = f'gnn_layer_{i}'
            output_name = f'gnn_{i}_out'

            nodes, initializers = self._add_mlp_layer(
                nodes, initializers, weights,
                f'encoder.gnn_layers.{i}',
                current_input,
                output_name,
                activation='relu'
            )
            current_input = output_name

        # Final layer norm
        nodes, initializers = self._add_layer_norm(
            nodes, initializers, weights,
            'encoder.final_norm',
            current_input,
            'encoded'
        )

        # Prediction heads
        for head_name in ['missing_father', 'missing_mother', 'missing_spouse',
                          'missing_children', 'missing_siblings']:
            weight_prefix = head_name.replace('missing_', '') + '_head'
            nodes, initializers = self._add_prediction_head(
                nodes, initializers, weights,
                weight_prefix,
                'encoded',
                head_name
            )

        # Create the graph
        graph = helper.make_graph(
            nodes,
            'OhanaAI_FamilyTreeGNN',
            inputs,
            outputs,
            initializer=initializers
        )

        # Create the model
        model_def = helper.make_model(
            graph,
            producer_name='OhanaAI',
            producer_version=model_version,
            opset_imports=[helper.make_operatorsetid('', 13)]
        )

        model_def.ir_version = 8

        # Add metadata
        model_def.metadata_props.append(
            helper.make_string_string_entry_proto('model_version', model_version)
        )
        model_def.metadata_props.append(
            helper.make_string_string_entry_proto('created_at', datetime.now().isoformat())
        )
        model_def.metadata_props.append(
            helper.make_string_string_entry_proto('node_feature_dim', str(self.config.node_feature_dim))
        )
        model_def.metadata_props.append(
            helper.make_string_string_entry_proto('hidden_dim', str(self.config.hidden_dim))
        )

        # Validate
        onnx.checker.check_model(model_def)

        # Save
        onnx.save(model_def, str(output_path))

        print(f"Exported ONNX model to {output_path}")
        print(f"  Model version: {model_version}")
        print(f"  Input shape: [batch, {self.config.node_feature_dim}]")
        print(f"  Hidden dim: {self.config.hidden_dim}")
        print(f"  Outputs: missing_father, missing_mother, missing_spouse, missing_children, missing_siblings")

    def _add_mlp_layer(
        self,
        nodes: list,
        initializers: list,
        weights: Dict[str, np.ndarray],
        weight_prefix: str,
        input_name: str,
        output_name: str,
        activation: Optional[str] = None
    ) -> tuple:
        """Add an MLP layer to the graph."""
        # Try to find weights with various naming conventions
        weight_key = None
        bias_key = None

        for key in weights.keys():
            if weight_prefix in key:
                if 'weight' in key.lower() or key.endswith('.W') or key.endswith('_W'):
                    weight_key = key
                elif 'bias' in key.lower() or key.endswith('.b') or key.endswith('_b'):
                    bias_key = key

        # If we can't find weights, create dummy ones
        if weight_key is None:
            print(f"  Warning: No weights found for {weight_prefix}, using random initialization")
            # Determine dimensions from config
            in_dim = self.config.node_feature_dim if 'input_proj' in weight_prefix else self.config.hidden_dim
            out_dim = self.config.hidden_dim
            W = np.random.randn(in_dim, out_dim).astype(np.float32) * 0.1
            b = np.zeros(out_dim, dtype=np.float32)
        else:
            W = weights[weight_key].astype(np.float32)
            b = weights.get(bias_key, np.zeros(W.shape[-1])).astype(np.float32)

        # Ensure weight is 2D
        if len(W.shape) == 1:
            W = W.reshape(-1, 1)

        # May need to transpose depending on convention
        if W.shape[0] != self.config.hidden_dim and W.shape[1] == self.config.hidden_dim:
            pass  # Already correct orientation
        elif W.shape[0] == self.config.hidden_dim:
            W = W.T

        weight_name = f'{output_name}_W'
        bias_name = f'{output_name}_b'

        initializers.append(numpy_helper.from_array(W, weight_name))
        initializers.append(numpy_helper.from_array(b.flatten(), bias_name))

        # MatMul
        matmul_out = f'{output_name}_matmul'
        nodes.append(helper.make_node('MatMul', [input_name, weight_name], [matmul_out]))

        # Add bias
        add_out = f'{output_name}_add'
        nodes.append(helper.make_node('Add', [matmul_out, bias_name], [add_out]))

        # Activation
        if activation == 'relu':
            nodes.append(helper.make_node('Relu', [add_out], [output_name]))
        elif activation == 'sigmoid':
            nodes.append(helper.make_node('Sigmoid', [add_out], [output_name]))
        else:
            nodes.append(helper.make_node('Identity', [add_out], [output_name]))

        return nodes, initializers

    def _add_layer_norm(
        self,
        nodes: list,
        initializers: list,
        weights: Dict[str, np.ndarray],
        weight_prefix: str,
        input_name: str,
        output_name: str
    ) -> tuple:
        """Add layer normalization."""
        # Find weights
        gamma = None
        beta = None

        for key in weights.keys():
            if weight_prefix in key:
                if 'weight' in key.lower() or 'gamma' in key.lower():
                    gamma = weights[key].astype(np.float32)
                elif 'bias' in key.lower() or 'beta' in key.lower():
                    beta = weights[key].astype(np.float32)

        # Use defaults if not found
        if gamma is None:
            gamma = np.ones(self.config.hidden_dim, dtype=np.float32)
        if beta is None:
            beta = np.zeros(self.config.hidden_dim, dtype=np.float32)

        gamma_name = f'{output_name}_gamma'
        beta_name = f'{output_name}_beta'

        initializers.append(numpy_helper.from_array(gamma.flatten(), gamma_name))
        initializers.append(numpy_helper.from_array(beta.flatten(), beta_name))

        # LayerNormalization
        nodes.append(helper.make_node(
            'LayerNormalization',
            [input_name, gamma_name, beta_name],
            [output_name],
            axis=-1,
            epsilon=1e-5
        ))

        return nodes, initializers

    def _add_prediction_head(
        self,
        nodes: list,
        initializers: list,
        weights: Dict[str, np.ndarray],
        weight_prefix: str,
        input_name: str,
        output_name: str
    ) -> tuple:
        """Add a prediction head (2-layer MLP with sigmoid)."""
        # First layer
        fc1_out = f'{output_name}_fc1'
        nodes, initializers = self._add_mlp_layer(
            nodes, initializers, weights,
            f'{weight_prefix}.fc1',
            input_name,
            fc1_out,
            activation='relu'
        )

        # Second layer with sigmoid
        fc2_out = f'{output_name}_fc2'
        nodes, initializers = self._add_mlp_layer(
            nodes, initializers, weights,
            f'{weight_prefix}.fc2',
            fc1_out,
            fc2_out,
            activation='sigmoid'
        )

        # Ensure output name
        nodes.append(helper.make_node('Identity', [fc2_out], [output_name]))

        return nodes, initializers


# ============================================================================
# Simplified Export (from scratch weights)
# ============================================================================

def export_simple_model(
    config: ModelConfig,
    output_path: Path,
    model_version: str = "1.0.0"
):
    """
    Export a simplified feedforward model for ONNX.

    This version doesn't require pre-trained weights and creates
    a model that can be fine-tuned or used as-is with heuristic fallback.
    """
    # Input
    inputs = [
        helper.make_tensor_value_info(
            'features',
            TensorProto.FLOAT,
            ['batch', config.node_feature_dim]
        )
    ]

    # Outputs
    outputs = [
        helper.make_tensor_value_info(
            'predictions',
            TensorProto.FLOAT,
            ['batch', 5]  # 5 prediction types
        )
    ]

    nodes = []
    initializers = []

    # Layer 1: Input -> Hidden
    W1 = np.random.randn(config.node_feature_dim, config.hidden_dim).astype(np.float32) * 0.1
    b1 = np.zeros(config.hidden_dim, dtype=np.float32)

    initializers.append(numpy_helper.from_array(W1, 'W1'))
    initializers.append(numpy_helper.from_array(b1, 'b1'))

    nodes.append(helper.make_node('MatMul', ['features', 'W1'], ['mm1']))
    nodes.append(helper.make_node('Add', ['mm1', 'b1'], ['add1']))
    nodes.append(helper.make_node('Relu', ['add1'], ['relu1']))

    # Layer 2: Hidden -> Hidden
    W2 = np.random.randn(config.hidden_dim, config.hidden_dim // 2).astype(np.float32) * 0.1
    b2 = np.zeros(config.hidden_dim // 2, dtype=np.float32)

    initializers.append(numpy_helper.from_array(W2, 'W2'))
    initializers.append(numpy_helper.from_array(b2, 'b2'))

    nodes.append(helper.make_node('MatMul', ['relu1', 'W2'], ['mm2']))
    nodes.append(helper.make_node('Add', ['mm2', 'b2'], ['add2']))
    nodes.append(helper.make_node('Relu', ['add2'], ['relu2']))

    # Layer 3: Hidden -> Output
    W3 = np.random.randn(config.hidden_dim // 2, 5).astype(np.float32) * 0.1
    b3 = np.zeros(5, dtype=np.float32)

    initializers.append(numpy_helper.from_array(W3, 'W3'))
    initializers.append(numpy_helper.from_array(b3, 'b3'))

    nodes.append(helper.make_node('MatMul', ['relu2', 'W3'], ['mm3']))
    nodes.append(helper.make_node('Add', ['mm3', 'b3'], ['logits']))
    nodes.append(helper.make_node('Sigmoid', ['logits'], ['predictions']))

    # Create graph
    graph = helper.make_graph(
        nodes,
        'OhanaAI_SimplePredictor',
        inputs,
        outputs,
        initializer=initializers
    )

    # Create model
    model_def = helper.make_model(
        graph,
        producer_name='OhanaAI',
        producer_version=model_version,
        opset_imports=[helper.make_operatorsetid('', 13)]
    )

    model_def.ir_version = 8

    # Add metadata
    model_def.metadata_props.extend([
        helper.make_string_string_entry_proto('model_version', model_version),
        helper.make_string_string_entry_proto('created_at', datetime.now().isoformat()),
        helper.make_string_string_entry_proto('node_feature_dim', str(config.node_feature_dim)),
        helper.make_string_string_entry_proto('model_type', 'simple_mlp'),
        helper.make_string_string_entry_proto('output_labels', 'missing_father,missing_mother,missing_spouse,missing_children,missing_siblings')
    ])

    # Validate and save
    onnx.checker.check_model(model_def)
    onnx.save(model_def, str(output_path))

    print(f"Exported simple ONNX model to {output_path}")

    # Save config
    config_path = output_path.with_suffix('.json')
    with open(config_path, 'w') as f:
        json.dump({
            'node_feature_dim': config.node_feature_dim,
            'hidden_dim': config.hidden_dim,
            'model_version': model_version,
            'output_labels': ['missing_father', 'missing_mother', 'missing_spouse', 'missing_children', 'missing_siblings'],
            'created_at': datetime.now().isoformat()
        }, f, indent=2)


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Export model to ONNX')
    parser.add_argument('--checkpoint', type=str, default=None,
                        help='Path to model checkpoint (.npz)')
    parser.add_argument('--output', type=str, default='models/family_tree_gnn/model.onnx',
                        help='Output ONNX file path')
    parser.add_argument('--simple', action='store_true',
                        help='Export simple model (no checkpoint needed)')
    parser.add_argument('--version', type=str, default='1.0.0',
                        help='Model version string')
    parser.add_argument('--hidden-dim', type=int, default=128,
                        help='Hidden dimension (for simple model)')
    parser.add_argument('--feature-dim', type=int, default=176,
                        help='Node feature dimension')

    args = parser.parse_args()

    # Create config
    config = ModelConfig()
    config.hidden_dim = args.hidden_dim
    config.node_feature_dim = args.feature_dim

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if args.simple:
        export_simple_model(config, output_path, args.version)
    else:
        if not args.checkpoint:
            print("Error: --checkpoint required for full model export")
            print("Use --simple to export a model without pre-trained weights")
            sys.exit(1)

        checkpoint_path = Path(args.checkpoint)
        if not checkpoint_path.exists():
            print(f"Error: Checkpoint not found: {checkpoint_path}")
            sys.exit(1)

        exporter = ONNXExporter(config)
        weights = exporter.load_weights(checkpoint_path)
        exporter.export(weights, output_path, args.version)


if __name__ == '__main__':
    main()
