#!/usr/bin/env python3
"""
OhanaAI - Model Training Script

This is the single, authoritative way to train the OhanaAI model.
It handles the complete pipeline:
1. Prepares training data from GEDCOM files (optional)
2. Trains the GNN model with attribute generation
3. Exports the model for deployment

Usage:
    # Train with existing data
    python retrain.py

    # Prepare new data and train
    python retrain.py path/to/gedcom.ged

    # Just prepare data (no training)
    python retrain.py path/to/gedcom.ged --prepare-only

    # Train with custom settings
    python retrain.py --epochs 200 --lr 0.0005
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))


def prepare_data(gedcom_path: Path, output_dir: Path) -> bool:
    """Prepare training data from GEDCOM file."""
    print(f"\n{'='*60}")
    print("STEP 1: Preparing Training Data")
    print(f"{'='*60}")
    print(f"Source: {gedcom_path}")
    print(f"Output: {output_dir}")

    try:
        from prepare_data import parse_gedcom, prepare_training_data, FEATURE_DIM

        print("\nParsing GEDCOM file...")
        individuals, families = parse_gedcom(gedcom_path)
        print(f"  Found {len(individuals)} individuals")
        print(f"  Found {len(families)} families")

        print("\nExtracting features...")
        training_data = prepare_training_data(individuals, families)
        print(f"  Feature dimension: {FEATURE_DIM}")
        print(f"  Nodes: {len(training_data['nodeFeatures'])}")
        print(f"  Edges: {len(training_data['edgeIndex'][0])}")

        # Count labels
        labels = training_data['labels']
        print("\nLabel distribution:")
        for key, values in labels.items():
            missing = sum(values)
            print(f"  {key}: {int(missing)} missing ({100*missing/len(values):.1f}%)")

        # Save training data
        output_dir.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        base_name = gedcom_path.stem.replace(' ', '_')
        output_path = output_dir / f"training_{base_name}_{timestamp}.json"

        output = {
            'metadata': {
                'sourceFile': gedcom_path.name,
                'exportedAt': datetime.now().isoformat(),
                'individuals': len(individuals),
                'families': len(families),
                'featureDimension': FEATURE_DIM
            },
            'data': [{
                'id': base_name,
                **training_data
            }]
        }

        with open(output_path, 'w') as f:
            json.dump(output, f)

        print(f"\nTraining data saved to: {output_path}")
        return True

    except Exception as e:
        print(f"Error preparing data: {e}")
        import traceback
        traceback.print_exc()
        return False


def train_model(data_dir: Path, output_dir: Path, epochs: int = 100,
                learning_rate: float = 1e-3, hidden_dim: int = 128,
                num_layers: int = 3, num_heads: int = 4,
                patience: int = 15) -> bool:
    """Train the model with attribute generation."""
    print(f"\n{'='*60}")
    print("STEP 2: Training Model")
    print(f"{'='*60}")
    print(f"Data: {data_dir}")
    print(f"Output: {output_dir}")
    print(f"Config: {epochs} epochs, lr={learning_rate}, hidden={hidden_dim}")
    print(f"        {num_layers} layers, {num_heads} attention heads")

    try:
        import random
        import numpy as np

        # Set seeds
        random.seed(42)
        np.random.seed(42)

        from train import GedcomDataset, Trainer
        from model import ModelConfig, create_model, count_parameters

        # Create model config
        model_config = ModelConfig()
        model_config.hidden_dim = hidden_dim
        model_config.num_gnn_layers = num_layers
        model_config.num_attention_heads = num_heads

        # Load data
        print("\nLoading training data...")
        dataset = GedcomDataset(data_dir, model_config)

        if len(dataset) == 0:
            print("Error: No valid training examples found")
            return False

        print(f"Loaded {len(dataset)} training examples")

        # Split data
        if len(dataset) == 1:
            print("Single file mode: using same data for train and validation")
            train_dataset = dataset
            val_dataset = dataset
        else:
            from sklearn.model_selection import train_test_split
            indices = list(range(len(dataset)))
            train_indices, val_indices = train_test_split(indices, test_size=0.2, random_state=42)

            train_dataset = GedcomDataset.__new__(GedcomDataset)
            train_dataset.config = model_config
            train_dataset.examples = [dataset[i] for i in train_indices]

            val_dataset = GedcomDataset.__new__(GedcomDataset)
            val_dataset.config = model_config
            val_dataset.examples = [dataset[i] for i in val_indices]

        # Create model
        print("\nInitializing model...")
        model = create_model(model_config)
        print(f"Parameters: {count_parameters(model):,}")

        # Training config
        train_config = {
            'learning_rate': learning_rate,
            'weight_decay': 0.01,
            'patience': patience,
            'augmentation': True,
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
            'train_attributes': True,
            'attribute_loss_weight': 0.5,
            'attribute_weights': {
                'birth_year': 1.0,
                'ethnic_origin': 0.5,
                'location': 0.3
            }
        }

        # Train
        trainer = Trainer(model, train_dataset, val_dataset, train_config)
        trainer.train(epochs, output_dir)

        return True

    except Exception as e:
        print(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return False


def export_model(model_dir: Path) -> bool:
    """Export model to ONNX for deployment."""
    print(f"\n{'='*60}")
    print("STEP 3: Exporting Model")
    print(f"{'='*60}")

    model_path = model_dir / 'best_model.npz'
    if not model_path.exists():
        model_path = model_dir / 'final_model.npz'

    if not model_path.exists():
        print("No model checkpoint found, skipping export")
        return True

    try:
        export_script = Path(__file__).parent / 'export_onnx.py'
        if export_script.exists():
            import subprocess
            result = subprocess.run([
                sys.executable, str(export_script),
                '--checkpoint', str(model_path),
                '--output', str(model_dir / 'model.onnx'),
                '--hidden-dim', '128'
            ], capture_output=True, text=True)

            if result.returncode == 0:
                print(f"ONNX model exported to: {model_dir / 'model.onnx'}")
            else:
                print(f"ONNX export warning: {result.stderr}")
        else:
            print("export_onnx.py not found, skipping ONNX export")

        return True

    except Exception as e:
        print(f"Warning: ONNX export failed: {e}")
        return True  # Non-fatal


def save_config(output_dir: Path):
    """Save model configuration."""
    config = {
        'input_dim': 224,
        'hidden_dim': 128,
        'num_outputs': 5,
        'output_labels': [
            'missing_father',
            'missing_mother',
            'missing_spouse',
            'missing_children',
            'missing_siblings'
        ],
        'attribute_outputs': {
            'father': ['birth_year', 'ethnic_origin', 'location'],
            'mother': ['birth_year', 'ethnic_origin', 'location']
        },
        'ethnic_classes': [
            'irish', 'german', 'italian', 'polish',
            'scandinavian', 'scottish', 'jewish', 'portuguese',
            'hawaiian', 'chinese', 'japanese', 'filipino'
        ],
        'saved_at': datetime.now().isoformat(),
        'version': '2.0.0'
    }

    config_path = output_dir / 'model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)
    print(f"Config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='OhanaAI Model Training',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python retrain.py                           # Train with existing data
  python retrain.py data.ged                  # Prepare data and train
  python retrain.py data.ged --prepare-only   # Only prepare data
  python retrain.py --epochs 200              # Train with 200 epochs
        """
    )
    parser.add_argument('gedcom_file', nargs='?', help='GEDCOM file to process')
    parser.add_argument('--data-dir', type=str, default='training_data',
                        help='Training data directory (default: training_data)')
    parser.add_argument('--output-dir', type=str, default='models/family_tree_gnn',
                        help='Model output directory (default: models/family_tree_gnn)')
    parser.add_argument('--epochs', type=int, default=100, help='Training epochs (default: 100)')
    parser.add_argument('--lr', type=float, default=1e-3, help='Learning rate (default: 0.001)')
    parser.add_argument('--hidden-dim', type=int, default=128, help='Hidden dimension (default: 128)')
    parser.add_argument('--num-layers', type=int, default=3, help='GNN layers (default: 3)')
    parser.add_argument('--num-heads', type=int, default=4, help='Attention heads (default: 4)')
    parser.add_argument('--patience', type=int, default=15, help='Early stopping patience (default: 15)')
    parser.add_argument('--prepare-only', action='store_true', help='Only prepare data, skip training')
    parser.add_argument('--skip-export', action='store_true', help='Skip ONNX export')

    args = parser.parse_args()

    print("="*60)
    print("OhanaAI Model Training")
    print("="*60)
    print(f"Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Step 1: Prepare data if GEDCOM provided
    if args.gedcom_file:
        gedcom_path = Path(args.gedcom_file)
        if not gedcom_path.exists():
            print(f"Error: File not found: {gedcom_path}")
            sys.exit(1)

        if not prepare_data(gedcom_path, data_dir):
            sys.exit(1)

        if args.prepare_only:
            print("\nData preparation complete (--prepare-only)")
            sys.exit(0)

    elif not data_dir.exists():
        print(f"Error: No training data found at {data_dir}")
        print("Provide a GEDCOM file or ensure training data exists")
        sys.exit(1)
    else:
        print(f"\nUsing existing training data: {data_dir}")

    # Step 2: Train model
    if not train_model(
        data_dir, output_dir,
        epochs=args.epochs,
        learning_rate=args.lr,
        hidden_dim=args.hidden_dim,
        num_layers=args.num_layers,
        num_heads=args.num_heads,
        patience=args.patience
    ):
        sys.exit(1)

    # Step 3: Save config
    save_config(output_dir)

    # Step 4: Export
    if not args.skip_export:
        export_model(output_dir)

    print(f"\n{'='*60}")
    print("TRAINING COMPLETE")
    print(f"{'='*60}")
    print(f"Model saved to: {output_dir}")
    print(f"Finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("Model capabilities:")
    print("  - Missing relative detection (father, mother, spouse, children, siblings)")
    print("  - Birth year prediction with confidence")
    print("  - Ethnic origin prediction (12 ethnicities)")
    print("  - Location prediction")
    print("  - Family pattern analysis")


if __name__ == '__main__':
    main()
