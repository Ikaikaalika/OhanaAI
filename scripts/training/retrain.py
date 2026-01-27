#!/usr/bin/env python3
"""
OhanaAI - Retraining Script with Attribute Generation

This script handles the complete retraining pipeline:
1. Re-prepares training data with new ethnic origin and family pattern features
2. Trains the model with attribute generation capabilities
3. Exports the model for use in the web app

Usage:
    python retrain.py path/to/gedcom.ged
    python retrain.py --data-dir training_data  # Use existing prepared data
"""

import argparse
import json
import sys
from pathlib import Path
from datetime import datetime
import subprocess

# Add parent directory for imports
sys.path.insert(0, str(Path(__file__).parent))


def prepare_data(gedcom_path: Path, output_dir: Path) -> bool:
    """Prepare training data from GEDCOM file."""
    print(f"\n{'='*60}")
    print("STEP 1: Preparing Training Data")
    print(f"{'='*60}")

    prepare_script = Path(__file__).parent / 'prepare_data.py'

    cmd = [
        sys.executable,
        str(prepare_script),
        str(gedcom_path),
        '--output-dir', str(output_dir)
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("Error: Data preparation failed")
        return False

    print("Data preparation complete!")
    return True


def train_model(data_dir: Path, output_dir: Path, epochs: int = 100,
                learning_rate: float = 1e-3, hidden_dim: int = 128) -> bool:
    """Train the model with attribute generation."""
    print(f"\n{'='*60}")
    print("STEP 2: Training Model with Attribute Generation")
    print(f"{'='*60}")

    train_script = Path(__file__).parent / 'train.py'

    cmd = [
        sys.executable,
        str(train_script),
        '--data-dir', str(data_dir),
        '--output-dir', str(output_dir),
        '--epochs', str(epochs),
        '--lr', str(learning_rate),
        '--hidden-dim', str(hidden_dim),
        '--num-layers', '3',
        '--num-heads', '4',
        '--patience', '15'
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("Error: Training failed")
        return False

    print("Training complete!")
    return True


def export_model(model_dir: Path) -> bool:
    """Export model to ONNX for web deployment."""
    print(f"\n{'='*60}")
    print("STEP 3: Exporting Model")
    print(f"{'='*60}")

    export_script = Path(__file__).parent / 'export_onnx.py'

    if not export_script.exists():
        print("Note: export_onnx.py not found, skipping ONNX export")
        return True

    cmd = [
        sys.executable,
        str(export_script),
        '--model-path', str(model_dir / 'best_model.npz'),
        '--output-path', str(model_dir / 'model.onnx')
    ]

    print(f"Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, capture_output=False)

    if result.returncode != 0:
        print("Warning: ONNX export failed, but training was successful")
        return True

    print("Export complete!")
    return True


def save_model_config(output_dir: Path):
    """Save model configuration for the web app."""
    config = {
        'input_dim': 224,  # Updated feature dimension
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

    config_path = output_dir / 'best_model_config.json'
    with open(config_path, 'w') as f:
        json.dump(config, f, indent=2)

    print(f"Model config saved to: {config_path}")


def main():
    parser = argparse.ArgumentParser(
        description='Retrain OhanaAI model with attribute generation'
    )
    parser.add_argument(
        'gedcom_file',
        nargs='?',
        help='Path to GEDCOM file (optional if using --data-dir)'
    )
    parser.add_argument(
        '--data-dir',
        type=str,
        default='training_data',
        help='Directory containing training data'
    )
    parser.add_argument(
        '--output-dir',
        type=str,
        default='models/family_tree_gnn',
        help='Directory to save model'
    )
    parser.add_argument(
        '--epochs',
        type=int,
        default=100,
        help='Number of training epochs'
    )
    parser.add_argument(
        '--lr',
        type=float,
        default=1e-3,
        help='Learning rate'
    )
    parser.add_argument(
        '--hidden-dim',
        type=int,
        default=128,
        help='Hidden dimension size'
    )
    parser.add_argument(
        '--skip-prepare',
        action='store_true',
        help='Skip data preparation (use existing training data)'
    )
    parser.add_argument(
        '--skip-export',
        action='store_true',
        help='Skip ONNX export'
    )

    args = parser.parse_args()

    print("="*60)
    print("OhanaAI Model Retraining with Attribute Generation")
    print("="*60)
    print(f"Started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    data_dir = Path(args.data_dir)
    output_dir = Path(args.output_dir)

    # Step 1: Prepare data if GEDCOM file provided
    if args.gedcom_file and not args.skip_prepare:
        gedcom_path = Path(args.gedcom_file)
        if not gedcom_path.exists():
            print(f"Error: GEDCOM file not found: {gedcom_path}")
            sys.exit(1)

        if not prepare_data(gedcom_path, data_dir):
            sys.exit(1)
    elif not data_dir.exists():
        print(f"Error: Training data directory not found: {data_dir}")
        print("Please provide a GEDCOM file or ensure training data exists")
        sys.exit(1)
    else:
        print(f"\nUsing existing training data from: {data_dir}")

    # Step 2: Train model
    if not train_model(data_dir, output_dir, args.epochs, args.lr, args.hidden_dim):
        sys.exit(1)

    # Step 3: Save config
    save_model_config(output_dir)

    # Step 4: Export to ONNX
    if not args.skip_export:
        export_model(output_dir)

    print(f"\n{'='*60}")
    print("RETRAINING COMPLETE!")
    print(f"{'='*60}")
    print(f"Model saved to: {output_dir}")
    print(f"Completed at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    print("New model capabilities:")
    print("  - Missing relative detection (father, mother, spouse, children, siblings)")
    print("  - Birth year prediction for missing parents")
    print("  - Ethnic origin prediction (12 ethnicities)")
    print("  - Location prediction for missing relatives")
    print("  - Family pattern analysis (naming traditions, age gaps)")
    print()
    print("To use the new model in the web app, copy the model files to")
    print("the appropriate location and update the model loading code.")


if __name__ == '__main__':
    main()
