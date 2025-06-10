"""
Main entry point for OhanaAI - Genealogical Parent Prediction System.
Provides both CLI and GUI interfaces for training models and making predictions.
"""

import argparse
import sys
import os
import logging
from typing import List, Optional
import yaml

# Add the parent directory to the path so we can import our modules
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ohana_ai.gedcom_parser import parse_gedcom_file
from ohana_ai.trainer import train_model, OhanaAITrainer
from ohana_ai.predictor import predict_parents, OhanaAIPredictor
from ohana_ai.data_deduplication import deduplicate_gedcom_files
from ohana_ai.gui import OhanaAIGUI

def setup_logging(log_level: str = "INFO"):
    """Setup logging configuration."""
    logging.basicConfig(
        level=getattr(logging, log_level.upper()),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout)
        ]
    )

def cli_train(args):
    """Train model via CLI."""
    print("Training OhanaAI model...")
    print(f"GEDCOM files: {args.gedcom_files}")
    print(f"Config: {args.config}")
    
    try:
        trainer = train_model(args.gedcom_files, args.config)
        print(f"Training completed! Model saved to: {trainer.config['paths']['models']}")
        
        # Print training summary
        if trainer.training_history:
            last_metrics = trainer.training_history[-1]
            print(f"Final training accuracy: {last_metrics.train_accuracy:.4f}")
            print(f"Final validation accuracy: {last_metrics.val_accuracy:.4f}")
            print(f"Total epochs: {last_metrics.epoch}")
        
    except Exception as e:
        print(f"Error during training: {e}")
        return 1
    
    return 0

def cli_predict(args):
    """Run predictions via CLI."""
    print("Running parent predictions...")
    print(f"GEDCOM files: {args.gedcom_files}")
    print(f"Model: {args.model}")
    print(f"Config: {args.config}")
    
    try:
        predictions = predict_parents(
            args.gedcom_files, 
            args.model, 
            args.config
        )
        
        print(f"Generated {len(predictions)} predictions")
        
        # Show top predictions
        high_confidence = [p for p in predictions if p.confidence_score > 0.7]
        print(f"High confidence predictions (>0.7): {len(high_confidence)}")
        
        if high_confidence:
            print("\nTop 5 predictions:")
            for i, pred in enumerate(high_confidence[:5]):
                print(f"{i+1}. {pred.child_name} -> {pred.candidate_parent_name} "
                      f"(confidence: {pred.confidence_score:.3f})")
        
        print(f"Results exported to outputs/ directory")
        
    except Exception as e:
        print(f"Error during prediction: {e}")
        return 1
    
    return 0

def cli_deduplicate(args):
    """Run deduplication via CLI."""
    print("Detecting and removing duplicates...")
    print(f"GEDCOM files: {args.gedcom_files}")
    
    try:
        individuals, families, remaining_duplicates = deduplicate_gedcom_files(
            args.gedcom_files, 
            auto_merge_threshold=args.threshold
        )
        
        print(f"Deduplication completed!")
        print(f"Final counts: {len(individuals)} individuals, {len(families)} families")
        print(f"Remaining potential duplicates: {len(remaining_duplicates)}")
        
        if remaining_duplicates:
            print("\nTop potential duplicates for manual review:")
            for i, dup in enumerate(remaining_duplicates[:5]):
                print(f"{i+1}. {dup.individual1_id} <-> {dup.individual2_id} "
                      f"(similarity: {dup.similarity_score:.3f})")
        
    except Exception as e:
        print(f"Error during deduplication: {e}")
        return 1
    
    return 0

def cli_info(args):
    """Show information about GEDCOM files."""
    print("GEDCOM File Information:")
    
    total_individuals = 0
    total_families = 0
    
    for gedcom_file in args.gedcom_files:
        try:
            print(f"\nFile: {gedcom_file}")
            individuals, families = parse_gedcom_file(gedcom_file)
            
            print(f"  Individuals: {len(individuals)}")
            print(f"  Families: {len(families)}")
            
            # Gender distribution
            genders = {'M': 0, 'F': 0, 'U': 0}
            for ind in individuals.values():
                genders[ind.gender] = genders.get(ind.gender, 0) + 1
            
            print(f"  Gender distribution: M={genders['M']}, F={genders['F']}, U={genders['U']}")
            
            # Birth year range
            birth_years = [ind.birth_year for ind in individuals.values() if ind.birth_year]
            if birth_years:
                print(f"  Birth year range: {min(birth_years)} - {max(birth_years)}")
            
            # Missing parents
            missing_parents = sum(1 for ind in individuals.values() if not ind.parent_families)
            print(f"  Missing parents: {missing_parents}/{len(individuals)} ({missing_parents/len(individuals)*100:.1f}%)")
            
            total_individuals += len(individuals)
            total_families += len(families)
            
        except Exception as e:
            print(f"  Error parsing file: {e}")
    
    print(f"\nTotal across all files:")
    print(f"  Individuals: {total_individuals}")
    print(f"  Families: {total_families}")
    
    return 0

def gui_mode(args):
    """Launch GUI mode."""
    print("Launching OhanaAI GUI...")
    
    try:
        app = OhanaAIGUI(args.config)
        app.run()
    except Exception as e:
        print(f"Error launching GUI: {e}")
        return 1
    
    return 0

def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="OhanaAI - Genealogical Parent Prediction using Graph Neural Networks"
    )
    
    # Global arguments
    parser.add_argument(
        "--config", "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)"
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)"
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # GUI mode (default)
    gui_parser = subparsers.add_parser("gui", help="Launch GUI interface")
    
    # Training command
    train_parser = subparsers.add_parser("train", help="Train OhanaAI model")
    train_parser.add_argument(
        "gedcom_files",
        nargs="+",
        help="GEDCOM files to train on"
    )
    train_parser.add_argument(
        "--epochs", "-e",
        type=int,
        help="Number of training epochs (overrides config)"
    )
    
    # Prediction command
    predict_parser = subparsers.add_parser("predict", help="Run parent predictions")
    predict_parser.add_argument(
        "gedcom_files",
        nargs="+",
        help="GEDCOM files to analyze"
    )
    predict_parser.add_argument(
        "--model", "-m",
        help="Path to trained model file (default: from config)"
    )
    predict_parser.add_argument(
        "--output", "-o",
        help="Output directory for results (default: from config)"
    )
    
    # Deduplication command
    dedup_parser = subparsers.add_parser("deduplicate", help="Detect and merge duplicates")
    dedup_parser.add_argument(
        "gedcom_files",
        nargs="+",
        help="GEDCOM files to deduplicate"
    )
    dedup_parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=0.95,
        help="Auto-merge threshold (default: 0.95)"
    )
    
    # Info command
    info_parser = subparsers.add_parser("info", help="Show GEDCOM file information")
    info_parser.add_argument(
        "gedcom_files",
        nargs="+",
        help="GEDCOM files to analyze"
    )
    
    # Parse arguments
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    
    # Check if config file exists
    if not os.path.exists(args.config):
        print(f"Config file not found: {args.config}")
        print("Creating default config file...")
        
        # Create default config directory
        config_dir = os.path.dirname(args.config) or "."
        os.makedirs(config_dir, exist_ok=True)
        
        # Copy default config
        default_config_path = os.path.join(os.path.dirname(__file__), "..", "config.yaml")
        if os.path.exists(default_config_path):
            import shutil
            shutil.copy(default_config_path, args.config)
            print(f"Default config created at: {args.config}")
        else:
            print("Warning: Could not create default config")
    
    # Execute command
    if not args.command or args.command == "gui":
        return gui_mode(args)
    elif args.command == "train":
        return cli_train(args)
    elif args.command == "predict":
        return cli_predict(args)
    elif args.command == "deduplicate":
        return cli_deduplicate(args)
    elif args.command == "info":
        return cli_info(args)
    else:
        parser.print_help()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)