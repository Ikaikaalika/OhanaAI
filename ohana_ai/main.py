"""
Main entry point for OhanaAI - Genealogical Parent Prediction System.
Provides both CLI and GUI interfaces for training models and making predictions.
"""

import argparse
import logging
import shutil
import sys
from pathlib import Path

from .core.config import load_config, setup_logging
from .core.exceptions import ConfigError, GedcomParseError, OhanaAIError
from .data_deduplication import deduplicate_gedcom_files
from .gedcom_parser import parse_gedcom_file
from .gui import OhanaAIGUI
from .predictor import predict_parents
from .trainer import train_model


def cli_train(args) -> int:
    """Train model via CLI.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = logging.getLogger(__name__)
    logger.info("Training OhanaAI model...")
    logger.info(f"GEDCOM files: {args.gedcom_files}")
    logger.info(f"Config: {args.config}")

    try:
        config = load_config(args.config)
        if args.epochs:
            config.epochs = args.epochs

        trainer = train_model(args.gedcom_files, config)
        logger.info(f"Training completed! Model saved to: {config.model_path}")

        # Print training summary
        if trainer.training_history:
            last_metrics = trainer.training_history[-1]
            print(f"Final training accuracy: {last_metrics.train_accuracy:.4f}")
            print(f"Final validation accuracy: {last_metrics.val_accuracy:.4f}")
            print(f"Total epochs: {last_metrics.epoch}")

    except (OhanaAIError, ConfigError, GedcomParseError) as e:
        logger.error(f"Training failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during training: {e}", exc_info=True)
        return 1

    return 0


def cli_predict(args) -> int:
    """Run predictions via CLI.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = logging.getLogger(__name__)
    logger.info("Running parent predictions...")
    logger.info(f"GEDCOM files: {args.gedcom_files}")
    logger.info(f"Model: {args.model}")
    logger.info(f"Config: {args.config}")

    try:
        config = load_config(args.config)
        model_path = args.model if args.model else config.model_path
        output_dir = args.output if args.output else config.outputs_dir

        predictions = predict_parents(args.gedcom_files, model_path, config)

        print(f"Generated {len(predictions)} predictions")

        # Show top predictions
        high_confidence = [p for p in predictions if p.confidence_score > 0.7]
        print(f"High confidence predictions (>0.7): {len(high_confidence)}")

        if high_confidence:
            print("\nTop 5 predictions:")
            for i, pred in enumerate(high_confidence[:5]):
                print(
                    f"{i+1}. {pred.child_name} -> {pred.candidate_parent_name} "
                    f"(confidence: {pred.confidence_score:.3f})"
                )

        print(f"Results exported to {output_dir} directory")

    except (OhanaAIError, ConfigError, GedcomParseError) as e:
        logger.error(f"Prediction failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during prediction: {e}", exc_info=True)
        return 1

    return 0


def cli_deduplicate(args) -> int:
    """Run deduplication via CLI.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = logging.getLogger(__name__)
    logger.info("Detecting and removing duplicates...")
    logger.info(f"GEDCOM files: {args.gedcom_files}")

    try:
        individuals, families, remaining_duplicates = deduplicate_gedcom_files(
            args.gedcom_files, auto_merge_threshold=args.threshold
        )

        print("Deduplication completed!")
        print(f"Final counts: {len(individuals)} individuals, {len(families)} families")
        print(f"Remaining potential duplicates: {len(remaining_duplicates)}")

        if remaining_duplicates:
            print("\nTop potential duplicates for manual review:")
            for i, dup in enumerate(remaining_duplicates[:5]):
                print(
                    f"{i+1}. {dup.individual1_id} <-> {dup.individual2_id} "
                    f"(similarity: {dup.similarity_score:.3f})"
                )

    except (OhanaAIError, GedcomParseError) as e:
        logger.error(f"Deduplication failed: {e}")
        return 1
    except Exception as e:
        logger.error(f"Unexpected error during deduplication: {e}", exc_info=True)
        return 1

    return 0


def cli_info(args) -> int:
    """Show information about GEDCOM files.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = logging.getLogger(__name__)
    print("GEDCOM File Information:")

    total_individuals = 0
    total_families = 0
    errors = 0

    for gedcom_file in args.gedcom_files:
        try:
            print(f"\nFile: {gedcom_file}")
            individuals, families = parse_gedcom_file(gedcom_file)

            print(f"  Individuals: {len(individuals)}")
            print(f"  Families: {len(families)}")

            # Gender distribution
            genders = {"M": 0, "F": 0, "U": 0}
            for ind in individuals.values():
                genders[ind.gender] = genders.get(ind.gender, 0) + 1

            print(
                f"  Gender distribution: M={genders['M']}, F={genders['F']}, U={genders['U']}"
            )

            # Birth year range
            birth_years = [
                ind.birth_year for ind in individuals.values() if ind.birth_year
            ]
            if birth_years:
                print(f"  Birth year range: {min(birth_years)} - {max(birth_years)}")

            # Missing parents
            missing_parents = sum(
                1 for ind in individuals.values() if not ind.parent_families
            )
            if individuals:
                missing_percent = missing_parents / len(individuals) * 100
                print(
                    f"  Missing parents: {missing_parents}/{len(individuals)} ({missing_percent:.1f}%)"
                )

            total_individuals += len(individuals)
            total_families += len(families)

        except GedcomParseError as e:
            logger.error(f"Error parsing {gedcom_file}: {e}")
            print(f"  Error parsing file: {e}")
            errors += 1
        except Exception as e:
            logger.error(f"Unexpected error parsing {gedcom_file}: {e}", exc_info=True)
            print(f"  Unexpected error: {e}")
            errors += 1

    print(f"\nTotal across all files:")
    print(f"  Individuals: {total_individuals}")
    print(f"  Families: {total_families}")
    if errors:
        print(f"  Files with errors: {errors}")

    return 1 if errors > 0 else 0


def gui_mode(args) -> int:
    """Launch GUI mode.

    Args:
        args: Command line arguments

    Returns:
        Exit code (0 for success, 1 for error)
    """
    logger = logging.getLogger(__name__)
    logger.info("Launching OhanaAI GUI...")

    try:
        config = load_config(args.config)
        app = OhanaAIGUI(config)
        app.run()
    except ConfigError as e:
        logger.error(f"Configuration error: {e}")
        print(f"Configuration error: {e}")
        return 1
    except Exception as e:
        logger.error(f"Error launching GUI: {e}", exc_info=True)
        print(f"Error launching GUI: {e}")
        return 1

    return 0


def main() -> int:
    """Main entry point.

    Returns:
        Exit code (0 for success, 1 for error)
    """
    parser = argparse.ArgumentParser(
        description="OhanaAI - Genealogical Parent Prediction using Graph Neural Networks"
    )

    # Global arguments
    parser.add_argument(
        "--config",
        "-c",
        default="config.yaml",
        help="Configuration file path (default: config.yaml)",
    )
    parser.add_argument(
        "--log-level",
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
        help="Logging level (default: INFO)",
    )

    subparsers = parser.add_subparsers(dest="command", help="Available commands")

    # GUI mode (default)
    subparsers.add_parser("gui", help="Launch GUI interface")

    # Training command
    train_parser = subparsers.add_parser("train", help="Train OhanaAI model")
    train_parser.add_argument(
        "gedcom_files", nargs="+", help="GEDCOM files to train on"
    )
    train_parser.add_argument(
        "--epochs", "-e", type=int, help="Number of training epochs (overrides config)"
    )

    # Prediction command
    predict_parser = subparsers.add_parser("predict", help="Run parent predictions")
    predict_parser.add_argument(
        "gedcom_files", nargs="+", help="GEDCOM files to analyze"
    )
    predict_parser.add_argument(
        "--model", "-m", help="Path to trained model file (default: from config)"
    )
    predict_parser.add_argument(
        "--output", "-o", help="Output directory for results (default: from config)"
    )

    # Deduplication command
    dedup_parser = subparsers.add_parser(
        "deduplicate", help="Detect and merge duplicates"
    )
    dedup_parser.add_argument(
        "gedcom_files", nargs="+", help="GEDCOM files to deduplicate"
    )
    dedup_parser.add_argument(
        "--threshold",
        "-t",
        type=float,
        default=0.95,
        help="Auto-merge threshold (default: 0.95)",
    )

    # Info command
    info_parser = subparsers.add_parser("info", help="Show GEDCOM file information")
    info_parser.add_argument("gedcom_files", nargs="+", help="GEDCOM files to analyze")

    # Parse arguments
    args = parser.parse_args()

    # Check if config file exists, create default if missing
    config_path = Path(args.config)
    if not config_path.exists():
        print(f"Config file not found: {config_path}")
        print("Creating default config file...")

        # Create default config directory
        config_path.parent.mkdir(parents=True, exist_ok=True)

        # Copy default config from package
        default_config_path = Path(__file__).parent.parent / "config.yaml"
        if default_config_path.exists():
            shutil.copy(default_config_path, config_path)
            print(f"Default config created at: {config_path}")
        else:
            print("Warning: Could not find default config to copy")
            return 1

    # Load config for logging setup
    try:
        config = load_config(config_path)
        # Override log level from command line
        if args.log_level:
            config.log_level = args.log_level
        setup_logging(config)
    except ConfigError as e:
        print(f"Configuration error: {e}")
        return 1

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
