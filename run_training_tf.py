"""
Run TensorFlow training for OhanaAI model.
"""

import argparse
import glob

from ohana_ai.gedcom_parser import parse_gedcom_file
from ohana_ai.trainer_tf import OhanaAITrainerTF

def main():
    parser = argparse.ArgumentParser(description="Train OhanaAI TensorFlow model.")
    parser.add_argument("--config", type=str, default="config.yaml", help="Path to config file.")
    parser.add_argument("--gedcoms", type=str, default="uploads/*.ged", help="Glob pattern for GEDCOM files.")
    args = parser.parse_args()

    gedcom_files = glob.glob(args.gedcoms)
    if not gedcom_files:
        print(f"No GEDCOM files found matching: {args.gedcoms}")
        return

    all_individuals = {}
    all_families = {}

    for gedcom_file in gedcom_files:
        individuals, families = parse_gedcom_file(gedcom_file)
        all_individuals.update(individuals)
        all_families.update(families)

    trainer = OhanaAITrainerTF(config_path=args.config)
    trainer.prepare_data(all_individuals, all_families)
    trainer.train()

if __name__ == "__main__":
    main()
