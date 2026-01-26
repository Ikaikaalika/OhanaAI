#!/usr/bin/env python3
"""
OhanaAI - Run predictions on a GEDCOM file

Usage: python predict.py path/to/file.ged [--top N]
"""

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

import numpy as np

# Import from prepare_data
from prepare_data import parse_gedcom, prepare_training_data, Individual


@dataclass
class Prediction:
    person_id: str
    name: str
    missing_father: float
    missing_mother: float
    missing_spouse: float
    missing_children: float
    missing_siblings: float
    birth_year: Optional[int]
    has_father: bool
    has_mother: bool


def load_model(model_path: Path) -> Dict:
    """Load model weights."""
    weights = np.load(model_path)
    return {k: weights[k] for k in weights.files}


def predict(features: np.ndarray, weights: Dict) -> np.ndarray:
    """Run inference with the model."""
    x = features

    # Layer 1
    x = x @ weights['fc1_weight'].T + weights['fc1_bias']
    x = np.maximum(x, 0)  # ReLU

    # Layer 2
    x = x @ weights['fc2_weight'].T + weights['fc2_bias']
    x = np.maximum(x, 0)

    # Layer 3
    x = x @ weights['fc3_weight'].T + weights['fc3_bias']
    x = np.maximum(x, 0)

    # Output
    x = x @ weights['out_weight'].T + weights['out_bias']
    x = 1 / (1 + np.exp(-x))  # Sigmoid

    return x


def find_candidates(
    person: Individual,
    individuals: Dict[str, Individual],
    relation_type: str
) -> List[Tuple[str, str, float]]:
    """Find candidate relatives for a person."""
    candidates = []

    for cand_id, cand in individuals.items():
        if cand_id == person.id:
            continue

        # Skip known relatives
        if person.father == cand_id or person.mother == cand_id:
            continue
        if cand_id in person.spouses:
            continue

        score = 0.0
        reasons = []

        if relation_type == 'father':
            if cand.gender != 'M':
                continue
            # Age check
            if cand.birth_year and person.birth_year:
                age_diff = person.birth_year - cand.birth_year
                if 18 <= age_diff <= 50:
                    score += 0.3
                    reasons.append(f"Age gap: {age_diff} years")
                elif age_diff < 15 or age_diff > 60:
                    continue
            # Surname match
            if person.surname and cand.surname:
                if person.surname.lower() == cand.surname.lower():
                    score += 0.5
                    reasons.append("Same surname")

        elif relation_type == 'mother':
            if cand.gender != 'F':
                continue
            if cand.birth_year and person.birth_year:
                age_diff = person.birth_year - cand.birth_year
                if 16 <= age_diff <= 45:
                    score += 0.3
                    reasons.append(f"Age gap: {age_diff} years")
                elif age_diff < 12 or age_diff > 55:
                    continue

        elif relation_type == 'spouse':
            if cand.birth_year and person.birth_year:
                age_diff = abs(person.birth_year - cand.birth_year)
                if age_diff <= 10:
                    score += 0.3
                    reasons.append(f"Similar age ({age_diff} years apart)")
                elif age_diff > 25:
                    continue

        # Location match
        if person.birth_place and cand.birth_place:
            p_comps = set(c.lower() for c in person.birth_place.components)
            c_comps = set(c.lower() for c in cand.birth_place.components)
            common = p_comps & c_comps
            if common:
                score += 0.2 * len(common)
                reasons.append(f"Location: {', '.join(common)}")

        if score > 0:
            name = f"{cand.given_name or ''} {cand.surname or ''}".strip() or cand_id
            reason_str = "; ".join(reasons)
            candidates.append((cand_id, name, score, reason_str))

    # Sort by score
    candidates.sort(key=lambda x: -x[2])
    return candidates[:5]


def main():
    parser = argparse.ArgumentParser(description='Run predictions on GEDCOM file')
    parser.add_argument('gedcom_file', help='Path to GEDCOM file')
    parser.add_argument('--model', default='models/family_tree_gnn/best_model.npz',
                        help='Path to model weights')
    parser.add_argument('--top', type=int, default=20,
                        help='Number of top predictions to show')
    parser.add_argument('--threshold', type=float, default=0.5,
                        help='Prediction threshold')
    parser.add_argument('--find-candidates', action='store_true',
                        help='Find candidate relatives for missing relations')
    args = parser.parse_args()

    gedcom_path = Path(args.gedcom_file)
    model_path = Path(args.model)

    # Handle relative paths
    if not model_path.is_absolute():
        model_path = Path(__file__).parent.parent.parent / model_path

    if not gedcom_path.exists():
        print(f"Error: GEDCOM file not found: {gedcom_path}")
        sys.exit(1)

    if not model_path.exists():
        print(f"Error: Model not found: {model_path}")
        sys.exit(1)

    print(f"Loading GEDCOM: {gedcom_path}")
    individuals, families = parse_gedcom(gedcom_path)
    print(f"  {len(individuals)} individuals, {len(families)} families")

    print(f"\nLoading model: {model_path}")
    weights = load_model(model_path)

    print("\nPreparing features...")
    training_data = prepare_training_data(individuals, families)
    features = np.array(training_data['nodeFeatures'], dtype=np.float32)
    node_ids = training_data['nodeIds']

    print(f"Running predictions on {len(features)} individuals...")
    predictions = predict(features, weights)

    # Build prediction results
    results: List[Prediction] = []
    for i, node_id in enumerate(node_ids):
        ind = individuals[node_id]
        name = f"{ind.given_name or ''} {ind.surname or ''}".strip() or node_id

        pred = Prediction(
            person_id=node_id,
            name=name,
            missing_father=predictions[i, 0],
            missing_mother=predictions[i, 1],
            missing_spouse=predictions[i, 2],
            missing_children=predictions[i, 3],
            missing_siblings=predictions[i, 4],
            birth_year=ind.birth_year,
            has_father=ind.father is not None,
            has_mother=ind.mother is not None
        )
        results.append(pred)

    # Filter and sort
    # Show people with high missing parent scores who don't have parents
    missing_parents = [r for r in results
                       if (r.missing_father > args.threshold and not r.has_father) or
                          (r.missing_mother > args.threshold and not r.has_mother)]

    missing_parents.sort(key=lambda x: max(
        x.missing_father if not x.has_father else 0,
        x.missing_mother if not x.has_mother else 0
    ), reverse=True)

    print("\n" + "=" * 70)
    print("PEOPLE LIKELY MISSING PARENTS")
    print("=" * 70)

    for pred in missing_parents[:args.top]:
        print(f"\n{pred.name}")
        if pred.birth_year:
            print(f"  Born: {pred.birth_year}")

        if not pred.has_father and pred.missing_father > args.threshold:
            print(f"  ⚠ Missing Father (confidence: {pred.missing_father:.1%})")
            if args.find_candidates:
                candidates = find_candidates(individuals[pred.person_id], individuals, 'father')
                if candidates:
                    print("    Possible fathers:")
                    for cand_id, cand_name, score, reasons in candidates[:3]:
                        cand = individuals[cand_id]
                        year_str = f" (b. {cand.birth_year})" if cand.birth_year else ""
                        print(f"      • {cand_name}{year_str} - {reasons}")

        if not pred.has_mother and pred.missing_mother > args.threshold:
            print(f"  ⚠ Missing Mother (confidence: {pred.missing_mother:.1%})")
            if args.find_candidates:
                candidates = find_candidates(individuals[pred.person_id], individuals, 'mother')
                if candidates:
                    print("    Possible mothers:")
                    for cand_id, cand_name, score, reasons in candidates[:3]:
                        cand = individuals[cand_id]
                        year_str = f" (b. {cand.birth_year})" if cand.birth_year else ""
                        print(f"      • {cand_name}{year_str} - {reasons}")

    # Summary stats
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    total_missing_father = sum(1 for r in results if not r.has_father and r.missing_father > args.threshold)
    total_missing_mother = sum(1 for r in results if not r.has_mother and r.missing_mother > args.threshold)
    print(f"Total individuals: {len(results)}")
    print(f"Missing fathers (high confidence): {total_missing_father}")
    print(f"Missing mothers (high confidence): {total_missing_mother}")

    # Save full results
    output_path = gedcom_path.with_suffix('.predictions.json')
    output_data = {
        'source': str(gedcom_path),
        'total_individuals': len(results),
        'predictions': [
            {
                'id': r.person_id,
                'name': r.name,
                'birth_year': r.birth_year,
                'has_father': r.has_father,
                'has_mother': r.has_mother,
                'missing_father_prob': round(float(r.missing_father), 4),
                'missing_mother_prob': round(float(r.missing_mother), 4),
                'missing_spouse_prob': round(float(r.missing_spouse), 4),
            }
            for r in results
            if r.missing_father > 0.3 or r.missing_mother > 0.3
        ]
    }

    with open(output_path, 'w') as f:
        json.dump(output_data, f, indent=2)

    print(f"\nFull predictions saved to: {output_path}")


if __name__ == '__main__':
    main()
