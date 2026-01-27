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
from dataclasses import dataclass, field

import numpy as np

# Import from prepare_data
from prepare_data import (
    parse_gedcom, prepare_training_data, Individual,
    detect_ethnic_origin, get_era_occupation, REGIONAL_NAMES
)


ETHNIC_CLASSES = [
    'irish', 'german', 'italian', 'polish',
    'scandinavian', 'scottish', 'jewish', 'portuguese',
    'hawaiian', 'chinese', 'japanese', 'filipino'
]


@dataclass
class AttributePrediction:
    """Predicted attributes for a missing relative."""
    predicted_birth_year: Optional[int] = None
    birth_year_range: Tuple[int, int] = (0, 0)
    birth_year_confidence: float = 0.0
    ethnic_origin: Optional[str] = None
    ethnic_origin_confidence: float = 0.0
    predicted_location: Optional[str] = None
    predicted_surname: Optional[str] = None
    predicted_given_names: List[str] = field(default_factory=list)
    predicted_occupation: Optional[str] = None


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
    father_attributes: Optional[AttributePrediction] = None
    mother_attributes: Optional[AttributePrediction] = None


def load_model(model_path: Path) -> Dict:
    """Load model weights."""
    weights = np.load(model_path)
    return {k: weights[k] for k in weights.files}


def predict(features: np.ndarray, weights: Dict) -> np.ndarray:
    """Run inference with the model."""
    x = features

    # Handle feature dimension mismatch (224 vs old 176)
    expected_dim = weights.get('fc1_weight', np.zeros((1, 224))).shape[1]
    if x.shape[1] < expected_dim:
        padding = np.zeros((x.shape[0], expected_dim - x.shape[1]))
        x = np.concatenate([x, padding], axis=1)
    elif x.shape[1] > expected_dim:
        x = x[:, :expected_dim]

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


def predict_attributes(
    person: Individual,
    individuals: Dict[str, Individual],
    relation_type: str,
    year_range: Tuple[int, int]
) -> AttributePrediction:
    """Predict attributes for a missing relative using heuristics + family patterns."""
    attrs = AttributePrediction()

    # Predict birth year based on person's birth year
    if person.birth_year:
        if relation_type == 'father':
            # Average father age gap
            gap = 28
            attrs.predicted_birth_year = person.birth_year - gap
            attrs.birth_year_range = (person.birth_year - 50, person.birth_year - 18)
            attrs.birth_year_confidence = 0.7
        elif relation_type == 'mother':
            gap = 25
            attrs.predicted_birth_year = person.birth_year - gap
            attrs.birth_year_range = (person.birth_year - 45, person.birth_year - 16)
            attrs.birth_year_confidence = 0.7

    # Predict surname
    if relation_type == 'father':
        attrs.predicted_surname = person.surname
    elif relation_type == 'mother':
        # For mother, try to infer maiden name from maternal line
        if person.mother and person.mother in individuals:
            mother = individuals[person.mother]
            if mother.mother and mother.mother in individuals:
                maternal_grandma = individuals[mother.mother]
                if maternal_grandma.surname:
                    attrs.predicted_surname = maternal_grandma.surname
        if not attrs.predicted_surname:
            attrs.predicted_surname = "Unknown"

    # Predict ethnic origin from person's surname
    ethnic, conf = detect_ethnic_origin(person.surname, person.birth_place)
    if ethnic:
        attrs.ethnic_origin = ethnic
        attrs.ethnic_origin_confidence = conf

        # Get era-appropriate given names
        if ethnic in REGIONAL_NAMES:
            gender = 'male' if relation_type == 'father' else 'female'
            attrs.predicted_given_names = REGIONAL_NAMES[ethnic].get(gender, [])[:5]

    # Predict location from person's birth place
    if person.birth_place and person.birth_place.components:
        attrs.predicted_location = person.birth_place.raw

    # Predict occupation
    if attrs.predicted_birth_year:
        attrs.predicted_occupation = get_era_occupation(
            attrs.predicted_birth_year,
            person.birth_place
        )

    return attrs


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

    # Get year range for attribute predictions
    all_years = [ind.birth_year for ind in individuals.values() if ind.birth_year]
    year_range = (min(all_years) if all_years else 1700, max(all_years) if all_years else 2024)

    # Build prediction results
    results: List[Prediction] = []
    for i, node_id in enumerate(node_ids):
        ind = individuals[node_id]
        name = f"{ind.given_name or ''} {ind.surname or ''}".strip() or node_id

        # Generate attribute predictions for missing parents
        father_attrs = None
        mother_attrs = None

        if ind.father is None and predictions[i, 0] > 0.3:
            father_attrs = predict_attributes(ind, individuals, 'father', year_range)

        if ind.mother is None and predictions[i, 1] > 0.3:
            mother_attrs = predict_attributes(ind, individuals, 'mother', year_range)

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
            has_mother=ind.mother is not None,
            father_attributes=father_attrs,
            mother_attributes=mother_attrs
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

            # Show predicted attributes
            if pred.father_attributes:
                fa = pred.father_attributes
                print("    Predicted attributes:")
                if fa.predicted_surname:
                    print(f"      Surname: {fa.predicted_surname}")
                if fa.predicted_birth_year:
                    print(f"      Birth year: ~{fa.predicted_birth_year} (range: {fa.birth_year_range[0]}-{fa.birth_year_range[1]})")
                if fa.ethnic_origin:
                    print(f"      Ethnic origin: {fa.ethnic_origin.title()} ({fa.ethnic_origin_confidence:.0%})")
                if fa.predicted_given_names:
                    print(f"      Common names: {', '.join(fa.predicted_given_names[:5])}")
                if fa.predicted_occupation:
                    print(f"      Likely occupation: {fa.predicted_occupation}")
                if fa.predicted_location:
                    print(f"      Probable location: {fa.predicted_location}")

            if args.find_candidates:
                candidates = find_candidates(individuals[pred.person_id], individuals, 'father')
                if candidates:
                    print("    Possible fathers in tree:")
                    for cand_id, cand_name, score, reasons in candidates[:3]:
                        cand = individuals[cand_id]
                        year_str = f" (b. {cand.birth_year})" if cand.birth_year else ""
                        print(f"      • {cand_name}{year_str} - {reasons}")

        if not pred.has_mother and pred.missing_mother > args.threshold:
            print(f"  ⚠ Missing Mother (confidence: {pred.missing_mother:.1%})")

            # Show predicted attributes
            if pred.mother_attributes:
                ma = pred.mother_attributes
                print("    Predicted attributes:")
                if ma.predicted_surname:
                    print(f"      Maiden name: {ma.predicted_surname}")
                if ma.predicted_birth_year:
                    print(f"      Birth year: ~{ma.predicted_birth_year} (range: {ma.birth_year_range[0]}-{ma.birth_year_range[1]})")
                if ma.ethnic_origin:
                    print(f"      Ethnic origin: {ma.ethnic_origin.title()} ({ma.ethnic_origin_confidence:.0%})")
                if ma.predicted_given_names:
                    print(f"      Common names: {', '.join(ma.predicted_given_names[:5])}")
                if ma.predicted_occupation:
                    print(f"      Likely occupation: {ma.predicted_occupation}")
                if ma.predicted_location:
                    print(f"      Probable location: {ma.predicted_location}")

            if args.find_candidates:
                candidates = find_candidates(individuals[pred.person_id], individuals, 'mother')
                if candidates:
                    print("    Possible mothers in tree:")
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

    def attrs_to_dict(attrs: Optional[AttributePrediction]) -> Optional[Dict]:
        if attrs is None:
            return None
        return {
            'predicted_birth_year': attrs.predicted_birth_year,
            'birth_year_range': list(attrs.birth_year_range),
            'birth_year_confidence': round(attrs.birth_year_confidence, 3),
            'ethnic_origin': attrs.ethnic_origin,
            'ethnic_origin_confidence': round(attrs.ethnic_origin_confidence, 3),
            'predicted_location': attrs.predicted_location,
            'predicted_surname': attrs.predicted_surname,
            'predicted_given_names': attrs.predicted_given_names,
            'predicted_occupation': attrs.predicted_occupation
        }

    output_data = {
        'source': str(gedcom_path),
        'total_individuals': len(results),
        'model_version': '2.0.0',
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
                'father_attributes': attrs_to_dict(r.father_attributes),
                'mother_attributes': attrs_to_dict(r.mother_attributes)
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
