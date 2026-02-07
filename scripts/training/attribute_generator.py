#!/usr/bin/env python3
"""
OhanaAI - Attribute Generator for Missing Ancestors

Instead of suggesting existing people, this model generates predicted
attributes for missing ancestors based on:
1. The child's attributes (birth year, location, surname)
2. The spouse's attributes (if known)
3. Patterns from known ancestors in the tree
4. Historical/statistical priors

Generated attributes:
- Estimated birth year (with confidence range)
- Predicted surname
- Likely birth location
- Estimated death year
- Occupation likelihood
- Ethnicity/origin estimation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
from collections import Counter
import re


@dataclass
class PredictedAncestor:
    """Predicted attributes for a missing ancestor."""
    relationship: str  # 'father' or 'mother'
    confidence: float  # Overall prediction confidence

    # Name predictions
    predicted_surname: str
    surname_confidence: float
    possible_given_names: List[str]  # Common names for era/location

    # Temporal predictions
    estimated_birth_year: int
    birth_year_range: Tuple[int, int]  # (min, max)
    estimated_death_year: Optional[int]
    death_year_range: Optional[Tuple[int, int]]
    estimated_lifespan: int

    # Location predictions
    predicted_birth_location: str
    location_confidence: float
    location_components: Dict[str, str]  # city, county, state, country

    # Additional predictions
    predicted_occupation: Optional[str]
    occupation_confidence: float
    ethnic_origin: Optional[str]
    origin_confidence: float

    # Reasoning
    reasoning: List[str]  # Explanations for predictions


class AncestorAttributeGenerator:
    """Generates predicted attributes for missing ancestors."""

    def __init__(self):
        # Historical data for name generation
        self.male_names_by_era = {
            1500: ['John', 'William', 'Thomas', 'Richard', 'Robert', 'Henry', 'Edward', 'James'],
            1600: ['John', 'William', 'Thomas', 'James', 'Robert', 'Richard', 'George', 'Samuel'],
            1700: ['John', 'William', 'James', 'Thomas', 'Samuel', 'Joseph', 'Benjamin', 'David'],
            1800: ['John', 'William', 'James', 'George', 'Charles', 'Thomas', 'Joseph', 'Henry'],
            1850: ['John', 'William', 'James', 'George', 'Charles', 'Joseph', 'Samuel', 'Henry', 'David'],
            1900: ['John', 'William', 'James', 'Robert', 'Joseph', 'Charles', 'George', 'Edward'],
            1950: ['James', 'Robert', 'John', 'Michael', 'David', 'William', 'Richard', 'Thomas'],
        }

        self.female_names_by_era = {
            1500: ['Mary', 'Elizabeth', 'Margaret', 'Anne', 'Alice', 'Joan', 'Catherine', 'Agnes'],
            1600: ['Mary', 'Elizabeth', 'Sarah', 'Anne', 'Margaret', 'Jane', 'Martha', 'Hannah'],
            1700: ['Mary', 'Elizabeth', 'Sarah', 'Hannah', 'Margaret', 'Martha', 'Abigail', 'Rebecca'],
            1800: ['Mary', 'Elizabeth', 'Sarah', 'Ann', 'Margaret', 'Jane', 'Martha', 'Hannah'],
            1850: ['Mary', 'Elizabeth', 'Sarah', 'Margaret', 'Anna', 'Emma', 'Alice', 'Catherine'],
            1900: ['Mary', 'Helen', 'Margaret', 'Anna', 'Ruth', 'Elizabeth', 'Dorothy', 'Marie'],
            1950: ['Mary', 'Linda', 'Patricia', 'Barbara', 'Susan', 'Nancy', 'Deborah', 'Sandra'],
        }

        # Hawaiian names for Hawaiian genealogies
        self.hawaiian_male_names = [
            'Keoni', 'Kekoa', 'Kalani', 'Makoa', 'Kainoa', 'Ikaika', 'Kawika', 'Kaleo',
            'Keanu', 'Moku', 'Nainoa', 'Palani', 'Kamaka', 'Kanoa', 'Kapono', 'Kahoku'
        ]

        self.hawaiian_female_names = [
            'Leilani', 'Malia', 'Kealoha', 'Kalea', 'Kailani', 'Moana', 'Nani', 'Mahina',
            'Lani', 'Kalena', 'Noelani', 'Haunani', 'Kawena', 'Makana', 'Mele', 'Pualani'
        ]

        # Average parent age by era
        self.avg_parent_age = {
            1500: {'father': 30, 'mother': 25},
            1600: {'father': 28, 'mother': 24},
            1700: {'father': 27, 'mother': 23},
            1800: {'father': 28, 'mother': 24},
            1850: {'father': 29, 'mother': 25},
            1900: {'father': 28, 'mother': 24},
            1950: {'father': 27, 'mother': 24},
            2000: {'father': 31, 'mother': 28},
        }

        # Average lifespan by era
        self.avg_lifespan = {
            1500: 45,
            1600: 45,
            1700: 50,
            1800: 55,
            1850: 55,
            1900: 60,
            1950: 70,
            2000: 78,
        }

    def predict_ancestor(
        self,
        child: Dict,
        relationship: str,  # 'father' or 'mother'
        spouse: Optional[Dict] = None,
        known_ancestors: List[Dict] = None,
        siblings: List[Dict] = None
    ) -> PredictedAncestor:
        """Generate predicted attributes for a missing ancestor."""

        reasoning = []
        known_ancestors = known_ancestors or []
        siblings = siblings or []

        # === Birth Year Prediction ===
        birth_year, birth_range = self._predict_birth_year(
            child, relationship, spouse, reasoning
        )

        # === Surname Prediction ===
        surname, surname_conf = self._predict_surname(
            child, relationship, spouse, known_ancestors, reasoning
        )

        # === Location Prediction ===
        location, loc_conf, loc_components = self._predict_location(
            child, spouse, known_ancestors, reasoning
        )

        # === Given Names ===
        given_names = self._suggest_given_names(
            birth_year, relationship, location, known_ancestors, reasoning
        )

        # === Death Year Prediction ===
        death_year, death_range = self._predict_death_year(
            birth_year, birth_range, child, reasoning
        )

        # === Lifespan ===
        lifespan = self._estimate_lifespan(birth_year)

        # === Occupation ===
        occupation, occ_conf = self._predict_occupation(
            birth_year, location, relationship, known_ancestors, reasoning
        )

        # === Ethnic Origin ===
        origin, origin_conf = self._predict_origin(
            surname, location, known_ancestors, reasoning
        )

        # === Overall Confidence ===
        confidence = np.mean([surname_conf, loc_conf, 0.7])  # Base confidence

        return PredictedAncestor(
            relationship=relationship,
            confidence=confidence,
            predicted_surname=surname,
            surname_confidence=surname_conf,
            possible_given_names=given_names,
            estimated_birth_year=birth_year,
            birth_year_range=birth_range,
            estimated_death_year=death_year,
            death_year_range=death_range,
            estimated_lifespan=lifespan,
            predicted_birth_location=location,
            location_confidence=loc_conf,
            location_components=loc_components,
            predicted_occupation=occupation,
            occupation_confidence=occ_conf,
            ethnic_origin=origin,
            origin_confidence=origin_conf,
            reasoning=reasoning
        )

    def _predict_birth_year(
        self,
        child: Dict,
        relationship: str,
        spouse: Optional[Dict],
        reasoning: List[str]
    ) -> Tuple[int, Tuple[int, int]]:
        """Predict birth year of missing ancestor."""

        child_birth = child.get('birthYear')

        if not child_birth:
            # Try to estimate from other data
            if child.get('deathYear'):
                child_birth = child['deathYear'] - 60  # Assume ~60 year life
            else:
                child_birth = 1900  # Default

        # Get era-appropriate parent age
        era = self._get_era(child_birth - 25)
        avg_age = self.avg_parent_age.get(era, {'father': 28, 'mother': 24})
        parent_age = avg_age[relationship]

        # Base prediction
        birth_year = child_birth - parent_age

        # Adjust based on spouse if known
        if spouse and spouse.get('birthYear'):
            spouse_birth = spouse['birthYear']
            if relationship == 'father':
                # Father usually 2-5 years older than mother
                birth_year = spouse_birth - 3
            else:
                # Mother usually 2-5 years younger than father
                birth_year = spouse_birth + 3

            reasoning.append(f"Estimated from spouse's birth year ({spouse_birth})")
        else:
            reasoning.append(f"Estimated {parent_age} years before child's birth ({child_birth})")

        # Calculate range
        min_year = birth_year - 10
        max_year = birth_year + 5

        return birth_year, (min_year, max_year)

    def _predict_surname(
        self,
        child: Dict,
        relationship: str,
        spouse: Optional[Dict],
        known_ancestors: List[Dict],
        reasoning: List[str]
    ) -> Tuple[str, float]:
        """Predict surname of missing ancestor."""

        if relationship == 'father':
            # Father typically shares surname with child
            if child.get('surname'):
                reasoning.append(f"Father's surname inherited by child")
                return child['surname'], 0.95

            # Check known paternal ancestors
            paternal_surnames = [
                a.get('surname') for a in known_ancestors
                if a.get('surname') and a.get('relationship', '').startswith('paternal')
            ]
            if paternal_surnames:
                surname = Counter(paternal_surnames).most_common(1)[0][0]
                reasoning.append(f"Surname from paternal line pattern")
                return surname, 0.8

            return "Unknown", 0.1

        else:  # mother
            # Mother's maiden name is harder to predict
            if spouse and spouse.get('surname'):
                # Often took husband's surname, maiden name unknown
                reasoning.append("Mother's maiden name unknown (married name may differ)")

            # Check if any maternal ancestors have different surname
            maternal_surnames = [
                a.get('surname') for a in known_ancestors
                if a.get('surname') and 'maternal' in a.get('relationship', '')
            ]
            if maternal_surnames:
                surname = Counter(maternal_surnames).most_common(1)[0][0]
                reasoning.append(f"Possible maiden name from maternal line")
                return surname, 0.5

            # Use child's surname as placeholder
            if child.get('surname'):
                return f"(née unknown, married {child['surname']})", 0.3

            return "Unknown", 0.1

    def _predict_location(
        self,
        child: Dict,
        spouse: Optional[Dict],
        known_ancestors: List[Dict],
        reasoning: List[str]
    ) -> Tuple[str, float, Dict[str, str]]:
        """Predict birth location of missing ancestor."""

        locations = []

        # Child's birth place
        if child.get('birthPlace'):
            locations.append(child['birthPlace'])

        # Spouse's birth place
        if spouse and spouse.get('birthPlace'):
            locations.append(spouse['birthPlace'])

        # Known ancestors' locations
        for anc in known_ancestors:
            if anc.get('birthPlace'):
                locations.append(anc['birthPlace'])

        if not locations:
            return "Unknown", 0.1, {}

        # Parse locations into components
        all_components = []
        for loc in locations:
            parts = [p.strip() for p in loc.split(',')]
            all_components.append(parts)

        # Find most common location components
        location_parts = {}

        # Get most common at each level
        for level in range(4):  # city, county, state, country
            parts_at_level = [c[level] if len(c) > level else None for c in all_components]
            parts_at_level = [p for p in parts_at_level if p]
            if parts_at_level:
                most_common = Counter(parts_at_level).most_common(1)[0][0]
                level_names = ['city', 'county', 'state', 'country']
                if level < len(level_names):
                    location_parts[level_names[level]] = most_common

        # Construct location string
        if location_parts:
            location = ', '.join(location_parts.values())
            confidence = min(len(locations) / 3, 0.9)
            reasoning.append(f"Location based on family pattern ({len(locations)} known locations)")
            return location, confidence, location_parts

        return locations[0], 0.6, {}

    def _suggest_given_names(
        self,
        birth_year: int,
        relationship: str,
        location: str,
        known_ancestors: List[Dict],
        reasoning: List[str]
    ) -> List[str]:
        """Suggest likely given names for the era and location."""

        era = self._get_era(birth_year)

        # Check if Hawaiian location
        is_hawaiian = any(term in location.lower() for term in
                         ['hawaii', 'honolulu', 'maui', 'oahu', 'kauai', 'kona', 'hilo'])

        if is_hawaiian:
            if relationship == 'father':
                names = self.hawaiian_male_names[:8]
            else:
                names = self.hawaiian_female_names[:8]
            reasoning.append("Hawaiian names suggested based on location")
        else:
            if relationship == 'father':
                names = self.male_names_by_era.get(era, self.male_names_by_era[1850])
            else:
                names = self.female_names_by_era.get(era, self.female_names_by_era[1850])
            reasoning.append(f"Names common for {era}s era")

        # Check ancestor names for family patterns
        ancestor_names = [
            a.get('givenName') for a in known_ancestors
            if a.get('givenName') and
               ((relationship == 'father' and a.get('gender') == 'M') or
                (relationship == 'mother' and a.get('gender') == 'F'))
        ]

        if ancestor_names:
            # Prioritize family names
            family_names = [n for n in ancestor_names if n in names]
            other_names = [n for n in names if n not in ancestor_names]
            names = family_names[:3] + other_names[:5]
            if family_names:
                reasoning.append("Some names from family naming patterns")

        return names[:8]

    def _predict_death_year(
        self,
        birth_year: int,
        birth_range: Tuple[int, int],
        child: Dict,
        reasoning: List[str]
    ) -> Tuple[Optional[int], Optional[Tuple[int, int]]]:
        """Predict death year based on era lifespan."""

        lifespan = self._estimate_lifespan(birth_year)
        death_year = birth_year + lifespan

        # Ensure they were alive when child was born
        child_birth = child.get('birthYear', birth_year + 25)
        if death_year < child_birth:
            death_year = child_birth + 10  # At least alive for child's early years

        death_range = (death_year - 15, death_year + 15)

        reasoning.append(f"Death estimated from {lifespan}-year average lifespan for era")

        return death_year, death_range

    def _estimate_lifespan(self, birth_year: int) -> int:
        """Estimate lifespan based on era."""
        era = self._get_era(birth_year)
        return self.avg_lifespan.get(era, 60)

    def _predict_occupation(
        self,
        birth_year: int,
        location: str,
        relationship: str,
        known_ancestors: List[Dict],
        reasoning: List[str]
    ) -> Tuple[Optional[str], float]:
        """Predict likely occupation."""

        # Check ancestor occupations
        ancestor_occs = [
            a.get('occupation') for a in known_ancestors
            if a.get('occupation')
        ]

        if ancestor_occs:
            occ = Counter(ancestor_occs).most_common(1)[0][0]
            reasoning.append(f"Occupation from family pattern")
            return occ, 0.5

        # Era-based defaults
        era = self._get_era(birth_year)
        is_hawaiian = 'hawaii' in location.lower()

        if relationship == 'mother':
            return "Homemaker", 0.4

        if is_hawaiian:
            if era < 1900:
                return "Farmer/Fisherman", 0.4
            else:
                return "Laborer", 0.3
        else:
            if era < 1800:
                return "Farmer", 0.4
            elif era < 1900:
                return "Farmer/Laborer", 0.3
            else:
                return None, 0.1

    def _predict_origin(
        self,
        surname: str,
        location: str,
        known_ancestors: List[Dict],
        reasoning: List[str]
    ) -> Tuple[Optional[str], float]:
        """Predict ethnic/geographic origin."""

        location_lower = location.lower()

        # Hawaiian indicators
        if any(term in location_lower for term in ['hawaii', 'honolulu', 'maui', 'oahu']):
            # Check if surname seems Hawaiian
            hawaiian_patterns = ['ka', 'ke', 'la', 'na', 'ma', 'ho', 'hu', 'ah']
            if surname and any(surname.lower().startswith(p) for p in hawaiian_patterns):
                reasoning.append("Hawaiian origin based on surname and location")
                return "Native Hawaiian", 0.7
            else:
                reasoning.append("Hawaii location, mixed heritage likely")
                return "Hawaiian/Mixed", 0.5

        # European indicators
        if any(term in location_lower for term in ['england', 'english', 'britain']):
            return "English", 0.6
        if any(term in location_lower for term in ['ireland', 'irish']):
            return "Irish", 0.6
        if any(term in location_lower for term in ['german', 'prussia']):
            return "German", 0.6
        if any(term in location_lower for term in ['scotland', 'scottish']):
            return "Scottish", 0.6

        # Surname-based guessing (simplified)
        if surname:
            surname_lower = surname.lower()
            if surname_lower.endswith(('son', 'sen')):
                return "Scandinavian", 0.4
            if surname_lower.startswith(("mc", "mac", "o'")):
                return "Irish/Scottish", 0.5
            if surname_lower.endswith(('ski', 'sky')):
                return "Polish/Eastern European", 0.4

        return None, 0.1

    def _get_era(self, year: int) -> int:
        """Get era bucket for a year."""
        eras = [1500, 1600, 1700, 1800, 1850, 1900, 1950, 2000]
        for era in reversed(eras):
            if year >= era:
                return era
        return 1500


def generate_ancestor_predictions(
    individuals: Dict,
    person_id: str,
    generator: AncestorAttributeGenerator = None
) -> Dict[str, PredictedAncestor]:
    """Generate predictions for all missing ancestors of a person."""

    if generator is None:
        generator = AncestorAttributeGenerator()

    person = individuals.get(person_id)
    if not person:
        return {}

    predictions = {}

    # Get known ancestors for context
    known_ancestors = []
    queue = [person_id]
    visited = set()

    while queue and len(visited) < 50:
        curr_id = queue.pop(0)
        if curr_id in visited:
            continue
        visited.add(curr_id)

        curr = individuals.get(curr_id)
        if not curr:
            continue

        if curr_id != person_id:
            known_ancestors.append(curr)

        if curr.get('father'):
            queue.append(curr['father'])
        if curr.get('mother'):
            queue.append(curr['mother'])

    # Get spouse info
    spouse = None
    if person.get('father'):
        father = individuals.get(person['father'])
        if father:
            spouse = individuals.get(person.get('mother')) if person.get('mother') else None

    # Predict missing father
    if not person.get('father'):
        mother = individuals.get(person.get('mother')) if person.get('mother') else None
        pred = generator.predict_ancestor(
            child=person,
            relationship='father',
            spouse=mother,
            known_ancestors=known_ancestors
        )
        predictions['father'] = pred

    # Predict missing mother
    if not person.get('mother'):
        father = individuals.get(person.get('father')) if person.get('father') else None
        pred = generator.predict_ancestor(
            child=person,
            relationship='mother',
            spouse=father,
            known_ancestors=known_ancestors
        )
        predictions['mother'] = pred

    return predictions


# ============================================================================
# Export to JSON for web app
# ============================================================================

def prediction_to_dict(pred: PredictedAncestor) -> Dict:
    """Convert prediction to JSON-serializable dict."""
    return {
        'relationship': pred.relationship,
        'confidence': round(pred.confidence, 2),
        'surname': pred.predicted_surname,
        'surnameConfidence': round(pred.surname_confidence, 2),
        'givenNames': pred.possible_given_names,
        'birthYear': pred.estimated_birth_year,
        'birthYearRange': list(pred.birth_year_range),
        'deathYear': pred.estimated_death_year,
        'deathYearRange': list(pred.death_year_range) if pred.death_year_range else None,
        'lifespan': pred.estimated_lifespan,
        'birthLocation': pred.predicted_birth_location,
        'locationConfidence': round(pred.location_confidence, 2),
        'locationComponents': pred.location_components,
        'occupation': pred.predicted_occupation,
        'occupationConfidence': round(pred.occupation_confidence, 2),
        'origin': pred.ethnic_origin,
        'originConfidence': round(pred.origin_confidence, 2),
        'reasoning': pred.reasoning
    }


if __name__ == '__main__':
    # Test with sample data
    test_child = {
        'id': 'test1',
        'givenName': 'Keoni',
        'surname': 'Hussey',
        'gender': 'M',
        'birthYear': 1920,
        'birthPlace': 'Honolulu, Hawaii',
        'father': None,
        'mother': None
    }

    generator = AncestorAttributeGenerator()

    father_pred = generator.predict_ancestor(
        child=test_child,
        relationship='father',
        known_ancestors=[]
    )

    mother_pred = generator.predict_ancestor(
        child=test_child,
        relationship='mother',
        known_ancestors=[]
    )

    print("=== Predicted Father ===")
    print(json.dumps(prediction_to_dict(father_pred), indent=2))

    print("\n=== Predicted Mother ===")
    print(json.dumps(prediction_to_dict(mother_pred), indent=2))
