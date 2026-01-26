#!/usr/bin/env python3
"""
Prepare GEDCOM file for ML training.

Parses a GEDCOM file and converts it to the training format.

Usage: python prepare_data.py path/to/file.ged
"""

import argparse
import json
import re
import sys
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import numpy as np


# ============================================================================
# Data Classes
# ============================================================================

@dataclass
class DateInfo:
    raw: str
    year: Optional[int] = None
    month: Optional[int] = None
    day: Optional[int] = None
    circa: bool = False


@dataclass
class PlaceInfo:
    raw: str
    components: List[str] = field(default_factory=list)


@dataclass
class Individual:
    id: str
    given_name: Optional[str] = None
    surname: Optional[str] = None
    gender: Optional[str] = None
    birth_date: Optional[DateInfo] = None
    birth_place: Optional[PlaceInfo] = None
    death_date: Optional[DateInfo] = None
    death_place: Optional[PlaceInfo] = None
    father: Optional[str] = None
    mother: Optional[str] = None
    spouses: List[str] = field(default_factory=list)
    children: List[str] = field(default_factory=list)
    siblings: List[str] = field(default_factory=list)
    residences: List[PlaceInfo] = field(default_factory=list)

    @property
    def birth_year(self) -> Optional[int]:
        return self.birth_date.year if self.birth_date else None

    @property
    def death_year(self) -> Optional[int]:
        return self.death_date.year if self.death_date else None


@dataclass
class Family:
    id: str
    husband: Optional[str] = None
    wife: Optional[str] = None
    children: List[str] = field(default_factory=list)
    marriage_date: Optional[DateInfo] = None
    marriage_place: Optional[PlaceInfo] = None


# ============================================================================
# GEDCOM Parser
# ============================================================================

def parse_date(raw: str) -> DateInfo:
    """Parse a GEDCOM date string."""
    result = DateInfo(raw=raw)

    # Check for approximate dates
    if re.match(r'^(ABT|ABOUT|EST|CAL|CIRCA)\s+', raw, re.I):
        result.circa = True
        raw = re.sub(r'^(ABT|ABOUT|EST|CAL|CIRCA)\s+', '', raw, flags=re.I)

    # Handle BEF/AFT
    raw = re.sub(r'^(BEF|AFT|BEFORE|AFTER)\s+', '', raw, flags=re.I)

    # Handle date ranges - use start date
    bet_match = re.match(r'^BET\s+(.+)\s+AND\s+', raw, re.I)
    if bet_match:
        raw = bet_match.group(1)

    # Extract year
    year_match = re.search(r'\b(\d{4})\b', raw)
    if year_match:
        result.year = int(year_match.group(1))

    # Extract month
    months = {
        'JAN': 1, 'FEB': 2, 'MAR': 3, 'APR': 4, 'MAY': 5, 'JUN': 6,
        'JUL': 7, 'AUG': 8, 'SEP': 9, 'OCT': 10, 'NOV': 11, 'DEC': 12
    }
    month_match = re.search(r'\b(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)\b', raw, re.I)
    if month_match:
        result.month = months[month_match.group(1).upper()]

    # Extract day
    day_match = re.search(r'\b(\d{1,2})\s+(JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)', raw, re.I)
    if day_match:
        result.day = int(day_match.group(1))

    return result


def parse_place(raw: str) -> PlaceInfo:
    """Parse a GEDCOM place string."""
    components = [s.strip() for s in raw.split(',') if s.strip()]
    return PlaceInfo(raw=raw, components=components)


def parse_name(value: str) -> Tuple[Optional[str], Optional[str]]:
    """Parse a GEDCOM name, returning (given_name, surname)."""
    # Surname is between slashes
    surname_match = re.search(r'/([^/]+)/', value)
    surname = surname_match.group(1).strip() if surname_match else None

    # Given name is before the first slash
    given_match = re.match(r'^([^/]+)/', value)
    given = given_match.group(1).strip() if given_match else None

    return given, surname


def parse_gedcom(file_path: Path) -> Tuple[Dict[str, Individual], Dict[str, Family]]:
    """Parse a GEDCOM file and return individuals and families."""

    individuals: Dict[str, Individual] = {}
    families: Dict[str, Family] = {}

    with open(file_path, 'r', encoding='utf-8', errors='replace') as f:
        lines = f.readlines()

    current_record = None
    current_type = None
    current_event = None

    for line in lines:
        line = line.strip()
        if not line:
            continue

        # Parse line: LEVEL [POINTER] TAG [VALUE]
        match = re.match(r'^(\d+)\s+(?:(@[^@]+@)\s+)?(\S+)(?:\s+(.*))?$', line)
        if not match:
            continue

        level = int(match.group(1))
        pointer = match.group(2)
        tag = match.group(3)
        value = match.group(4) or ''

        # Level 0: New record
        if level == 0:
            current_event = None

            if pointer:
                if tag == 'INDI' or value == 'INDI':
                    current_record = Individual(id=pointer)
                    current_type = 'INDI'
                    individuals[pointer] = current_record
                elif tag == 'FAM' or value == 'FAM':
                    current_record = Family(id=pointer)
                    current_type = 'FAM'
                    families[pointer] = current_record
                else:
                    current_record = None
                    current_type = None
            else:
                current_record = None
                current_type = None
            continue

        if not current_record:
            continue

        # Level 1: Main attributes
        if level == 1:
            current_event = None

            if current_type == 'INDI':
                if tag == 'NAME':
                    given, surname = parse_name(value)
                    current_record.given_name = given
                    current_record.surname = surname
                elif tag == 'SEX':
                    current_record.gender = 'M' if value == 'M' else 'F' if value == 'F' else None
                elif tag == 'BIRT':
                    current_event = 'BIRT'
                elif tag == 'DEAT':
                    current_event = 'DEAT'
                elif tag == 'RESI':
                    current_event = 'RESI'

            elif current_type == 'FAM':
                if tag == 'HUSB':
                    current_record.husband = value
                elif tag == 'WIFE':
                    current_record.wife = value
                elif tag == 'CHIL':
                    current_record.children.append(value)
                elif tag == 'MARR':
                    current_event = 'MARR'

        # Level 2: Event details
        elif level == 2 and current_event:
            if tag == 'DATE':
                date = parse_date(value)
                if current_event == 'BIRT':
                    current_record.birth_date = date
                elif current_event == 'DEAT':
                    current_record.death_date = date
                elif current_event == 'MARR' and current_type == 'FAM':
                    current_record.marriage_date = date
            elif tag == 'PLAC':
                place = parse_place(value)
                if current_event == 'BIRT':
                    current_record.birth_place = place
                elif current_event == 'DEAT':
                    current_record.death_place = place
                elif current_event == 'RESI':
                    current_record.residences.append(place)
                elif current_event == 'MARR' and current_type == 'FAM':
                    current_record.marriage_place = place

    # Build relationships
    for fam in families.values():
        for child_id in fam.children:
            child = individuals.get(child_id)
            if child:
                if fam.husband:
                    child.father = fam.husband
                if fam.wife:
                    child.mother = fam.wife

        # Spouses
        if fam.husband and fam.wife:
            husband = individuals.get(fam.husband)
            wife = individuals.get(fam.wife)
            if husband and fam.wife not in husband.spouses:
                husband.spouses.append(fam.wife)
            if wife and fam.husband not in wife.spouses:
                wife.spouses.append(fam.husband)

        # Children
        if fam.husband:
            parent = individuals.get(fam.husband)
            if parent:
                for child_id in fam.children:
                    if child_id not in parent.children:
                        parent.children.append(child_id)
        if fam.wife:
            parent = individuals.get(fam.wife)
            if parent:
                for child_id in fam.children:
                    if child_id not in parent.children:
                        parent.children.append(child_id)

        # Siblings
        for i, child1_id in enumerate(fam.children):
            for child2_id in fam.children[i+1:]:
                child1 = individuals.get(child1_id)
                child2 = individuals.get(child2_id)
                if child1 and child2_id not in child1.siblings:
                    child1.siblings.append(child2_id)
                if child2 and child1_id not in child2.siblings:
                    child2.siblings.append(child1_id)

    return individuals, families


# ============================================================================
# Feature Extraction
# ============================================================================

FEATURE_DIM = 176


def extract_features(ind: Individual, all_individuals: Dict[str, Individual],
                     locations: Set[str], year_range: Tuple[int, int]) -> List[float]:
    """Extract feature vector for an individual."""
    features = []

    min_year, max_year = year_range
    year_span = max(max_year - min_year, 1)

    # === Demographic features (8) ===
    features.append(1 if ind.gender is None else 0)  # unknown gender
    features.append(1 if ind.gender == 'M' else 0)   # male
    features.append(1 if ind.gender == 'F' else 0)   # female
    features.append(1 if ind.given_name else 0)      # has given name
    features.append(1 if ind.surname else 0)         # has surname
    features.append(min(len(ind.given_name or '') / 20, 1))  # given name length
    features.append(min(len(ind.surname or '') / 20, 1))     # surname length
    features.append(1 if len(ind.residences) > 0 else 0)     # has residence

    # === Temporal features (16) ===
    birth_year_norm = (ind.birth_year - min_year) / year_span if ind.birth_year else 0
    death_year_norm = (ind.death_year - min_year) / year_span if ind.death_year else 0

    features.append(birth_year_norm)
    features.append(death_year_norm)
    features.append(1 if ind.birth_year else 0)
    features.append(1 if ind.death_year else 0)
    features.append(1 if ind.birth_date and ind.birth_date.month else 0)
    features.append(1 if ind.birth_date and ind.birth_date.day else 0)
    features.append(1 if ind.death_date and ind.death_date.month else 0)
    features.append(1 if ind.death_date and ind.death_date.day else 0)

    lifespan = (ind.death_year - ind.birth_year) / 100 if ind.birth_year and ind.death_year else 0
    features.append(min(lifespan, 1.5))

    features.append(1 if ind.birth_date and ind.birth_date.circa else 0)
    features.append(1 if ind.death_date and ind.death_date.circa else 0)

    # Era indicators
    century = ind.birth_year // 100 if ind.birth_year else 0
    features.append(1 if century == 17 else 0)
    features.append(1 if century == 18 else 0)
    features.append(1 if century == 19 else 0)
    features.append(1 if century >= 20 else 0)
    features.append(0)  # padding

    # === Geographic features (32) ===
    # Simple location encoding
    loc_features = [0] * 32
    if ind.birth_place:
        for i, comp in enumerate(ind.birth_place.components[:4]):
            hash_val = hash(comp.lower()) % 8
            loc_features[i * 8 + hash_val] = 1
    features.extend(loc_features)

    # === Name embedding (64) ===
    name_features = [0] * 64
    given = (ind.given_name or '').lower()
    surname = (ind.surname or '').lower()

    # Character frequencies
    for c in given:
        idx = ord(c) - ord('a')
        if 0 <= idx < 26:
            name_features[idx] += 1 / max(len(given), 1)
    for c in surname:
        idx = ord(c) - ord('a')
        if 0 <= idx < 26:
            name_features[32 + idx] += 1 / max(len(surname), 1)

    # Vowel ratios
    vowels = set('aeiou')
    name_features[58] = sum(1 for c in given if c in vowels) / max(len(given), 1)
    name_features[59] = sum(1 for c in surname if c in vowels) / max(len(surname), 1)

    features.extend(name_features)

    # === Graph structural features (16) ===
    total_connections = (1 if ind.father else 0) + (1 if ind.mother else 0) + \
                        len(ind.spouses) + len(ind.children) + len(ind.siblings)

    features.append(min(total_connections / 20, 1))  # degree
    features.append(min(len(ind.children) / 10, 1))  # in-degree
    features.append((1 if ind.father else 0) * 0.5 + (1 if ind.mother else 0) * 0.5)
    features.append(min(len(ind.siblings) / 10, 1))
    features.append(min(len(ind.spouses) / 5, 1))
    features.append(1 if not ind.father and not ind.mother else 0)  # is root
    features.append(1 if len(ind.children) == 0 else 0)  # is leaf
    features.append(1 if ind.father and ind.mother else 0)  # has both parents

    # Generation estimate (simplified)
    generation = 0
    current = ind
    visited = set()
    while current and current.id not in visited:
        visited.add(current.id)
        if current.father:
            current = all_individuals.get(current.father)
            generation += 1
        elif current.mother:
            current = all_individuals.get(current.mother)
            generation += 1
        else:
            break
    features.append(min(generation / 10, 1))

    # Padding
    while len(features) < 56 + 16:
        features.append(0)

    # === Relationship features (24) ===
    features.append(0 if ind.father else 1)  # missing father
    features.append(0 if ind.mother else 1)  # missing mother
    features.append(1 if not ind.father and not ind.mother else 0)  # missing both
    features.append(1 if ind.father and ind.mother else 0)  # has both
    features.append(1 if len(ind.spouses) == 0 else 0)
    features.append(1 if len(ind.spouses) == 1 else 0)
    features.append(1 if len(ind.spouses) > 1 else 0)
    features.append(1 if len(ind.children) == 0 else 0)
    features.append(1 if 0 < len(ind.children) <= 3 else 0)
    features.append(1 if len(ind.children) > 3 else 0)
    features.append(1 if len(ind.siblings) == 0 else 0)
    features.append(1 if len(ind.siblings) > 0 else 0)

    # Completeness
    completeness = (1 if ind.father else 0) * 0.25 + \
                   (1 if ind.mother else 0) * 0.25 + \
                   (1 if len(ind.spouses) > 0 else 0) * 0.25 + \
                   (1 if len(ind.children) > 0 or len(ind.siblings) > 0 else 0) * 0.25
    features.append(completeness)

    features.append(min(len(ind.spouses) / 5, 1))
    features.append(min(len(ind.children) / 15, 1))
    features.append(min(len(ind.siblings) / 15, 1))

    # Padding
    while len(features) < 56 + 16 + 24:
        features.append(0)

    # === Attribute mask (16) ===
    features.append(1 if ind.given_name else 0)
    features.append(1 if ind.surname else 0)
    features.append(1 if ind.gender else 0)
    features.append(1 if ind.birth_date else 0)
    features.append(1 if ind.birth_place else 0)
    features.append(1 if ind.death_date else 0)
    features.append(1 if ind.death_place else 0)
    features.append(1 if len(ind.residences) > 0 else 0)

    # Padding to 176
    while len(features) < FEATURE_DIM:
        features.append(0)

    return features[:FEATURE_DIM]


def prepare_training_data(individuals: Dict[str, Individual],
                          families: Dict[str, Family]) -> Dict:
    """Convert parsed GEDCOM to training format."""

    # Collect statistics
    all_years = []
    all_locations: Set[str] = set()

    for ind in individuals.values():
        if ind.birth_year:
            all_years.append(ind.birth_year)
        if ind.death_year:
            all_years.append(ind.death_year)
        if ind.birth_place:
            all_locations.add(ind.birth_place.raw)
        if ind.death_place:
            all_locations.add(ind.death_place.raw)

    year_range = (min(all_years) if all_years else 1700,
                  max(all_years) if all_years else 2024)

    # Create node mapping
    node_ids = list(individuals.keys())
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Extract features
    node_features = []
    for nid in node_ids:
        ind = individuals[nid]
        features = extract_features(ind, individuals, all_locations, year_range)
        node_features.append(features)

    # Build edges
    edge_index = [[], []]
    edge_features = []
    edge_types = []

    type_to_idx = {'parent': 0, 'child': 1, 'spouse': 2, 'sibling': 3}

    for nid, ind in individuals.items():
        src_idx = node_id_to_idx[nid]

        # Parent edges
        if ind.father and ind.father in node_id_to_idx:
            tgt_idx = node_id_to_idx[ind.father]
            edge_index[0].append(src_idx)
            edge_index[1].append(tgt_idx)
            edge_features.append([0, 1, 0, 0, 1.0, 0, 0, 0])  # child -> parent
            edge_types.append(1)

        if ind.mother and ind.mother in node_id_to_idx:
            tgt_idx = node_id_to_idx[ind.mother]
            edge_index[0].append(src_idx)
            edge_index[1].append(tgt_idx)
            edge_features.append([0, 1, 0, 0, 1.0, 0, 0, 0])
            edge_types.append(1)

        # Spouse edges
        for spouse_id in ind.spouses:
            if spouse_id in node_id_to_idx:
                tgt_idx = node_id_to_idx[spouse_id]
                edge_index[0].append(src_idx)
                edge_index[1].append(tgt_idx)
                edge_features.append([0, 0, 1, 0, 1.0, 0, 0, 0])
                edge_types.append(2)

        # Sibling edges
        for sib_id in ind.siblings:
            if sib_id in node_id_to_idx:
                tgt_idx = node_id_to_idx[sib_id]
                edge_index[0].append(src_idx)
                edge_index[1].append(tgt_idx)
                edge_features.append([0, 0, 0, 1, 0.8, 0, 0, 0])
                edge_types.append(3)

    # Build labels
    labels = {
        'missingFather': [0 if individuals[nid].father else 1 for nid in node_ids],
        'missingMother': [0 if individuals[nid].mother else 1 for nid in node_ids],
        'missingSpouse': [0 if len(individuals[nid].spouses) > 0 else 1 for nid in node_ids],
        'missingChildren': [0 if len(individuals[nid].children) > 0 else 1 for nid in node_ids],
        'missingSiblings': [0 if len(individuals[nid].siblings) > 0 else 1 for nid in node_ids]
    }

    # Global features
    global_features = [
        min(len(node_ids) / 10000, 1),
        min(len(edge_index[0]) / 50000, 1),
        0.5,  # placeholder
        0.1,  # density placeholder
        0.1,  # components placeholder
        (year_range[1] - year_range[0]) / 500,
        0.5   # generations placeholder
    ]

    return {
        'nodeFeatures': node_features,
        'edgeIndex': edge_index,
        'edgeFeatures': edge_features,
        'edgeTypes': edge_types,
        'labels': labels,
        'nodeIds': node_ids,
        'globalFeatures': global_features
    }


# ============================================================================
# Main
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description='Prepare GEDCOM file for training')
    parser.add_argument('gedcom_file', help='Path to GEDCOM file')
    parser.add_argument('--output-dir', default='training_data',
                        help='Output directory')
    args = parser.parse_args()

    gedcom_path = Path(args.gedcom_file)
    if not gedcom_path.exists():
        print(f"Error: File not found: {gedcom_path}")
        sys.exit(1)

    print(f"Processing: {gedcom_path}")

    # Parse GEDCOM
    individuals, families = parse_gedcom(gedcom_path)

    print(f"  Individuals: {len(individuals)}")
    print(f"  Families: {len(families)}")

    # Prepare training data
    print("\nExtracting features...")
    training_data = prepare_training_data(individuals, families)

    print(f"  Nodes: {len(training_data['nodeFeatures'])}")
    print(f"  Edges: {len(training_data['edgeIndex'][0])}")
    print(f"  Feature dimension: {FEATURE_DIM}")

    # Count missing
    missing_father = sum(training_data['labels']['missingFather'])
    missing_mother = sum(training_data['labels']['missingMother'])
    print(f"  Missing fathers: {missing_father}")
    print(f"  Missing mothers: {missing_mother}")

    # Save
    output_dir = Path(args.output_dir)
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
    print("\nTo train the model, run:")
    print(f"  python train.py --data-dir {args.output_dir}")


if __name__ == '__main__':
    main()
