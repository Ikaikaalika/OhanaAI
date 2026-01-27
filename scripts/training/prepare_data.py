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
from collections import Counter, defaultdict


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
# Ethnic Origin and Name Pattern Detection
# ============================================================================

SURNAME_PATTERNS = {
    'irish': [r"^O'", r'^Mc', r'^Mac', r'(?:agh|igan|ihan|elly|eary|arty)$'],
    'german': [r'(?:mann|berg|stein|burg|feld|bach|hoff|meyer|müller|schmidt|schneider|fischer|weber)$'],
    'italian': [r'(?:ini|ino|etti|ello|ucci|acci|oli|ari|one|oni|isi|ese|ato|iti|otti)$'],
    'polish': [r'(?:ski|ska|wicz|czyk|iak|czak|owski|ewicz|owicz)$'],
    'scandinavian': [r'(?:son|sen|sson|berg|gren|lund|qvist|strom|dahl)$'],
    'scottish': [r'^Mac', r'^Mc', r'(?:ston|son)$'],
    'jewish': [r'(?:berg|stein|man|witz|feld|baum|blum|gold|silver|rosen|green|klein)$'],
    'portuguese': [r'(?:eira|eiro|ães|es|inho|inha|ões)$'],
    'hawaiian': [r'(?:^Ka|^Ke|^Ku|^La|^Ma|^Na|^Pa|lani|moku|aloha|lei|kai|mana|nui|iki)'],
    'chinese': [r'^(?:Wong|Wang|Chen|Li|Liu|Zhang|Huang|Lin|Wu|Yang|Zhou|Xu|Lee|Cheng|Lau|Ho|Ng)$'],
    'japanese': [r'(?:moto|yama|shima|mura|kawa|saki|hara|uchi|guchi|wara|zaki|zawa|ta|da|no|ki)$'],
    'filipino': [r'(?:Cruz|Santos|Reyes|Garcia|Mendoza|Torres|Flores|Rivera|Ramos|Aquino|Bautista|Villanueva)$'],
}

REGIONAL_NAMES = {
    'irish': {'male': ['Patrick', 'Michael', 'John', 'Sean', 'James', 'William', 'Thomas', 'Daniel'],
              'female': ['Mary', 'Bridget', 'Catherine', 'Margaret', 'Anne', 'Ellen', 'Nora', 'Rose']},
    'german': {'male': ['Johann', 'Friedrich', 'Wilhelm', 'Heinrich', 'Karl', 'Hans', 'Otto', 'Ludwig'],
               'female': ['Maria', 'Anna', 'Margarethe', 'Elisabeth', 'Katharina', 'Frieda', 'Helene', 'Emma']},
    'italian': {'male': ['Giuseppe', 'Giovanni', 'Antonio', 'Francesco', 'Luigi', 'Salvatore', 'Angelo', 'Vincenzo'],
                'female': ['Maria', 'Rosa', 'Angela', 'Giovanna', 'Lucia', 'Concetta', 'Teresa', 'Carmela']},
    'polish': {'male': ['Jan', 'Stanislaw', 'Jozef', 'Wladyslaw', 'Kazimierz', 'Tadeusz', 'Antoni', 'Franciszek'],
               'female': ['Maria', 'Anna', 'Zofia', 'Jadwiga', 'Helena', 'Stefania', 'Bronislawa', 'Wanda']},
    'scandinavian': {'male': ['Erik', 'Lars', 'Anders', 'Nils', 'Olaf', 'Magnus', 'Johan', 'Sven'],
                     'female': ['Anna', 'Maria', 'Ingrid', 'Kristina', 'Karin', 'Margareta', 'Astrid', 'Sigrid']},
    'hawaiian': {'male': ['Kekoa', 'Keoni', 'Kalani', 'Makoa', 'Kaleo', 'Kai', 'Lono', 'Manu'],
                 'female': ['Leilani', 'Mahina', 'Malia', 'Kailani', 'Noelani', 'Keala', 'Nalani', 'Haunani']},
    'chinese': {'male': ['Wing', 'Hing', 'Ming', 'Chung', 'Wai', 'Cheung', 'Yuen', 'Fong'],
                'female': ['Mei', 'Lin', 'Ying', 'Fong', 'Lai', 'Wai', 'Siu', 'Kit']},
    'japanese': {'male': ['Takeshi', 'Hiroshi', 'Masao', 'Yoshio', 'Kenji', 'Taro', 'Saburo', 'Jiro'],
                 'female': ['Yuki', 'Hanako', 'Yoshiko', 'Sachiko', 'Kimiko', 'Masako', 'Noriko', 'Fumiko']},
    'portuguese': {'male': ['Joao', 'Jose', 'Manuel', 'Antonio', 'Francisco', 'Joaquim', 'Carlos', 'Luis'],
                   'female': ['Maria', 'Ana', 'Rosa', 'Isabel', 'Francisca', 'Antonia', 'Joaquina', 'Teresa']},
}

ERA_OCCUPATIONS = {
    (1500, 1700): ['Farmer', 'Blacksmith', 'Cooper', 'Miller', 'Carpenter', 'Weaver', 'Tailor'],
    (1700, 1850): ['Farmer', 'Laborer', 'Merchant', 'Craftsman', 'Sailor', 'Innkeeper', 'Clerk'],
    (1850, 1920): ['Farmer', 'Laborer', 'Factory Worker', 'Miner', 'Railroad Worker', 'Clerk', 'Teacher'],
    (1920, 1970): ['Factory Worker', 'Farmer', 'Clerk', 'Salesman', 'Mechanic', 'Teacher', 'Office Worker'],
    (1970, 2030): ['Office Worker', 'Teacher', 'Engineer', 'Manager', 'Healthcare Worker', 'Service Worker'],
}


def detect_ethnic_origin(surname: Optional[str], birth_place: Optional['PlaceInfo'] = None) -> Tuple[Optional[str], float]:
    """Detect likely ethnic origin from surname patterns and location."""
    if not surname:
        return None, 0.0

    surname_upper = surname.strip()
    matches = []

    for ethnicity, patterns in SURNAME_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, surname_upper, re.IGNORECASE):
                matches.append(ethnicity)
                break

    # Check location for additional hints
    if birth_place and birth_place.components:
        location_str = ' '.join(birth_place.components).lower()
        if any(loc in location_str for loc in ['hawaii', 'honolulu', 'maui', 'oahu', 'kauai']):
            if 'hawaiian' not in matches:
                matches.append('hawaiian')
        elif any(loc in location_str for loc in ['ireland', 'dublin', 'cork', 'galway']):
            if 'irish' not in matches:
                matches.append('irish')
        elif any(loc in location_str for loc in ['germany', 'prussia', 'bavaria', 'saxony']):
            if 'german' not in matches:
                matches.append('german')
        elif any(loc in location_str for loc in ['italy', 'sicily', 'naples', 'rome']):
            if 'italian' not in matches:
                matches.append('italian')
        elif any(loc in location_str for loc in ['poland', 'warsaw', 'krakow']):
            if 'polish' not in matches:
                matches.append('polish')

    if matches:
        return matches[0], min(0.3 + 0.2 * len(matches), 0.9)
    return None, 0.0


def get_era_occupation(birth_year: Optional[int], location: Optional['PlaceInfo'] = None) -> Optional[str]:
    """Get typical occupation for an era and location."""
    if not birth_year:
        return None

    for (start, end), occupations in ERA_OCCUPATIONS.items():
        if start <= birth_year < end:
            # Location-specific adjustments
            if location and location.components:
                loc_str = ' '.join(location.components).lower()
                if 'hawaii' in loc_str:
                    return 'Plantation Worker' if birth_year < 1950 else occupations[0]
                if any(loc in loc_str for loc in ['mining', 'coal', 'pennsylvania', 'west virginia']):
                    return 'Miner'
                if any(loc in loc_str for loc in ['new york', 'chicago', 'boston', 'philadelphia']):
                    return 'Factory Worker' if birth_year < 1950 else 'Office Worker'
            return occupations[0]
    return None


# ============================================================================
# Family Pattern Analysis
# ============================================================================

class FamilyPatternAnalyzer:
    """Analyzes patterns across the family tree for improved predictions."""

    def __init__(self, individuals: Dict[str, 'Individual'], families: Dict[str, 'Family']):
        self.individuals = individuals
        self.families = families
        self.naming_patterns: Dict[str, Counter] = defaultdict(Counter)
        self.occupation_patterns: Counter = Counter()
        self.location_patterns: Counter = Counter()
        self.surname_variants: Dict[str, Set[str]] = defaultdict(set)
        self.sibling_birth_gaps: List[float] = []
        self.parent_child_age_gaps: List[Tuple[str, float]] = []  # (relation_type, gap)
        self._analyze()

    def _analyze(self):
        """Analyze all patterns in the family tree."""
        for ind in self.individuals.values():
            # Track given names by surname
            if ind.given_name and ind.surname:
                self.naming_patterns[ind.surname.lower()][ind.given_name.lower()] += 1

            # Track locations
            if ind.birth_place:
                for comp in ind.birth_place.components:
                    self.location_patterns[comp.lower()] += 1

            # Analyze parent-child age gaps
            if ind.birth_year:
                if ind.father and ind.father in self.individuals:
                    father = self.individuals[ind.father]
                    if father.birth_year:
                        gap = ind.birth_year - father.birth_year
                        if 15 <= gap <= 70:
                            self.parent_child_age_gaps.append(('father', gap))

                if ind.mother and ind.mother in self.individuals:
                    mother = self.individuals[ind.mother]
                    if mother.birth_year:
                        gap = ind.birth_year - mother.birth_year
                        if 12 <= gap <= 55:
                            self.parent_child_age_gaps.append(('mother', gap))

            # Analyze sibling birth gaps
            if ind.siblings:
                sibling_years = []
                if ind.birth_year:
                    sibling_years.append(ind.birth_year)
                for sib_id in ind.siblings:
                    sib = self.individuals.get(sib_id)
                    if sib and sib.birth_year:
                        sibling_years.append(sib.birth_year)

                if len(sibling_years) >= 2:
                    sibling_years.sort()
                    for i in range(1, len(sibling_years)):
                        gap = sibling_years[i] - sibling_years[i-1]
                        if 0 < gap <= 15:
                            self.sibling_birth_gaps.append(gap)

    def get_avg_parent_age_gap(self, relation: str) -> float:
        """Get average age gap for a parent type."""
        gaps = [g for t, g in self.parent_child_age_gaps if t == relation]
        if gaps:
            return sum(gaps) / len(gaps)
        return 28.0 if relation == 'father' else 25.0  # Default

    def get_avg_sibling_gap(self) -> float:
        """Get average sibling birth gap."""
        if self.sibling_birth_gaps:
            return sum(self.sibling_birth_gaps) / len(self.sibling_birth_gaps)
        return 2.5  # Default

    def get_common_names(self, surname: str, gender: str, n: int = 5) -> List[str]:
        """Get most common given names for a surname."""
        names = self.naming_patterns.get(surname.lower(), Counter())
        # Filter by gender (rough heuristic based on name endings)
        return [name for name, _ in names.most_common(n * 2)][:n]

    def get_common_locations(self, n: int = 5) -> List[str]:
        """Get most common locations in the tree."""
        return [loc for loc, _ in self.location_patterns.most_common(n)]


# ============================================================================
# Feature Extraction
# ============================================================================

# Increased feature dimension to include ethnic and pattern features
FEATURE_DIM = 224  # Was 176, added 48 for new features


def extract_features(ind: Individual, all_individuals: Dict[str, Individual],
                     locations: Set[str], year_range: Tuple[int, int],
                     pattern_analyzer: Optional[FamilyPatternAnalyzer] = None) -> List[float]:
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

    # === Ethnic origin features (16) ===
    ethnic_origin, ethnic_confidence = detect_ethnic_origin(ind.surname, ind.birth_place)
    ethnic_features = [0] * 16

    ethnic_to_idx = {
        'irish': 0, 'german': 1, 'italian': 2, 'polish': 3,
        'scandinavian': 4, 'scottish': 5, 'jewish': 6, 'portuguese': 7,
        'hawaiian': 8, 'chinese': 9, 'japanese': 10, 'filipino': 11
    }
    if ethnic_origin and ethnic_origin in ethnic_to_idx:
        ethnic_features[ethnic_to_idx[ethnic_origin]] = 1
    ethnic_features[12] = ethnic_confidence
    ethnic_features[13] = 1 if ethnic_origin else 0  # has detected origin
    # padding
    features.extend(ethnic_features)

    # === Family pattern features (16) ===
    pattern_features = [0] * 16

    if pattern_analyzer:
        # Average parent age gaps (normalized)
        avg_father_gap = pattern_analyzer.get_avg_parent_age_gap('father')
        avg_mother_gap = pattern_analyzer.get_avg_parent_age_gap('mother')
        pattern_features[0] = avg_father_gap / 50.0
        pattern_features[1] = avg_mother_gap / 50.0

        # Average sibling gap (normalized)
        avg_sib_gap = pattern_analyzer.get_avg_sibling_gap()
        pattern_features[2] = avg_sib_gap / 10.0

        # Name commonality (is this person's name common in the family?)
        if ind.given_name and ind.surname:
            common_names = pattern_analyzer.get_common_names(ind.surname, ind.gender or 'M')
            pattern_features[3] = 1 if ind.given_name.lower() in [n.lower() for n in common_names] else 0

        # Location commonality
        if ind.birth_place and ind.birth_place.components:
            common_locs = pattern_analyzer.get_common_locations()
            for comp in ind.birth_place.components:
                if comp.lower() in [l.lower() for l in common_locs]:
                    pattern_features[4] = 1
                    break

        # Has known naming tradition
        pattern_features[5] = 1 if len(pattern_analyzer.naming_patterns) > 3 else 0

        # Number of unique surnames in family (diversity indicator)
        pattern_features[6] = min(len(pattern_analyzer.naming_patterns) / 20, 1)

        # Number of analyzed gaps
        pattern_features[7] = min(len(pattern_analyzer.parent_child_age_gaps) / 50, 1)

    features.extend(pattern_features)

    # === Era-specific occupation features (16) ===
    occupation_features = [0] * 16

    era_occupation = get_era_occupation(ind.birth_year, ind.birth_place)
    occupation_to_idx = {
        'Farmer': 0, 'Laborer': 1, 'Factory Worker': 2, 'Miner': 3,
        'Plantation Worker': 4, 'Office Worker': 5, 'Teacher': 6,
        'Merchant': 7, 'Craftsman': 8, 'Clerk': 9
    }
    if era_occupation and era_occupation in occupation_to_idx:
        occupation_features[occupation_to_idx[era_occupation]] = 1
    occupation_features[10] = 1 if era_occupation else 0  # has predicted occupation
    # Additional era indicators for occupation context
    if ind.birth_year:
        occupation_features[11] = 1 if ind.birth_year < 1850 else 0  # pre-industrial
        occupation_features[12] = 1 if 1850 <= ind.birth_year < 1920 else 0  # industrial
        occupation_features[13] = 1 if 1920 <= ind.birth_year < 1970 else 0  # modern
        occupation_features[14] = 1 if ind.birth_year >= 1970 else 0  # contemporary

    features.extend(occupation_features)

    # Padding to FEATURE_DIM (224)
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

    # Analyze family patterns
    pattern_analyzer = FamilyPatternAnalyzer(individuals, families)

    # Create node mapping
    node_ids = list(individuals.keys())
    node_id_to_idx = {nid: i for i, nid in enumerate(node_ids)}

    # Extract features
    node_features = []
    for nid in node_ids:
        ind = individuals[nid]
        features = extract_features(ind, individuals, all_locations, year_range, pattern_analyzer)
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

    # Build attribute labels for training attribute generation
    # These labels are derived from known relatives to train the model to predict attributes
    attribute_labels = {
        'fatherBirthYear': [],  # Normalized birth year of father
        'motherBirthYear': [],  # Normalized birth year of mother
        'fatherEthnicOrigin': [],  # One-hot encoded ethnic origin (12 classes)
        'motherEthnicOrigin': [],
        'parentLocation': [],  # Encoded location features
    }

    for nid in node_ids:
        ind = individuals[nid]

        # Father birth year (if known)
        father_birth_norm = 0.0
        if ind.father and ind.father in individuals:
            father = individuals[ind.father]
            if father.birth_year:
                father_birth_norm = (father.birth_year - year_range[0]) / max(year_range[1] - year_range[0], 1)
        attribute_labels['fatherBirthYear'].append(father_birth_norm)

        # Mother birth year (if known)
        mother_birth_norm = 0.0
        if ind.mother and ind.mother in individuals:
            mother = individuals[ind.mother]
            if mother.birth_year:
                mother_birth_norm = (mother.birth_year - year_range[0]) / max(year_range[1] - year_range[0], 1)
        attribute_labels['motherBirthYear'].append(mother_birth_norm)

        # Father ethnic origin (one-hot, 12 classes)
        father_ethnic = [0] * 12
        if ind.father and ind.father in individuals:
            father = individuals[ind.father]
            ethnic, _ = detect_ethnic_origin(father.surname, father.birth_place)
            ethnic_to_idx = {
                'irish': 0, 'german': 1, 'italian': 2, 'polish': 3,
                'scandinavian': 4, 'scottish': 5, 'jewish': 6, 'portuguese': 7,
                'hawaiian': 8, 'chinese': 9, 'japanese': 10, 'filipino': 11
            }
            if ethnic and ethnic in ethnic_to_idx:
                father_ethnic[ethnic_to_idx[ethnic]] = 1
        attribute_labels['fatherEthnicOrigin'].append(father_ethnic)

        # Mother ethnic origin
        mother_ethnic = [0] * 12
        if ind.mother and ind.mother in individuals:
            mother = individuals[ind.mother]
            ethnic, _ = detect_ethnic_origin(mother.surname, mother.birth_place)
            ethnic_to_idx = {
                'irish': 0, 'german': 1, 'italian': 2, 'polish': 3,
                'scandinavian': 4, 'scottish': 5, 'jewish': 6, 'portuguese': 7,
                'hawaiian': 8, 'chinese': 9, 'japanese': 10, 'filipino': 11
            }
            if ethnic and ethnic in ethnic_to_idx:
                mother_ethnic[ethnic_to_idx[ethnic]] = 1
        attribute_labels['motherEthnicOrigin'].append(mother_ethnic)

        # Parent location (use hash-based encoding, 8 dims)
        parent_loc = [0] * 8
        if ind.father and ind.father in individuals:
            father = individuals[ind.father]
            if father.birth_place and father.birth_place.components:
                for comp in father.birth_place.components[:2]:
                    hash_val = hash(comp.lower()) % 8
                    parent_loc[hash_val] = 1
        if ind.mother and ind.mother in individuals:
            mother = individuals[ind.mother]
            if mother.birth_place and mother.birth_place.components:
                for comp in mother.birth_place.components[:2]:
                    hash_val = hash(comp.lower()) % 8
                    parent_loc[hash_val] = 1
        attribute_labels['parentLocation'].append(parent_loc)

    # Add family pattern statistics
    pattern_stats = {
        'avgFatherAgeGap': pattern_analyzer.get_avg_parent_age_gap('father'),
        'avgMotherAgeGap': pattern_analyzer.get_avg_parent_age_gap('mother'),
        'avgSiblingGap': pattern_analyzer.get_avg_sibling_gap(),
        'numNamingPatterns': len(pattern_analyzer.naming_patterns),
        'numLocationPatterns': len(pattern_analyzer.location_patterns),
        'commonLocations': pattern_analyzer.get_common_locations(10),
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
        'attributeLabels': attribute_labels,
        'patternStats': pattern_stats,
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
