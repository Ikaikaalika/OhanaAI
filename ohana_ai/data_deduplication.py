"""
Data deduplication system for handling duplicate individuals across GEDCOM files.
Uses similarity scoring and intelligent merging with user confirmation.
"""

import difflib
import logging
import re
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Set, Tuple

from .api.gedcom_parser import Family, Individual

logger = logging.getLogger(__name__)


@dataclass
class DuplicateMatch:
    """Represents a potential duplicate match between two individuals."""

    individual1_id: str
    individual2_id: str
    individual1_source: str  # source file
    individual2_source: str  # source file
    similarity_score: float
    name_similarity: float
    date_similarity: float
    location_similarity: float
    reasons: List[str] = field(default_factory=list)


@dataclass
class MergeAction:
    """Represents a merge action to be performed."""

    primary_id: str
    duplicate_id: str
    merged_individual: Individual
    family_updates: Dict[str, Family]  # families that need updating


class DeduplicationEngine:
    """Engine for detecting and merging duplicate individuals."""

    def __init__(self, name_threshold: float = 0.8, date_tolerance: int = 5):
        self.name_threshold = name_threshold
        self.date_tolerance = date_tolerance

        # Tracking
        self.source_files: Dict[str, str] = {}  # individual_id -> source_file
        self.all_individuals: Dict[str, Individual] = {}
        self.all_families: Dict[str, Family] = {}

    def add_gedcom_data(
        self,
        individuals: Dict[str, Individual],
        families: Dict[str, Family],
        source_file: str,
    ):
        """Add data from a GEDCOM file for deduplication."""
        logger.info(f"Adding {len(individuals)} individuals from {source_file}")

        # Track source files
        for ind_id in individuals:
            self.source_files[ind_id] = source_file

        # Add to collections
        self.all_individuals.update(individuals)
        self.all_families.update(families)

    def detect_duplicates(self) -> List[DuplicateMatch]:
        """Detect potential duplicate individuals across all loaded data."""
        logger.info("Detecting duplicate individuals...")

        duplicates = []
        individual_ids = list(self.all_individuals.keys())

        # Compare all pairs of individuals
        for i in range(len(individual_ids)):
            for j in range(i + 1, len(individual_ids)):
                id1, id2 = individual_ids[i], individual_ids[j]

                # Skip if from same source file (should be handled by GEDCOM parser)
                if self.source_files[id1] == self.source_files[id2]:
                    continue

                individual1 = self.all_individuals[id1]
                individual2 = self.all_individuals[id2]

                match = self._calculate_similarity(individual1, individual2, id1, id2)

                if (
                    match and match.similarity_score >= 0.6
                ):  # Threshold for potential match
                    duplicates.append(match)

        # Sort by similarity score (highest first)
        duplicates.sort(key=lambda x: x.similarity_score, reverse=True)

        logger.info(f"Found {len(duplicates)} potential duplicate pairs")
        return duplicates

    def _calculate_similarity(
        self, ind1: Individual, ind2: Individual, id1: str, id2: str
    ) -> Optional[DuplicateMatch]:
        """Calculate similarity between two individuals."""
        reasons = []

        # Name similarity
        name_sim = self._calculate_name_similarity(ind1, ind2)
        if name_sim < 0.3:  # Too different
            return None

        # Date similarity
        date_sim = self._calculate_date_similarity(ind1, ind2)

        # Location similarity
        location_sim = self._calculate_location_similarity(ind1, ind2)

        # Overall similarity (weighted average)
        similarity = name_sim * 0.5 + date_sim * 0.3 + location_sim * 0.2

        # Add reasons for match
        if name_sim > 0.8:
            reasons.append(f"High name similarity ({name_sim:.2f})")
        if date_sim > 0.8:
            reasons.append(f"Matching dates ({date_sim:.2f})")
        if location_sim > 0.8:
            reasons.append(f"Matching locations ({location_sim:.2f})")

        # Additional exact matches
        if ind1.full_name.lower() == ind2.full_name.lower() and ind1.full_name:
            reasons.append("Exact name match")
            similarity += 0.1

        if (
            ind1.birth_year
            and ind2.birth_year
            and abs(ind1.birth_year - ind2.birth_year) <= 1
        ):
            reasons.append("Birth years within 1 year")
            similarity += 0.1

        return DuplicateMatch(
            individual1_id=id1,
            individual2_id=id2,
            individual1_source=self.source_files[id1],
            individual2_source=self.source_files[id2],
            similarity_score=min(1.0, similarity),
            name_similarity=name_sim,
            date_similarity=date_sim,
            location_similarity=location_sim,
            reasons=reasons,
        )

    def _calculate_name_similarity(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate name similarity between two individuals."""
        name1 = self._normalize_name(ind1.full_name)
        name2 = self._normalize_name(ind2.full_name)

        if not name1 or not name2:
            return 0.0

        # Use difflib for sequence similarity
        similarity = difflib.SequenceMatcher(None, name1, name2).ratio()

        # Boost score for matching surname
        if ind1.surname and ind2.surname:
            surname1 = self._normalize_name(ind1.surname)
            surname2 = self._normalize_name(ind2.surname)
            if surname1 == surname2:
                similarity += 0.2

        # Boost score for matching given names
        if ind1.given_names and ind2.given_names:
            given1 = self._normalize_name(ind1.given_names)
            given2 = self._normalize_name(ind2.given_names)
            given_sim = difflib.SequenceMatcher(None, given1, given2).ratio()
            similarity += given_sim * 0.1

        return min(1.0, similarity)

    def _normalize_name(self, name: str) -> str:
        """Normalize name for comparison."""
        if not name:
            return ""

        # Convert to lowercase, remove extra spaces and punctuation
        normalized = re.sub(r"[^\w\s]", "", name.lower())
        normalized = " ".join(normalized.split())
        return normalized

    def _calculate_date_similarity(self, ind1: Individual, ind2: Individual) -> float:
        """Calculate date similarity between two individuals."""
        similarities = []

        # Birth date similarity
        if ind1.birth_year and ind2.birth_year:
            year_diff = abs(ind1.birth_year - ind2.birth_year)
            if year_diff <= self.date_tolerance:
                birth_sim = 1.0 - (year_diff / self.date_tolerance)
                similarities.append(birth_sim)

        # Death date similarity
        if ind1.death_year and ind2.death_year:
            year_diff = abs(ind1.death_year - ind2.death_year)
            if year_diff <= self.date_tolerance:
                death_sim = 1.0 - (year_diff / self.date_tolerance)
                similarities.append(death_sim)

        if not similarities:
            # If no dates available, don't penalize
            return 0.5

        return sum(similarities) / len(similarities)

    def _calculate_location_similarity(
        self, ind1: Individual, ind2: Individual
    ) -> float:
        """Calculate location similarity between two individuals."""
        similarities = []

        # Birth place similarity
        if ind1.birth_place and ind2.birth_place:
            place1 = self._normalize_place(ind1.birth_place)
            place2 = self._normalize_place(ind2.birth_place)
            if place1 == place2:
                similarities.append(1.0)
            else:
                # Use string similarity for partial matches
                sim = difflib.SequenceMatcher(None, place1, place2).ratio()
                if sim > 0.6:  # Only count reasonable matches
                    similarities.append(sim)

        # Death place similarity
        if ind1.death_place and ind2.death_place:
            place1 = self._normalize_place(ind1.death_place)
            place2 = self._normalize_place(ind2.death_place)
            if place1 == place2:
                similarities.append(1.0)
            else:
                sim = difflib.SequenceMatcher(None, place1, place2).ratio()
                if sim > 0.6:
                    similarities.append(sim)

        if not similarities:
            return 0.5  # Neutral score if no location data

        return sum(similarities) / len(similarities)

    def _normalize_place(self, place: str) -> str:
        """Normalize place name for comparison."""
        if not place:
            return ""

        # Convert to lowercase, remove common prefixes/suffixes
        normalized = place.lower().strip()

        # Remove common location prefixes
        prefixes = ["of ", "in ", "at ", "near "]
        for prefix in prefixes:
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix) :]

        # Remove punctuation and normalize spaces
        normalized = re.sub(r"[^\w\s]", " ", normalized)
        normalized = " ".join(normalized.split())

        return normalized

    def create_merge_actions(
        self, confirmed_matches: List[DuplicateMatch]
    ) -> List[MergeAction]:
        """Create merge actions for confirmed duplicate matches."""
        merge_actions = []

        for match in confirmed_matches:
            primary_id = match.individual1_id
            duplicate_id = match.individual2_id

            # Merge the individuals
            merged_individual = self._merge_individuals(
                self.all_individuals[primary_id],
                self.all_individuals[duplicate_id],
                primary_id,
            )

            # Update family references
            family_updates = self._update_family_references(duplicate_id, primary_id)

            merge_action = MergeAction(
                primary_id=primary_id,
                duplicate_id=duplicate_id,
                merged_individual=merged_individual,
                family_updates=family_updates,
            )

            merge_actions.append(merge_action)

        return merge_actions

    def _merge_individuals(
        self, primary: Individual, duplicate: Individual, merged_id: str
    ) -> Individual:
        """Merge two individuals, preferring non-empty fields."""
        merged = Individual(id=merged_id)

        # Merge names (prefer longer/more complete)
        merged.given_names = primary.given_names or duplicate.given_names
        if duplicate.given_names and len(duplicate.given_names) > len(
            primary.given_names or ""
        ):
            merged.given_names = duplicate.given_names

        merged.surname = primary.surname or duplicate.surname
        if duplicate.surname and len(duplicate.surname) > len(primary.surname or ""):
            merged.surname = duplicate.surname

        # Merge other fields (prefer primary, fallback to duplicate)
        merged.gender = primary.gender if primary.gender != "U" else duplicate.gender

        # Dates - prefer primary if available, otherwise use duplicate
        merged.birth_date = primary.birth_date or duplicate.birth_date
        merged.birth_year = primary.birth_year or duplicate.birth_year
        merged.birth_place = primary.birth_place or duplicate.birth_place

        merged.death_date = primary.death_date or duplicate.death_date
        merged.death_year = primary.death_year or duplicate.death_year
        merged.death_place = primary.death_place or duplicate.death_place

        # Other fields
        merged.occupation = primary.occupation or duplicate.occupation

        # Merge notes
        notes = []
        if primary.note:
            notes.append(primary.note)
        if duplicate.note:
            notes.append(duplicate.note)
        if notes:
            merged.note = "; ".join(notes)

        # Merge family relationships
        merged.parent_families = primary.parent_families | duplicate.parent_families
        merged.spouse_families = primary.spouse_families | duplicate.spouse_families

        return merged

    def _update_family_references(self, old_id: str, new_id: str) -> Dict[str, Family]:
        """Update family references after merging individuals."""
        updated_families = {}

        for family_id, family in self.all_families.items():
            family_changed = False
            updated_family = Family(
                id=family.id,
                husband_id=family.husband_id,
                wife_id=family.wife_id,
                children_ids=family.children_ids.copy(),
                marriage_date=family.marriage_date,
                marriage_year=family.marriage_year,
                marriage_place=family.marriage_place,
                divorce_date=family.divorce_date,
                divorce_year=family.divorce_year,
            )

            # Update husband reference
            if family.husband_id == old_id:
                updated_family.husband_id = new_id
                family_changed = True

            # Update wife reference
            if family.wife_id == old_id:
                updated_family.wife_id = new_id
                family_changed = True

            # Update children references
            if old_id in family.children_ids:
                updated_family.children_ids = [
                    new_id if child_id == old_id else child_id
                    for child_id in family.children_ids
                ]
                family_changed = True

            if family_changed:
                updated_families[family_id] = updated_family

        return updated_families

    def apply_merge_actions(self, merge_actions: List[MergeAction]):
        """Apply merge actions to the data."""
        logger.info(f"Applying {len(merge_actions)} merge actions...")

        for action in merge_actions:
            # Update individual
            self.all_individuals[action.primary_id] = action.merged_individual

            # Remove duplicate
            if action.duplicate_id in self.all_individuals:
                del self.all_individuals[action.duplicate_id]

            # Update families
            for family_id, updated_family in action.family_updates.items():
                self.all_families[family_id] = updated_family

            logger.info(f"Merged {action.duplicate_id} into {action.primary_id}")

    def get_deduplicated_data(self) -> Tuple[Dict[str, Individual], Dict[str, Family]]:
        """Get the deduplicated individuals and families."""
        return self.all_individuals, self.all_families


def deduplicate_gedcom_files(
    gedcom_files: List[str], auto_merge_threshold: float = 0.95
) -> Tuple[Dict[str, Individual], Dict[str, Family], List[DuplicateMatch]]:
    """
    Deduplicate individuals across multiple GEDCOM files.

    Returns:
        Tuple of (deduplicated_individuals, deduplicated_families, remaining_potential_duplicates)
    """
    from .gedcom_parser import parse_gedcom_file

    engine = DeduplicationEngine()

    # Load all GEDCOM files
    for gedcom_file in gedcom_files:
        individuals, families = parse_gedcom_file(gedcom_file)
        engine.add_gedcom_data(individuals, families, gedcom_file)

    # Detect duplicates
    duplicates = engine.detect_duplicates()

    # Auto-merge high-confidence matches
    auto_merge = [
        dup for dup in duplicates if dup.similarity_score >= auto_merge_threshold
    ]
    remaining_duplicates = [
        dup for dup in duplicates if dup.similarity_score < auto_merge_threshold
    ]

    if auto_merge:
        logger.info(f"Auto-merging {len(auto_merge)} high-confidence duplicates")
        merge_actions = engine.create_merge_actions(auto_merge)
        engine.apply_merge_actions(merge_actions)

    individuals, families = engine.get_deduplicated_data()

    return individuals, families, remaining_duplicates
