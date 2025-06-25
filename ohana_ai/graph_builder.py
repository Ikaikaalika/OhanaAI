"""
Graph construction system for converting family trees to MLX format.
Creates node features and edge lists for Graph Neural Network processing.
"""

import logging
from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Set, Tuple

import mlx.core as mx
import numpy as np
import yaml

from .gedcom_parser import Family, Individual

logger = logging.getLogger(__name__)


@dataclass
class GraphData:
    """Container for graph data in MLX format."""

    node_features: mx.array  # Shape: (num_nodes, feature_dim)
    edge_index: mx.array  # Shape: (2, num_edges)
    edge_types: mx.array  # Shape: (num_edges,)
    node_ids: List[str]  # Original node IDs
    missing_parents_mask: mx.array  # Shape: (num_nodes,) - 1 if missing parents

    # Metadata
    num_nodes: int
    num_edges: int
    feature_dim: int


class GraphBuilder:
    """Converts family tree data to graph format for GNN processing."""

    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.feature_dim = self.config["model"]["node_features"]
        self.edge_types = {
            "parent_to_child": 0,
            "child_to_parent": 1,
            "spouse": 2,
            "sibling": 3,
        }

        # For normalization
        self.min_year = 1800
        self.max_year = 2025

    def build_graph(
        self, individuals: Dict[str, Individual], families: Dict[str, Family]
    ) -> GraphData:
        """Build graph from individuals and families."""
        logger.info(
            f"Building graph from {len(individuals)} individuals and {len(families)} families"
        )

        # Filter out individuals without sufficient information
        valid_individuals = self._filter_valid_individuals(individuals)
        logger.info(f"Using {len(valid_individuals)} valid individuals")

        # Create node mappings
        node_ids = list(valid_individuals.keys())
        id_to_idx = {node_id: idx for idx, node_id in enumerate(node_ids)}

        # Build node features
        node_features = self._build_node_features(valid_individuals, node_ids)

        # Build edges
        edge_index, edge_types = self._build_edges(
            valid_individuals, families, id_to_idx
        )

        # Create missing parents mask
        missing_parents_mask = self._create_missing_parents_mask(
            valid_individuals, families, node_ids
        )

        graph_data = GraphData(
            node_features=mx.array(node_features),
            edge_index=mx.array(edge_index),
            edge_types=mx.array(edge_types),
            node_ids=node_ids,
            missing_parents_mask=mx.array(missing_parents_mask),
            num_nodes=len(node_ids),
            num_edges=len(edge_types),
            feature_dim=self.feature_dim,
        )

        logger.info(
            f"Built graph with {graph_data.num_nodes} nodes and {graph_data.num_edges} edges"
        )
        return graph_data

    def _filter_valid_individuals(
        self, individuals: Dict[str, Individual]
    ) -> Dict[str, Individual]:
        """Filter individuals with sufficient information for modeling."""
        valid = {}
        for ind_id, individual in individuals.items():
            # Require at least a name or birth year
            if individual.full_name.strip() or individual.birth_year is not None:
                valid[ind_id] = individual
        return valid

    def _build_node_features(
        self, individuals: Dict[str, Individual], node_ids: List[str]
    ) -> np.ndarray:
        """Build node feature matrix."""
        features = np.zeros((len(node_ids), self.feature_dim))

        for idx, node_id in enumerate(node_ids):
            individual = individuals[node_id]
            feature_vector = self._individual_to_features(individual)
            features[idx] = feature_vector

        return features

    def _individual_to_features(self, individual: Individual) -> np.ndarray:
        """Convert individual to feature vector."""
        features = np.zeros(self.feature_dim)
        idx = 0

        # Gender features (3 dimensions: M, F, U)
        if individual.gender == "M":
            features[idx] = 1.0
        elif individual.gender == "F":
            features[idx + 1] = 1.0
        else:
            features[idx + 2] = 1.0
        idx += 3

        # Birth year (normalized)
        if individual.birth_year:
            normalized_year = (individual.birth_year - self.min_year) / (
                self.max_year - self.min_year
            )
            features[idx] = max(0.0, min(1.0, normalized_year))
        idx += 1

        # Death year (normalized)
        if individual.death_year:
            normalized_year = (individual.death_year - self.min_year) / (
                self.max_year - self.min_year
            )
            features[idx] = max(0.0, min(1.0, normalized_year))
        idx += 1

        # Age at death (if known)
        if individual.birth_year and individual.death_year:
            age = individual.death_year - individual.birth_year
            features[idx] = min(1.0, age / 100.0)  # Normalize to 0-1
        idx += 1

        # Name features (basic statistics)
        full_name = individual.full_name
        if full_name:
            # Name length (normalized)
            features[idx] = min(1.0, len(full_name) / 50.0)
            idx += 1

            # Number of name parts
            name_parts = len(full_name.split())
            features[idx] = min(1.0, name_parts / 5.0)
            idx += 1

            # Has surname
            features[idx] = 1.0 if individual.surname else 0.0
            idx += 1
        else:
            idx += 3

        # Location features (basic encoding)
        if individual.birth_place:
            # Simple hash-based encoding for locations
            location_hash = hash(individual.birth_place.lower()) % 100
            features[idx] = location_hash / 100.0
        idx += 1

        # Has occupation
        features[idx] = 1.0 if individual.occupation else 0.0
        idx += 1

        # Number of families (proxy for connectivity)
        num_families = len(individual.parent_families) + len(individual.spouse_families)
        features[idx] = min(1.0, num_families / 5.0)
        idx += 1

        # Pad remaining features with name character statistics
        if full_name and idx < self.feature_dim:
            # Simple character frequency features
            char_counts = defaultdict(int)
            for char in full_name.lower():
                if char.isalpha():
                    char_counts[char] += 1

            # Most common characters
            common_chars = sorted(char_counts.items(), key=lambda x: x[1], reverse=True)
            for i, (char, count) in enumerate(
                common_chars[: min(10, self.feature_dim - idx)]
            ):
                features[idx + i] = count / len(full_name)

        return features

    def _build_edges(
        self,
        individuals: Dict[str, Individual],
        families: Dict[str, Family],
        id_to_idx: Dict[str, int],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Build edge index and edge type arrays."""
        edges = []
        edge_types = []

        # Process family relationships
        for family in families.values():
            parent_ids = family.parent_ids
            child_ids = family.children_ids

            # Parent-child relationships
            for parent_id in parent_ids:
                if parent_id in id_to_idx:
                    parent_idx = id_to_idx[parent_id]
                    for child_id in child_ids:
                        if child_id in id_to_idx:
                            child_idx = id_to_idx[child_id]

                            # Parent -> Child edge
                            edges.append([parent_idx, child_idx])
                            edge_types.append(self.edge_types["parent_to_child"])

                            # Child -> Parent edge
                            edges.append([child_idx, parent_idx])
                            edge_types.append(self.edge_types["child_to_parent"])

            # Spouse relationships
            if (
                family.husband_id
                and family.wife_id
                and family.husband_id in id_to_idx
                and family.wife_id in id_to_idx
            ):
                husband_idx = id_to_idx[family.husband_id]
                wife_idx = id_to_idx[family.wife_id]

                # Bidirectional spouse edges
                edges.append([husband_idx, wife_idx])
                edge_types.append(self.edge_types["spouse"])
                edges.append([wife_idx, husband_idx])
                edge_types.append(self.edge_types["spouse"])

            # Sibling relationships
            valid_children = [cid for cid in child_ids if cid in id_to_idx]
            for i, child1_id in enumerate(valid_children):
                for child2_id in valid_children[i + 1 :]:
                    child1_idx = id_to_idx[child1_id]
                    child2_idx = id_to_idx[child2_id]

                    # Bidirectional sibling edges
                    edges.append([child1_idx, child2_idx])
                    edge_types.append(self.edge_types["sibling"])
                    edges.append([child2_idx, child1_idx])
                    edge_types.append(self.edge_types["sibling"])

        if not edges:
            # Create empty arrays with correct shape
            edge_index = np.zeros((2, 0), dtype=np.int32)
            edge_types_array = np.zeros((0,), dtype=np.int32)
        else:
            edge_index = np.array(edges).T  # Shape: (2, num_edges)
            edge_types_array = np.array(edge_types, dtype=np.int32)

        return edge_index, edge_types_array

    def _create_missing_parents_mask(
        self,
        individuals: Dict[str, Individual],
        families: Dict[str, Family],
        node_ids: List[str],
    ) -> np.ndarray:
        """Create mask indicating which individuals are missing parents."""
        mask = np.zeros(len(node_ids), dtype=np.float32)

        for idx, node_id in enumerate(node_ids):
            individual = individuals[node_id]

            # Check if individual has known parents
            has_parents = False
            for family_id in individual.parent_families:
                if family_id in families:
                    family = families[family_id]
                    if family.husband_id or family.wife_id:
                        has_parents = True
                        break

            # Mark as missing parents if no parents found and not too old
            # (avoid marking very old individuals who legitimately might not have parent records)
            if not has_parents:
                if individual.birth_year is None or individual.birth_year > 1850:
                    mask[idx] = 1.0

        return mask

    def create_training_pairs(
        self,
        graph_data: GraphData,
        individuals: Dict[str, Individual],
        families: Dict[str, Family],
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Create positive and negative training pairs for parent prediction."""
        positive_pairs = []
        negative_pairs = []

        # Get existing parent-child relationships as positive examples
        for family in families.values():
            parent_ids = family.parent_ids
            child_ids = family.children_ids

            for parent_id in parent_ids:
                if parent_id in individuals:
                    parent_individual = individuals[parent_id]
                    for child_id in child_ids:
                        if child_id in individuals:
                            child_individual = individuals[child_id]

                            # Check age constraint
                            if self._is_valid_parent_child_pair(
                                parent_individual, child_individual
                            ):
                                try:
                                    parent_idx = graph_data.node_ids.index(parent_id)
                                    child_idx = graph_data.node_ids.index(child_id)
                                    positive_pairs.append([parent_idx, child_idx])
                                except ValueError:
                                    continue

        # Create negative examples
        # Sample random pairs that are not parent-child relationships
        positive_set = set(tuple(pair) for pair in positive_pairs)
        num_negative = min(len(positive_pairs) * 3, 10000)  # 3:1 ratio, max 10k

        import random

        random.seed(42)

        attempts = 0
        max_attempts = num_negative * 10

        while len(negative_pairs) < num_negative and attempts < max_attempts:
            parent_idx = random.randint(0, graph_data.num_nodes - 1)
            child_idx = random.randint(0, graph_data.num_nodes - 1)

            if parent_idx != child_idx and (parent_idx, child_idx) not in positive_set:
                parent_id = graph_data.node_ids[parent_idx]
                child_id = graph_data.node_ids[child_idx]
                parent_individual = individuals[parent_id]
                child_individual = individuals[child_id]

                # Only add if they could theoretically be parent-child (age constraints)
                if self._is_valid_parent_child_pair(
                    parent_individual, child_individual
                ):
                    negative_pairs.append([parent_idx, child_idx])

            attempts += 1

        logger.info(
            f"Created {len(positive_pairs)} positive and {len(negative_pairs)} negative training pairs"
        )

        return np.array(positive_pairs), np.array(negative_pairs)

    def _is_valid_parent_child_pair(
        self, parent: Individual, child: Individual
    ) -> bool:
        """Check if two individuals could theoretically be parent and child."""
        if not parent.birth_year or not child.birth_year:
            return True  # Can't verify without birth years

        age_diff = child.birth_year - parent.birth_year
        min_age = self.config["data"]["min_age_diff"]
        max_age = self.config["data"]["max_age_diff"]

        if not (min_age <= age_diff <= max_age):
            return False

        # Check if parent was alive when child was born
        if parent.death_year and parent.death_year < child.birth_year:
            return False

        return True


def build_graph_from_gedcom(
    gedcom_file: str, config_path: str = "config.yaml"
) -> GraphData:
    """Convenience function to build graph directly from GEDCOM file."""
    from .gedcom_parser import parse_gedcom_file

    individuals, families = parse_gedcom_file(gedcom_file)
    builder = GraphBuilder(config_path)
    return builder.build_graph(individuals, families)
