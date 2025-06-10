"""
Parent prediction and inference logic for OhanaAI.
Handles model inference, candidate generation, and constraint validation.
"""

import mlx.core as mx
import numpy as np
from typing import Dict, List, Tuple, Optional, NamedTuple
import logging
from dataclasses import dataclass
import yaml
import csv
import os

from .gnn_model import OhanaAIModel, create_model
from .graph_builder import GraphData, GraphBuilder
from .gedcom_parser import Individual, Family
from .trainer import OhanaAITrainer

logger = logging.getLogger(__name__)

@dataclass
class ParentPrediction:
    """Container for a parent prediction."""
    child_id: str
    child_name: str
    candidate_parent_id: str
    candidate_parent_name: str
    confidence_score: float
    age_difference: Optional[int]
    constraints_satisfied: bool
    constraint_violations: List[str]

class OhanaAIPredictor:
    """Predictor class for finding missing parents using trained OhanaAI model."""
    
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, 'r') as f:
            self.config = yaml.safe_load(f)
        
        self.model = None
        self.graph_data = None
        self.individuals = None
        self.families = None
        
        self.confidence_threshold = self.config['gui']['confidence_threshold']
        self.max_predictions = self.config['gui']['max_predictions']
        
    def load_model(self, model_path: Optional[str] = None):
        """Load trained model."""
        if model_path is None:
            model_path = self.config['paths']['models']
        
        logger.info(f"Loading model from {model_path}")
        
        # Initialize model
        self.model = create_model(self.config)
        
        # Load weights
        if os.path.exists(model_path):
            state_dict = mx.load(model_path)
            self.model.load_weights(list(state_dict.items()))
            logger.info("Model loaded successfully")
        else:
            logger.warning(f"Model file {model_path} not found. Using untrained model.")
    
    def prepare_data(self, individuals: Dict[str, Individual], 
                    families: Dict[str, Family]):
        """Prepare data for inference."""
        logger.info("Preparing data for inference...")
        
        self.individuals = individuals
        self.families = families
        
        # Build graph
        graph_builder = GraphBuilder()
        self.graph_data = graph_builder.build_graph(individuals, families)
        
        logger.info(f"Data prepared: {self.graph_data.num_nodes} nodes, {self.graph_data.num_edges} edges")
    
    def predict_missing_parents(self) -> List[ParentPrediction]:
        """Predict missing parents for individuals in the family tree."""
        if self.model is None:
            raise ValueError("Model not loaded. Call load_model() first.")
        
        if self.graph_data is None:
            raise ValueError("Data not prepared. Call prepare_data() first.")
        
        logger.info("Predicting missing parents...")
        
        # Get node embeddings
        with mx.no_grad():
            node_embeddings = self.model(
                self.graph_data.node_features,
                self.graph_data.edge_index,
                self.graph_data.edge_types
            )
        
        # Find individuals missing parents
        missing_parent_indices = mx.where(self.graph_data.missing_parents_mask)[0]
        
        if len(missing_parent_indices) == 0:
            logger.info("No individuals missing parents found.")
            return []
        
        predictions = []
        
        for child_idx in missing_parent_indices:
            child_id = self.graph_data.node_ids[int(child_idx)]
            child_individual = self.individuals[child_id]
            
            # Generate candidate parents
            candidate_pairs = self._generate_candidate_parents(child_idx, child_individual)
            
            if len(candidate_pairs) == 0:
                continue
            
            # Predict scores for candidates
            with mx.no_grad():
                candidate_pairs_mx = mx.array(candidate_pairs)
                scores = self.model.predict_parents(node_embeddings, candidate_pairs_mx)
                confidence_scores = mx.sigmoid(scores)
            
            # Convert to numpy for processing
            confidence_scores_np = np.array(confidence_scores)
            
            # Filter and rank candidates
            child_predictions = self._process_candidates(
                child_id, child_individual, candidate_pairs, confidence_scores_np
            )
            
            predictions.extend(child_predictions)
        
        # Sort all predictions by confidence
        predictions.sort(key=lambda x: x.confidence_score, reverse=True)
        
        logger.info(f"Generated {len(predictions)} parent predictions")
        return predictions
    
    def _generate_candidate_parents(self, child_idx: int, 
                                  child_individual: Individual) -> List[Tuple[int, int]]:
        """Generate candidate parent-child pairs for a given child."""
        candidates = []
        
        for parent_idx, parent_id in enumerate(self.graph_data.node_ids):
            if parent_idx == child_idx:
                continue
            
            parent_individual = self.individuals[parent_id]
            
            # Basic eligibility checks
            if self._is_eligible_parent(parent_individual, child_individual):
                candidates.append([parent_idx, child_idx])
        
        return candidates
    
    def _is_eligible_parent(self, parent: Individual, child: Individual) -> bool:
        """Check if an individual is eligible to be a parent of another."""
        # Age constraints
        if parent.birth_year and child.birth_year:
            age_diff = child.birth_year - parent.birth_year
            min_age = self.config['data']['min_age_diff']
            max_age = self.config['data']['max_age_diff']
            
            if not (min_age <= age_diff <= max_age):
                return False
        
        # Parent must be alive when child was born
        if (parent.death_year and child.birth_year and 
            parent.death_year < child.birth_year):
            return False
        
        # Don't suggest if already known to be parent
        if self._is_known_parent(parent.id, child.id):
            return False
        
        # Don't suggest if they're known siblings
        if self._are_known_siblings(parent.id, child.id):
            return False
        
        return True
    
    def _is_known_parent(self, parent_id: str, child_id: str) -> bool:
        """Check if parent-child relationship already exists."""
        child = self.individuals[child_id]
        
        for family_id in child.parent_families:
            if family_id in self.families:
                family = self.families[family_id]
                if parent_id in [family.husband_id, family.wife_id]:
                    return True
        
        return False
    
    def _are_known_siblings(self, id1: str, id2: str) -> bool:
        """Check if two individuals are known siblings."""
        individual1 = self.individuals[id1]
        individual2 = self.individuals[id2]
        
        # Check if they share any parent families
        shared_families = individual1.parent_families & individual2.parent_families
        return len(shared_families) > 0
    
    def _process_candidates(self, child_id: str, child_individual: Individual,
                          candidate_pairs: List[Tuple[int, int]], 
                          confidence_scores: np.ndarray) -> List[ParentPrediction]:
        """Process and filter candidate parents."""
        predictions = []
        
        for i, (parent_idx, child_idx) in enumerate(candidate_pairs):
            confidence = float(confidence_scores[i])
            
            # Skip low-confidence predictions
            if confidence < self.confidence_threshold:
                continue
            
            parent_id = self.graph_data.node_ids[parent_idx]
            parent_individual = self.individuals[parent_id]
            
            # Validate constraints
            constraints_satisfied, violations = self._validate_constraints(
                parent_individual, child_individual
            )
            
            # Calculate age difference
            age_diff = None
            if parent_individual.birth_year and child_individual.birth_year:
                age_diff = child_individual.birth_year - parent_individual.birth_year
            
            prediction = ParentPrediction(
                child_id=child_id,
                child_name=child_individual.full_name,
                candidate_parent_id=parent_id,
                candidate_parent_name=parent_individual.full_name,
                confidence_score=confidence,
                age_difference=age_diff,
                constraints_satisfied=constraints_satisfied,
                constraint_violations=violations
            )
            
            predictions.append(prediction)
        
        # Sort by confidence and limit results
        predictions.sort(key=lambda x: x.confidence_score, reverse=True)
        return predictions[:self.max_predictions]
    
    def _validate_constraints(self, parent: Individual, 
                            child: Individual) -> Tuple[bool, List[str]]:
        """Validate genealogical constraints for parent-child relationship."""
        violations = []
        
        # Age constraints
        if parent.birth_year and child.birth_year:
            age_diff = child.birth_year - parent.birth_year
            min_age = self.config['data']['min_age_diff']
            max_age = self.config['data']['max_age_diff']
            
            if age_diff < min_age:
                violations.append(f"Parent too young (age diff: {age_diff})")
            elif age_diff > max_age:
                violations.append(f"Parent too old (age diff: {age_diff})")
        
        # Death constraints
        if (parent.death_year and child.birth_year and 
            parent.death_year < child.birth_year):
            violations.append("Parent died before child was born")
        
        # Gender consistency (if we have multiple parents)
        # This would require more complex logic to check existing spouses
        
        # Geographic constraints (basic check)
        if (parent.birth_place and child.birth_place and 
            parent.birth_place.lower() != child.birth_place.lower()):
            # This is a soft constraint - different places don't rule out relationship
            pass
        
        return len(violations) == 0, violations
    
    def export_predictions_csv(self, predictions: List[ParentPrediction], 
                              filename: str = "parent_predictions.csv"):
        """Export predictions to CSV file."""
        output_dir = self.config['paths']['outputs']
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', newline='', encoding='utf-8') as csvfile:
            fieldnames = [
                'child_id', 'child_name', 'candidate_parent_id', 'candidate_parent_name',
                'confidence_score', 'age_difference', 'constraints_satisfied', 
                'constraint_violations'
            ]
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            
            writer.writeheader()
            for pred in predictions:
                writer.writerow({
                    'child_id': pred.child_id,
                    'child_name': pred.child_name,
                    'candidate_parent_id': pred.candidate_parent_id,
                    'candidate_parent_name': pred.candidate_parent_name,
                    'confidence_score': f"{pred.confidence_score:.4f}",
                    'age_difference': pred.age_difference or 'Unknown',
                    'constraints_satisfied': pred.constraints_satisfied,
                    'constraint_violations': '; '.join(pred.constraint_violations)
                })
        
        logger.info(f"Predictions exported to {filepath}")
    
    def export_predictions_gedcom(self, predictions: List[ParentPrediction],
                                 filename: str = "predicted_families.ged"):
        """Export predictions as GEDCOM supplement."""
        output_dir = self.config['paths']['outputs']
        os.makedirs(output_dir, exist_ok=True)
        
        filepath = os.path.join(output_dir, filename)
        
        with open(filepath, 'w', encoding='utf-8') as gedfile:
            gedfile.write("0 HEAD\n")
            gedfile.write("1 SOUR OhanaAI\n")
            gedfile.write("2 VERS 1.0\n")
            gedfile.write("2 NAME OhanaAI Parent Predictions\n")
            gedfile.write("1 GEDC\n")
            gedfile.write("2 VERS 5.5.1\n")
            gedfile.write("2 FORM LINEAGE-LINKED\n")
            gedfile.write("1 CHAR UTF-8\n")
            
            # Group predictions by child to create families
            child_predictions = {}
            for pred in predictions:
                if pred.constraints_satisfied:  # Only export constraint-satisfying predictions
                    if pred.child_id not in child_predictions:
                        child_predictions[pred.child_id] = []
                    child_predictions[pred.child_id].append(pred)
            
            family_counter = 1
            for child_id, preds in child_predictions.items():
                # Create a family for each child with predicted parents
                family_id = f"F{family_counter:04d}"
                gedfile.write(f"0 @{family_id}@ FAM\n")
                
                # Add best parent predictions (up to 2)
                for i, pred in enumerate(preds[:2]):  # Max 2 parents
                    parent = self.individuals[pred.candidate_parent_id]
                    if parent.gender == 'M' or (parent.gender == 'U' and i == 0):
                        gedfile.write(f"1 HUSB @{pred.candidate_parent_id}@\n")
                    else:
                        gedfile.write(f"1 WIFE @{pred.candidate_parent_id}@\n")
                
                gedfile.write(f"1 CHIL @{child_id}@\n")
                gedfile.write(f"1 NOTE Predicted family by OhanaAI\n")
                gedfile.write(f"2 CONT Confidence scores: {', '.join([f'{p.confidence_score:.3f}' for p in preds[:2]])}\n")
                
                family_counter += 1
            
            gedfile.write("0 TRLR\n")
        
        logger.info(f"GEDCOM predictions exported to {filepath}")

def predict_parents(gedcom_files: List[str], model_path: Optional[str] = None,
                   config_path: str = "config.yaml") -> List[ParentPrediction]:
    """Convenience function to predict parents from GEDCOM files."""
    from .gedcom_parser import parse_gedcom_file
    
    # Parse all GEDCOM files
    all_individuals = {}
    all_families = {}
    
    for gedcom_file in gedcom_files:
        individuals, families = parse_gedcom_file(gedcom_file)
        all_individuals.update(individuals)
        all_families.update(families)
    
    # Predict parents
    predictor = OhanaAIPredictor(config_path)
    predictor.load_model(model_path)
    predictor.prepare_data(all_individuals, all_families)
    
    predictions = predictor.predict_missing_parents()
    
    # Export results
    predictor.export_predictions_csv(predictions)
    predictor.export_predictions_gedcom(predictions)
    
    return predictions