"""
OhanaAI - Genealogical Parent Prediction using Graph Neural Networks

A complete system for predicting missing parents in genealogical GEDCOM files
using Graph Attention Networks implemented with Apple's MLX framework.
"""

__version__ = "1.0.0"
__author__ = "OhanaAI Team"

from .gedcom_parser import Individual, Family, GEDCOMParser, parse_gedcom_file
from .graph_builder import GraphData, GraphBuilder, build_graph_from_gedcom
from .gnn_model import OhanaAIModel, GraphAttentionLayer, ContrastiveLoss, create_model
from .trainer import OhanaAITrainer, TrainingMetrics, train_model
from .predictor import OhanaAIPredictor, ParentPrediction, predict_parents
from .data_deduplication import DeduplicationEngine, DuplicateMatch, deduplicate_gedcom_files
from .gui import OhanaAIGUI

__all__ = [
    # Core data structures
    'Individual',
    'Family', 
    'GraphData',
    'ParentPrediction',
    'DuplicateMatch',
    'TrainingMetrics',
    
    # Parsers and builders
    'GEDCOMParser',
    'parse_gedcom_file',
    'GraphBuilder',
    'build_graph_from_gedcom',
    
    # Models
    'OhanaAIModel',
    'GraphAttentionLayer',
    'ContrastiveLoss',
    'create_model',
    
    # Training and prediction
    'OhanaAITrainer',
    'train_model',
    'OhanaAIPredictor', 
    'predict_parents',
    
    # Deduplication
    'DeduplicationEngine',
    'deduplicate_gedcom_files',
    
    # GUI
    'OhanaAIGUI',
]