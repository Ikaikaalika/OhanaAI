"""
OhanaAI - Genealogical Parent Prediction using Graph Neural Networks

A complete system for predicting missing parents in genealogical GEDCOM files
using Graph Attention Networks implemented with Apple's MLX framework.
"""

__version__ = "1.0.0"
__author__ = "OhanaAI Team"

from .data_deduplication import (
    DeduplicationEngine,
    DuplicateMatch,
    deduplicate_gedcom_files,
)
from .gedcom_parser import Family, GEDCOMParser, Individual, parse_gedcom_file
from .gnn_model import ContrastiveLoss, GraphAttentionLayer, OhanaAIModel, create_model
from .graph_builder import GraphBuilder, GraphData, build_graph_from_gedcom
from .gui import OhanaAIGUI
from .predictor import OhanaAIPredictor, ParentPrediction, predict_parents
from .trainer import OhanaAITrainer, TrainingMetrics, train_model

__all__ = [
    # Core data structures
    "Individual",
    "Family",
    "GraphData",
    "ParentPrediction",
    "DuplicateMatch",
    "TrainingMetrics",
    # Parsers and builders
    "GEDCOMParser",
    "parse_gedcom_file",
    "GraphBuilder",
    "build_graph_from_gedcom",
    # Models
    "OhanaAIModel",
    "GraphAttentionLayer",
    "ContrastiveLoss",
    "create_model",
    # Training and prediction
    "OhanaAITrainer",
    "train_model",
    "OhanaAIPredictor",
    "predict_parents",
    # Deduplication
    "DeduplicationEngine",
    "deduplicate_gedcom_files",
    # GUI
    "OhanaAIGUI",
]
