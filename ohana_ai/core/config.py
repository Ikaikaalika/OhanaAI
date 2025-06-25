"""Configuration management for OhanaAI."""

import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml

from .exceptions import ConfigError


@dataclass
class OhanaConfig:
    """Configuration class for OhanaAI."""

    # Model parameters
    node_features: int = 128
    hidden_dim: int = 256
    num_heads: int = 4
    num_layers: int = 3
    dropout: float = 0.1
    edge_types: int = 4

    # Training parameters
    learning_rate: float = 0.001
    batch_size: int = 32
    epochs: int = 100
    patience: int = 5
    weight_decay: float = 0.0001
    validation_split: float = 0.2

    # Data processing parameters
    min_age_diff: int = 12
    max_age_diff: int = 70
    date_tolerance: int = 5
    name_similarity_threshold: float = 0.8
    max_graph_size: int = 10000

    # Paths
    checkpoints_dir: str = "checkpoints/"
    logs_dir: str = "logs/"
    outputs_dir: str = "outputs/"
    model_path: str = "checkpoints/ohana_model.npz"

    # GUI parameters
    window_width: int = 1200
    window_height: int = 800
    canvas_width: int = 800
    canvas_height: int = 600
    confidence_threshold: float = 0.5
    max_predictions: int = 5

    # Logging parameters
    log_level: str = "INFO"
    log_format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    log_file: str = "logs/ohana_ai.log"

    def __post_init__(self) -> None:
        """Validate configuration after initialization."""
        self._validate_config()
        self._ensure_directories()

    def _validate_config(self) -> None:
        """Validate configuration parameters."""
        if self.node_features <= 0:
            raise ConfigError("node_features must be positive")
        if self.hidden_dim <= 0:
            raise ConfigError("hidden_dim must be positive")
        if self.num_heads <= 0:
            raise ConfigError("num_heads must be positive")
        if self.num_layers <= 0:
            raise ConfigError("num_layers must be positive")
        if not 0 <= self.dropout <= 1:
            raise ConfigError("dropout must be between 0 and 1")
        if self.learning_rate <= 0:
            raise ConfigError("learning_rate must be positive")
        if self.batch_size <= 0:
            raise ConfigError("batch_size must be positive")
        if self.epochs <= 0:
            raise ConfigError("epochs must be positive")
        if not 0 < self.validation_split < 1:
            raise ConfigError("validation_split must be between 0 and 1")
        if self.min_age_diff <= 0:
            raise ConfigError("min_age_diff must be positive")
        if self.max_age_diff <= self.min_age_diff:
            raise ConfigError("max_age_diff must be greater than min_age_diff")

    def _ensure_directories(self) -> None:
        """Ensure required directories exist."""
        directories = [
            self.checkpoints_dir,
            self.logs_dir,
            self.outputs_dir,
        ]
        for directory in directories:
            Path(directory).mkdir(parents=True, exist_ok=True)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> "OhanaConfig":
        """Create config from dictionary.

        Args:
            config_dict: Configuration dictionary

        Returns:
            OhanaConfig instance

        Raises:
            ConfigError: If configuration is invalid
        """
        # Flatten nested configuration
        flat_config = {}

        # Model parameters
        if "model" in config_dict:
            model_config = config_dict["model"]
            flat_config.update(
                {
                    "node_features": model_config.get("node_features", 128),
                    "hidden_dim": model_config.get("hidden_dim", 256),
                    "num_heads": model_config.get("num_heads", 4),
                    "num_layers": model_config.get("num_layers", 3),
                    "dropout": model_config.get("dropout", 0.1),
                    "edge_types": model_config.get("edge_types", 4),
                }
            )

        # Training parameters
        if "training" in config_dict:
            training_config = config_dict["training"]
            flat_config.update(
                {
                    "learning_rate": training_config.get("learning_rate", 0.001),
                    "batch_size": training_config.get("batch_size", 32),
                    "epochs": training_config.get("epochs", 100),
                    "patience": training_config.get("patience", 5),
                    "weight_decay": training_config.get("weight_decay", 0.0001),
                    "validation_split": training_config.get("validation_split", 0.2),
                }
            )

        # Data parameters
        if "data" in config_dict:
            data_config = config_dict["data"]
            flat_config.update(
                {
                    "min_age_diff": data_config.get("min_age_diff", 12),
                    "max_age_diff": data_config.get("max_age_diff", 70),
                    "date_tolerance": data_config.get("date_tolerance", 5),
                    "name_similarity_threshold": data_config.get(
                        "name_similarity_threshold", 0.8
                    ),
                    "max_graph_size": data_config.get("max_graph_size", 10000),
                }
            )

        # Path parameters
        if "paths" in config_dict:
            paths_config = config_dict["paths"]
            flat_config.update(
                {
                    "checkpoints_dir": paths_config.get("checkpoints", "checkpoints/"),
                    "logs_dir": paths_config.get("logs", "logs/"),
                    "outputs_dir": paths_config.get("outputs", "outputs/"),
                    "model_path": paths_config.get(
                        "models", "checkpoints/ohana_model.npz"
                    ),
                }
            )

        # GUI parameters
        if "gui" in config_dict:
            gui_config = config_dict["gui"]
            flat_config.update(
                {
                    "window_width": gui_config.get("window_width", 1200),
                    "window_height": gui_config.get("window_height", 800),
                    "canvas_width": gui_config.get("canvas_width", 800),
                    "canvas_height": gui_config.get("canvas_height", 600),
                    "confidence_threshold": gui_config.get("confidence_threshold", 0.5),
                    "max_predictions": gui_config.get("max_predictions", 5),
                }
            )

        # Logging parameters
        if "logging" in config_dict:
            logging_config = config_dict["logging"]
            flat_config.update(
                {
                    "log_level": logging_config.get("level", "INFO"),
                    "log_format": logging_config.get(
                        "format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
                    ),
                    "log_file": logging_config.get("file", "logs/ohana_ai.log"),
                }
            )

        try:
            return cls(**flat_config)
        except TypeError as e:
            raise ConfigError(f"Invalid configuration parameters: {e}")

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to nested dictionary format.

        Returns:
            Nested configuration dictionary
        """
        return {
            "model": {
                "node_features": self.node_features,
                "hidden_dim": self.hidden_dim,
                "num_heads": self.num_heads,
                "num_layers": self.num_layers,
                "dropout": self.dropout,
                "edge_types": self.edge_types,
            },
            "training": {
                "learning_rate": self.learning_rate,
                "batch_size": self.batch_size,
                "epochs": self.epochs,
                "patience": self.patience,
                "weight_decay": self.weight_decay,
                "validation_split": self.validation_split,
            },
            "data": {
                "min_age_diff": self.min_age_diff,
                "max_age_diff": self.max_age_diff,
                "date_tolerance": self.date_tolerance,
                "name_similarity_threshold": self.name_similarity_threshold,
                "max_graph_size": self.max_graph_size,
            },
            "paths": {
                "checkpoints": self.checkpoints_dir,
                "logs": self.logs_dir,
                "outputs": self.outputs_dir,
                "models": self.model_path,
            },
            "gui": {
                "window_width": self.window_width,
                "window_height": self.window_height,
                "canvas_width": self.canvas_width,
                "canvas_height": self.canvas_height,
                "confidence_threshold": self.confidence_threshold,
                "max_predictions": self.max_predictions,
            },
            "logging": {
                "level": self.log_level,
                "format": self.log_format,
                "file": self.log_file,
            },
        }


def load_config(config_path: Union[str, Path]) -> OhanaConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        OhanaConfig instance

    Raises:
        ConfigError: If configuration file cannot be loaded or is invalid
    """
    config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigError(f"Configuration file not found: {config_path}")

    try:
        with open(config_path, "r", encoding="utf-8") as f:
            config_dict = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML in configuration file: {e}")
    except OSError as e:
        raise ConfigError(f"Cannot read configuration file: {e}")

    if not isinstance(config_dict, dict):
        raise ConfigError("Configuration file must contain a dictionary")

    return OhanaConfig.from_dict(config_dict)


def setup_logging(config: OhanaConfig) -> None:
    """Setup logging configuration.

    Args:
        config: OhanaAI configuration
    """
    # Ensure log directory exists
    log_path = Path(config.log_file)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    # Configure logging
    logging.basicConfig(
        level=getattr(logging, config.log_level.upper()),
        format=config.log_format,
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler(config.log_file),
        ],
    )

    # Set MLX logging level to reduce noise
    logging.getLogger("mlx").setLevel(logging.WARNING)
