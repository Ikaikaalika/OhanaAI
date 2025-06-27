"""
Model versioning and experiment tracking for OhanaAI.
"""

import json
import logging
import shutil
from dataclasses import dataclass, asdict
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np


@dataclass
class ModelMetadata:
    """Metadata for a trained model."""
    
    version: str
    created_at: datetime
    config: Dict[str, Any]
    metrics: Dict[str, float]
    training_data: Dict[str, Any]  # Info about training data used
    model_size_mb: float
    training_duration_seconds: float
    parent_version: Optional[str] = None
    tags: List[str] = None
    notes: str = ""
    is_production: bool = False
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


@dataclass
class ExperimentRun:
    """Represents a single experiment run."""
    
    id: str
    name: str
    created_at: datetime
    config: Dict[str, Any]
    status: str  # running, completed, failed, cancelled
    metrics: Dict[str, float]
    artifacts: Dict[str, str]  # artifact_name -> path/url
    duration_seconds: float = 0.0
    error_message: str = ""
    parent_run_id: Optional[str] = None
    tags: List[str] = None
    
    def __post_init__(self):
        if self.tags is None:
            self.tags = []


class ModelVersionManager:
    """Manages model versions and experiment tracking."""
    
    def __init__(self, base_path: str = "experiments"):
        """Initialize model version manager.
        
        Args:
            base_path: Base directory for storing experiments and models
        """
        self.base_path = Path(base_path)
        self.models_path = self.base_path / "models"
        self.experiments_path = self.base_path / "experiments"
        self.metadata_path = self.base_path / "metadata"
        
        # Create directories
        for path in [self.models_path, self.experiments_path, self.metadata_path]:
            path.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)

    def create_model_version(self, model_path: Path, config: Dict[str, Any],
                           metrics: Dict[str, float], training_data: Dict[str, Any],
                           training_duration: float, parent_version: Optional[str] = None,
                           tags: Optional[List[str]] = None, notes: str = "") -> str:
        """Create a new model version.
        
        Args:
            model_path: Path to the trained model file
            config: Training configuration used
            metrics: Final training metrics
            training_data: Information about training data
            training_duration: Training duration in seconds
            parent_version: Parent model version (for incremental training)
            tags: Optional tags for the model
            notes: Optional notes about the model
            
        Returns:
            Version identifier for the created model
        """
        # Generate version identifier
        version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Create version directory
        version_path = self.models_path / version
        version_path.mkdir(exist_ok=True)
        
        # Copy model file
        model_dest = version_path / "model.npz"
        shutil.copy2(model_path, model_dest)
        
        # Calculate model size
        model_size_mb = model_dest.stat().st_size / (1024 * 1024)
        
        # Create metadata
        metadata = ModelMetadata(
            version=version,
            created_at=datetime.now(),
            config=config,
            metrics=metrics,
            training_data=training_data,
            model_size_mb=model_size_mb,
            training_duration_seconds=training_duration,
            parent_version=parent_version,
            tags=tags or [],
            notes=notes
        )
        
        # Save metadata
        metadata_file = version_path / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        # Update version registry
        self._update_version_registry(metadata)
        
        self.logger.info(f"Created model version {version}")
        return version

    def get_model_metadata(self, version: str) -> Optional[ModelMetadata]:
        """Get metadata for a model version.
        
        Args:
            version: Model version identifier
            
        Returns:
            Model metadata or None if not found
        """
        metadata_file = self.models_path / version / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, "r") as f:
            data = json.load(f)
        
        # Convert datetime strings back to datetime objects
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return ModelMetadata(**data)

    def list_model_versions(self, limit: int = 50, 
                           tags: Optional[List[str]] = None) -> List[ModelMetadata]:
        """List model versions.
        
        Args:
            limit: Maximum number of versions to return
            tags: Filter by tags (optional)
            
        Returns:
            List of model metadata, sorted by creation time (newest first)
        """
        versions = []
        
        for version_dir in self.models_path.iterdir():
            if version_dir.is_dir():
                metadata = self.get_model_metadata(version_dir.name)
                if metadata:
                    # Filter by tags if specified
                    if tags and not any(tag in metadata.tags for tag in tags):
                        continue
                    versions.append(metadata)
        
        # Sort by creation time (newest first)
        versions.sort(key=lambda x: x.created_at, reverse=True)
        
        return versions[:limit]

    def get_production_model(self) -> Optional[ModelMetadata]:
        """Get the current production model.
        
        Returns:
            Production model metadata or None if no production model
        """
        for metadata in self.list_model_versions():
            if metadata.is_production:
                return metadata
        return None

    def promote_to_production(self, version: str) -> bool:
        """Promote a model version to production.
        
        Args:
            version: Model version to promote
            
        Returns:
            True if successful, False otherwise
        """
        # First, demote current production model
        current_production = self.get_production_model()
        if current_production:
            self._update_production_status(current_production.version, False)
        
        # Promote new version
        success = self._update_production_status(version, True)
        
        if success:
            self.logger.info(f"Promoted model version {version} to production")
        
        return success

    def compare_models(self, version1: str, version2: str) -> Dict[str, Any]:
        """Compare two model versions.
        
        Args:
            version1: First model version
            version2: Second model version
            
        Returns:
            Comparison results
        """
        metadata1 = self.get_model_metadata(version1)
        metadata2 = self.get_model_metadata(version2)
        
        if not metadata1 or not metadata2:
            raise ValueError("One or both model versions not found")
        
        comparison = {
            "versions": [version1, version2],
            "metrics_comparison": {},
            "config_differences": [],
            "performance_delta": {}
        }
        
        # Compare metrics
        all_metrics = set(metadata1.metrics.keys()) | set(metadata2.metrics.keys())
        for metric in all_metrics:
            val1 = metadata1.metrics.get(metric, 0.0)
            val2 = metadata2.metrics.get(metric, 0.0)
            comparison["metrics_comparison"][metric] = {
                version1: val1,
                version2: val2,
                "difference": val2 - val1,
                "percent_change": ((val2 - val1) / val1 * 100) if val1 != 0 else 0.0
            }
        
        # Find config differences
        config_diff = self._find_config_differences(metadata1.config, metadata2.config)
        comparison["config_differences"] = config_diff
        
        # Calculate performance improvements
        for metric in ["val_accuracy", "train_accuracy"]:
            if metric in comparison["metrics_comparison"]:
                comparison["performance_delta"][metric] = comparison["metrics_comparison"][metric]["difference"]
        
        return comparison

    def delete_model_version(self, version: str, force: bool = False) -> bool:
        """Delete a model version.
        
        Args:
            version: Model version to delete
            force: Force deletion even if it's the production model
            
        Returns:
            True if successful, False otherwise
        """
        metadata = self.get_model_metadata(version)
        if not metadata:
            return False
        
        # Prevent deletion of production model unless forced
        if metadata.is_production and not force:
            self.logger.warning(f"Cannot delete production model {version} without force=True")
            return False
        
        # Delete version directory
        version_path = self.models_path / version
        shutil.rmtree(version_path)
        
        # Update registry
        self._remove_from_version_registry(version)
        
        self.logger.info(f"Deleted model version {version}")
        return True

    def create_experiment_run(self, name: str, config: Dict[str, Any],
                            parent_run_id: Optional[str] = None,
                            tags: Optional[List[str]] = None) -> str:
        """Create a new experiment run.
        
        Args:
            name: Experiment name
            config: Experiment configuration
            parent_run_id: Parent experiment run ID (optional)
            tags: Experiment tags (optional)
            
        Returns:
            Experiment run ID
        """
        run_id = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"
        
        experiment_run = ExperimentRun(
            id=run_id,
            name=name,
            created_at=datetime.now(),
            config=config,
            status="running",
            metrics={},
            artifacts={},
            parent_run_id=parent_run_id,
            tags=tags or []
        )
        
        # Create experiment directory
        run_path = self.experiments_path / run_id
        run_path.mkdir(exist_ok=True)
        
        # Save experiment metadata
        self._save_experiment_run(experiment_run)
        
        self.logger.info(f"Created experiment run {run_id}: {name}")
        return run_id

    def update_experiment_run(self, run_id: str, status: Optional[str] = None,
                            metrics: Optional[Dict[str, float]] = None,
                            artifacts: Optional[Dict[str, str]] = None,
                            error_message: str = "") -> bool:
        """Update an experiment run.
        
        Args:
            run_id: Experiment run ID
            status: New status (optional)
            metrics: Updated metrics (optional)
            artifacts: New artifacts (optional)
            error_message: Error message if status is failed
            
        Returns:
            True if successful, False otherwise
        """
        experiment_run = self._load_experiment_run(run_id)
        if not experiment_run:
            return False
        
        # Update fields
        if status:
            experiment_run.status = status
        if metrics:
            experiment_run.metrics.update(metrics)
        if artifacts:
            experiment_run.artifacts.update(artifacts)
        if error_message:
            experiment_run.error_message = error_message
        
        # Calculate duration if completed
        if status in ["completed", "failed", "cancelled"]:
            duration = (datetime.now() - experiment_run.created_at).total_seconds()
            experiment_run.duration_seconds = duration
        
        # Save updated experiment
        self._save_experiment_run(experiment_run)
        
        return True

    def get_experiment_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Get an experiment run.
        
        Args:
            run_id: Experiment run ID
            
        Returns:
            Experiment run or None if not found
        """
        return self._load_experiment_run(run_id)

    def list_experiment_runs(self, limit: int = 50,
                           status: Optional[str] = None,
                           tags: Optional[List[str]] = None) -> List[ExperimentRun]:
        """List experiment runs.
        
        Args:
            limit: Maximum number of runs to return
            status: Filter by status (optional)
            tags: Filter by tags (optional)
            
        Returns:
            List of experiment runs, sorted by creation time (newest first)
        """
        runs = []
        
        for run_dir in self.experiments_path.iterdir():
            if run_dir.is_dir():
                run = self._load_experiment_run(run_dir.name)
                if run:
                    # Apply filters
                    if status and run.status != status:
                        continue
                    if tags and not any(tag in run.tags for tag in tags):
                        continue
                    runs.append(run)
        
        # Sort by creation time (newest first)
        runs.sort(key=lambda x: x.created_at, reverse=True)
        
        return runs[:limit]

    def cleanup_old_experiments(self, max_age_days: int = 30,
                              keep_completed: bool = True) -> int:
        """Clean up old experiment runs.
        
        Args:
            max_age_days: Maximum age of experiments to keep
            keep_completed: Whether to keep completed experiments longer
            
        Returns:
            Number of experiments deleted
        """
        deleted_count = 0
        cutoff_date = datetime.now() - datetime.timedelta(days=max_age_days)
        
        for run_dir in self.experiments_path.iterdir():
            if run_dir.is_dir():
                run = self._load_experiment_run(run_dir.name)
                if run and run.created_at < cutoff_date:
                    # Skip completed experiments if requested
                    if keep_completed and run.status == "completed":
                        continue
                    
                    # Delete experiment
                    shutil.rmtree(run_dir)
                    deleted_count += 1
                    self.logger.info(f"Deleted old experiment run {run.id}")
        
        return deleted_count

    def _update_version_registry(self, metadata: ModelMetadata) -> None:
        """Update the version registry file."""
        registry_file = self.metadata_path / "versions.json"
        
        # Load existing registry
        if registry_file.exists():
            with open(registry_file, "r") as f:
                registry = json.load(f)
        else:
            registry = {"versions": []}
        
        # Add new version
        version_info = {
            "version": metadata.version,
            "created_at": metadata.created_at.isoformat(),
            "metrics": metadata.metrics,
            "is_production": metadata.is_production,
            "tags": metadata.tags
        }
        
        registry["versions"].append(version_info)
        
        # Save registry
        with open(registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def _remove_from_version_registry(self, version: str) -> None:
        """Remove a version from the registry."""
        registry_file = self.metadata_path / "versions.json"
        
        if not registry_file.exists():
            return
        
        with open(registry_file, "r") as f:
            registry = json.load(f)
        
        # Remove version
        registry["versions"] = [
            v for v in registry["versions"] if v["version"] != version
        ]
        
        # Save registry
        with open(registry_file, "w") as f:
            json.dump(registry, f, indent=2)

    def _update_production_status(self, version: str, is_production: bool) -> bool:
        """Update production status for a model version."""
        metadata = self.get_model_metadata(version)
        if not metadata:
            return False
        
        metadata.is_production = is_production
        
        # Save updated metadata
        metadata_file = self.models_path / version / "metadata.json"
        with open(metadata_file, "w") as f:
            json.dump(asdict(metadata), f, indent=2, default=str)
        
        return True

    def _find_config_differences(self, config1: Dict[str, Any],
                               config2: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Find differences between two configurations."""
        differences = []
        
        all_keys = set(config1.keys()) | set(config2.keys())
        
        for key in all_keys:
            val1 = config1.get(key)
            val2 = config2.get(key)
            
            if val1 != val2:
                differences.append({
                    "key": key,
                    "value1": val1,
                    "value2": val2
                })
        
        return differences

    def _save_experiment_run(self, experiment_run: ExperimentRun) -> None:
        """Save experiment run to disk."""
        run_path = self.experiments_path / experiment_run.id
        metadata_file = run_path / "metadata.json"
        
        with open(metadata_file, "w") as f:
            json.dump(asdict(experiment_run), f, indent=2, default=str)

    def _load_experiment_run(self, run_id: str) -> Optional[ExperimentRun]:
        """Load experiment run from disk."""
        run_path = self.experiments_path / run_id
        metadata_file = run_path / "metadata.json"
        
        if not metadata_file.exists():
            return None
        
        with open(metadata_file, "r") as f:
            data = json.load(f)
        
        # Convert datetime string back to datetime object
        data["created_at"] = datetime.fromisoformat(data["created_at"])
        
        return ExperimentRun(**data)

    def get_version_statistics(self) -> Dict[str, Any]:
        """Get statistics about model versions.
        
        Returns:
            Dictionary containing version statistics
        """
        versions = self.list_model_versions(limit=1000)
        
        if not versions:
            return {"total_versions": 0}
        
        # Calculate statistics
        total_versions = len(versions)
        production_versions = sum(1 for v in versions if v.is_production)
        
        # Metric trends
        metrics_over_time = []
        for version in sorted(versions, key=lambda x: x.created_at):
            metrics_over_time.append({
                "version": version.version,
                "created_at": version.created_at.isoformat(),
                "metrics": version.metrics
            })
        
        # Average model size
        avg_model_size = sum(v.model_size_mb for v in versions) / total_versions
        
        return {
            "total_versions": total_versions,
            "production_versions": production_versions,
            "average_model_size_mb": avg_model_size,
            "metrics_over_time": metrics_over_time[-20:],  # Last 20 versions
            "latest_version": versions[0].version if versions else None
        }