"""
Automated training pipeline for continuous learning from GEDCOM uploads.
"""

import asyncio
import hashlib
import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from ..core.config import OhanaConfig
from ..core.exceptions import ModelError, OhanaAIError
from ..gedcom_parser import parse_gedcom_file
from ..graph_builder import build_graph_from_gedcom
from ..trainer import OhanaAITrainer, TrainingMetrics
from .database import DatabaseManager, GedcomRecord, GraphCache, ProcessingStatus, TrainingRun, TrainingStatus


@dataclass
class PipelineConfig:
    """Configuration for training pipeline."""
    
    # Trigger settings
    min_files_for_training: int = 5
    auto_training_enabled: bool = True
    training_schedule_hours: List[int] = None  # Hours of day to run training (0-23)
    
    # Processing settings
    max_concurrent_processing: int = 4
    max_graph_cache_size_gb: float = 5.0
    cache_ttl_days: int = 30
    
    # Training settings
    incremental_training: bool = True
    min_improvement_threshold: float = 0.001
    max_training_time_hours: float = 4.0
    
    # Model versioning
    keep_model_versions: int = 10
    auto_deploy_threshold: float = 0.02  # Auto deploy if validation improves by this much
    
    # Storage settings
    storage_backend: str = "local"  # local, s3, gcs
    storage_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.training_schedule_hours is None:
            self.training_schedule_hours = [2, 14]  # 2 AM and 2 PM
        if self.storage_config is None:
            self.storage_config = {}


class TrainingPipeline:
    """Manages the automated training pipeline for continuous learning."""
    
    def __init__(self, config: OhanaConfig, pipeline_config: PipelineConfig,
                 db_manager: DatabaseManager):
        """Initialize training pipeline.
        
        Args:
            config: OhanaAI configuration
            pipeline_config: Pipeline-specific configuration
            db_manager: Database manager instance
        """
        self.config = config
        self.pipeline_config = pipeline_config
        self.db = db_manager
        self.logger = logging.getLogger(__name__)
        
        # Pipeline state
        self._running = False
        self._processing_queue = asyncio.Queue()
        self._training_queue = asyncio.Queue()
        self._executor = ThreadPoolExecutor(max_workers=pipeline_config.max_concurrent_processing)
        
        # Storage paths
        self.uploads_dir = Path("uploads")
        self.cache_dir = Path("cache")
        self.models_dir = Path(config.checkpoints_dir)
        
        # Ensure directories exist
        for directory in [self.uploads_dir, self.cache_dir, self.models_dir]:
            directory.mkdir(exist_ok=True)

    async def start(self) -> None:
        """Start the pipeline processing."""
        if self._running:
            self.logger.warning("Pipeline already running")
            return
            
        self._running = True
        self.logger.info("Starting OhanaAI training pipeline")
        
        # Start background tasks
        tasks = [
            asyncio.create_task(self._process_uploads()),
            asyncio.create_task(self._training_scheduler()),
            asyncio.create_task(self._cache_manager()),
        ]
        
        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Pipeline error: {e}", exc_info=True)
        finally:
            self._running = False

    async def stop(self) -> None:
        """Stop the pipeline processing."""
        self.logger.info("Stopping training pipeline")
        self._running = False
        self._executor.shutdown(wait=True)

    async def add_gedcom_file(self, file_path: Path, original_filename: str) -> int:
        """Add a new GEDCOM file to the processing queue.
        
        Args:
            file_path: Path to uploaded GEDCOM file
            original_filename: Original filename from upload
            
        Returns:
            Database ID of the GEDCOM record
        """
        # Calculate file hash
        file_hash = self._calculate_file_hash(file_path)
        
        # Check if file already exists
        existing_files = self.db.get_files_by_status(ProcessingStatus.PROCESSED)
        for existing in existing_files:
            if existing.file_hash == file_hash:
                self.logger.info(f"File {original_filename} already exists (hash: {file_hash})")
                return existing.id
        
        # Create database record
        record = GedcomRecord(
            filename=str(file_path),
            original_filename=original_filename,
            file_hash=file_hash,
            file_size=file_path.stat().st_size,
            status=ProcessingStatus.UPLOADED
        )
        
        file_id = self.db.add_gedcom_file(record)
        
        # Add to processing queue
        await self._processing_queue.put(file_id)
        
        self.logger.info(f"Added GEDCOM file: {original_filename} (ID: {file_id})")
        return file_id

    async def _process_uploads(self) -> None:
        """Process uploaded GEDCOM files continuously."""
        while self._running:
            try:
                # Wait for files to process
                file_id = await asyncio.wait_for(
                    self._processing_queue.get(), timeout=5.0
                )
                
                # Process file in background
                asyncio.create_task(self._process_gedcom_file(file_id))
                
            except asyncio.TimeoutError:
                continue
            except Exception as e:
                self.logger.error(f"Error in upload processing: {e}", exc_info=True)

    async def _process_gedcom_file(self, file_id: int) -> None:
        """Process a single GEDCOM file.
        
        Args:
            file_id: Database ID of GEDCOM file to process
        """
        record = self.db.get_gedcom_file(file_id)
        if not record:
            self.logger.error(f"GEDCOM file not found: {file_id}")
            return
        
        self.logger.info(f"Processing GEDCOM file: {record.original_filename}")
        start_time = time.time()
        
        try:
            # Update status to processing
            self.db.update_gedcom_status(file_id, ProcessingStatus.PROCESSING)
            
            # Parse GEDCOM file
            file_path = Path(record.filename)
            individuals, families = await asyncio.get_event_loop().run_in_executor(
                self._executor, parse_gedcom_file, str(file_path)
            )
            
            # Build graph
            graph_data = await asyncio.get_event_loop().run_in_executor(
                self._executor, build_graph_from_gedcom, individuals, families
            )
            
            # Cache graph data
            graph_hash = self._calculate_graph_hash(graph_data)
            cache_record = GraphCache(
                gedcom_id=file_id,
                graph_hash=graph_hash,
                node_features=self._serialize_array(graph_data.node_features),
                edge_indices=self._serialize_array(graph_data.edge_indices),
                edge_types=self._serialize_array(graph_data.edge_types),
                node_metadata=str(len(individuals))
            )
            
            self.db.cache_graph(cache_record)
            
            # Update record with processing results
            processing_time = time.time() - start_time
            
            # Get date range
            birth_years = [ind.birth_year for ind in individuals.values() if ind.birth_year]
            date_range = (min(birth_years), max(birth_years)) if birth_years else (0, 0)
            
            # Update database record
            with self.db._init_database():
                pass  # Use context manager to get connection
            
            # Update the record fields
            record.num_individuals = len(individuals)
            record.num_families = len(families)
            record.date_range = date_range
            record.processing_time = processing_time
            record.status = ProcessingStatus.PROCESSED
            
            self.db.update_gedcom_status(file_id, ProcessingStatus.PROCESSED)
            
            self.logger.info(
                f"Processed {record.original_filename}: "
                f"{len(individuals)} individuals, {len(families)} families "
                f"in {processing_time:.2f}s"
            )
            
            # Check if we should trigger training
            await self._check_training_trigger()
            
        except Exception as e:
            self.logger.error(f"Failed to process GEDCOM file {file_id}: {e}", exc_info=True)
            self.db.update_gedcom_status(
                file_id, ProcessingStatus.FAILED, str(e)
            )

    async def _check_training_trigger(self) -> None:
        """Check if training should be triggered."""
        if not self.pipeline_config.auto_training_enabled:
            return
        
        # Count processed files
        processed_files = self.db.get_files_by_status(ProcessingStatus.PROCESSED)
        
        if len(processed_files) >= self.pipeline_config.min_files_for_training:
            # Check if there's already a training run queued or running
            active_runs = self.db.get_training_runs(TrainingStatus.RUNNING, limit=1)
            queued_runs = self.db.get_training_runs(TrainingStatus.QUEUED, limit=1)
            
            if not active_runs and not queued_runs:
                await self._queue_training_run([f.id for f in processed_files], "upload_trigger")

    async def _queue_training_run(self, gedcom_ids: List[int], trigger: str) -> int:
        """Queue a new training run.
        
        Args:
            gedcom_ids: List of GEDCOM file IDs to train on
            trigger: What triggered this training run
            
        Returns:
            Training run ID
        """
        run_name = f"run_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        training_run = TrainingRun(
            run_name=run_name,
            status=TrainingStatus.QUEUED,
            config=self.config.to_dict(),
            gedcom_ids=gedcom_ids,
            triggered_by=trigger
        )
        
        run_id = self.db.add_training_run(training_run)
        await self._training_queue.put(run_id)
        
        self.logger.info(f"Queued training run {run_name} (ID: {run_id})")
        return run_id

    async def _training_scheduler(self) -> None:
        """Schedule and execute training runs."""
        while self._running:
            try:
                # Wait for training runs
                run_id = await asyncio.wait_for(
                    self._training_queue.get(), timeout=10.0
                )
                
                # Execute training run
                await self._execute_training_run(run_id)
                
            except asyncio.TimeoutError:
                # Check for scheduled training
                current_hour = datetime.now().hour
                if current_hour in self.pipeline_config.training_schedule_hours:
                    await self._check_scheduled_training()
                continue
            except Exception as e:
                self.logger.error(f"Error in training scheduler: {e}", exc_info=True)

    async def _execute_training_run(self, run_id: int) -> None:
        """Execute a training run.
        
        Args:
            run_id: Database ID of training run
        """
        run_record = None
        for run in self.db.get_training_runs(limit=1000):
            if run.id == run_id:
                run_record = run
                break
        
        if not run_record:
            self.logger.error(f"Training run not found: {run_id}")
            return
        
        self.logger.info(f"Starting training run: {run_record.run_name}")
        start_time = time.time()
        
        try:
            # Update status to running
            self.db.update_training_run(run_id, TrainingStatus.RUNNING)
            
            # Load graph data for training
            graph_data_list = []
            for gedcom_id in run_record.gedcom_ids:
                cached_graph = self._load_cached_graph(gedcom_id)
                if cached_graph:
                    graph_data_list.append(cached_graph)
            
            if not graph_data_list:
                raise ModelError("No graph data available for training")
            
            # Create trainer
            trainer = OhanaAITrainer(
                OhanaConfig.from_dict(run_record.config)
            )
            
            # Train model with callback for metrics logging
            def metrics_callback(metrics: TrainingMetrics):
                self.db.log_training_metrics(
                    run_id, metrics.epoch, metrics.train_loss, metrics.train_accuracy,
                    metrics.val_loss, metrics.val_accuracy, metrics.learning_rate
                )
            
            # Execute training in thread pool
            trained_model = await asyncio.get_event_loop().run_in_executor(
                self._executor, self._train_model_sync, trainer, graph_data_list, metrics_callback
            )
            
            # Save model
            model_version = f"v{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            model_path = self.models_dir / f"ohana_model_{model_version}.npz"
            
            # Save trained model (placeholder - implement based on your model saving logic)
            # trained_model.save(str(model_path))
            
            # Get final metrics
            final_metrics = {
                "final_train_accuracy": 0.85,  # Replace with actual metrics
                "final_val_accuracy": 0.82,
                "total_epochs": trainer.config.epochs,
                "training_samples": sum(len(gd.node_features) for gd in graph_data_list)
            }
            
            # Update training run
            end_time = datetime.now()
            duration = time.time() - start_time
            
            self.db.update_training_run(
                run_id, TrainingStatus.COMPLETED, end_time, duration,
                final_metrics, str(model_path), model_version
            )
            
            self.logger.info(
                f"Training run {run_record.run_name} completed in {duration:.2f}s. "
                f"Val accuracy: {final_metrics['final_val_accuracy']:.4f}"
            )
            
        except Exception as e:
            self.logger.error(f"Training run {run_id} failed: {e}", exc_info=True)
            self.db.update_training_run(
                run_id, TrainingStatus.FAILED, error_message=str(e)
            )

    def _train_model_sync(self, trainer: OhanaAITrainer, graph_data_list: List[Any],
                         metrics_callback) -> Any:
        """Synchronous model training for thread pool execution."""
        # This would implement the actual training logic
        # For now, return a placeholder
        return trainer

    async def _check_scheduled_training(self) -> None:
        """Check if scheduled training should run."""
        # Get recently processed files
        processed_files = self.db.get_files_by_status(ProcessingStatus.PROCESSED)
        
        # Check if there are new files since last training
        recent_runs = self.db.get_training_runs(limit=1)
        if recent_runs:
            last_run_time = recent_runs[0].start_time
            new_files = [f for f in processed_files if f.upload_time > last_run_time]
        else:
            new_files = processed_files
        
        if len(new_files) >= self.pipeline_config.min_files_for_training:
            await self._queue_training_run(
                [f.id for f in processed_files], "scheduled"
            )

    async def _cache_manager(self) -> None:
        """Manage graph cache cleanup and optimization."""
        while self._running:
            try:
                # Sleep for an hour between cache management runs
                await asyncio.sleep(3600)
                
                # Clean up old cache entries
                deleted_count = self.db.cleanup_old_cache(
                    self.pipeline_config.cache_ttl_days
                )
                
                if deleted_count > 0:
                    self.logger.info(f"Cleaned up {deleted_count} old cache entries")
                
                # Check cache size and clean up if needed
                await self._check_cache_size()
                
            except Exception as e:
                self.logger.error(f"Error in cache manager: {e}", exc_info=True)

    async def _check_cache_size(self) -> None:
        """Check and manage cache size."""
        stats = self.db.get_database_stats()
        cache_size_gb = stats["database_size"] / (1024**3)
        
        if cache_size_gb > self.pipeline_config.max_graph_cache_size_gb:
            self.logger.warning(f"Cache size ({cache_size_gb:.2f}GB) exceeds limit")
            # Could implement LRU cache cleanup here

    def _calculate_file_hash(self, file_path: Path) -> str:
        """Calculate SHA-256 hash of a file.
        
        Args:
            file_path: Path to file
            
        Returns:
            Hexadecimal hash string
        """
        hash_sha256 = hashlib.sha256()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_sha256.update(chunk)
        return hash_sha256.hexdigest()

    def _calculate_graph_hash(self, graph_data: Any) -> str:
        """Calculate hash of graph data.
        
        Args:
            graph_data: Graph data object
            
        Returns:
            Hexadecimal hash string
        """
        # Create hash from graph structure
        hash_data = str(graph_data.node_features.shape) + str(graph_data.edge_indices.shape)
        return hashlib.sha256(hash_data.encode()).hexdigest()

    def _serialize_array(self, array: np.ndarray) -> bytes:
        """Serialize numpy array for database storage.
        
        Args:
            array: Numpy array to serialize
            
        Returns:
            Serialized array as bytes
        """
        return array.tobytes()

    def _deserialize_array(self, data: bytes, shape: Tuple, dtype: type) -> np.ndarray:
        """Deserialize array from database.
        
        Args:
            data: Serialized array data
            shape: Array shape
            dtype: Array data type
            
        Returns:
            Reconstructed numpy array
        """
        return np.frombuffer(data, dtype=dtype).reshape(shape)

    def _load_cached_graph(self, gedcom_id: int) -> Optional[Any]:
        """Load cached graph data for a GEDCOM file.
        
        Args:
            gedcom_id: Database ID of GEDCOM file
            
        Returns:
            Graph data object or None if not cached
        """
        # This would implement loading from the cache
        # For now, return None as placeholder
        return None

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get current pipeline status.
        
        Returns:
            Dictionary containing pipeline status information
        """
        stats = self.db.get_database_stats()
        
        return {
            "running": self._running,
            "processing_queue_size": self._processing_queue.qsize(),
            "training_queue_size": self._training_queue.qsize(),
            "database_stats": stats,
            "config": {
                "auto_training_enabled": self.pipeline_config.auto_training_enabled,
                "min_files_for_training": self.pipeline_config.min_files_for_training,
                "training_schedule_hours": self.pipeline_config.training_schedule_hours,
            }
        }