"""
FastAPI server for OhanaAI MLOps API.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..core.config import OhanaConfig, load_config
from ..mlops.database import DatabaseManager
from ..mlops.pipeline import TrainingPipeline, PipelineConfig
from ..mlops.storage import StorageManager
from ..mlops.versioning import ModelVersionManager
from ..mlops.monitoring import MetricsCollector, AlertManager, PipelineMonitor
from ..mlops.triggers import TriggerManager, UploadTrigger, TrainingTrigger, WebhookTrigger
from .models import *
from .auth import APIAuth, require_auth


class OhanaAPIServer:
    """OhanaAI API Server with MLOps pipeline integration."""
    
    def __init__(self, config_path: str = "config.yaml"):
        """Initialize API server.
        
        Args:
            config_path: Path to configuration file
        """
        # Load configuration
        self.config = load_config(config_path)
        
        # Initialize components
        self.db_manager = DatabaseManager("ohana_mlops.db")
        self.storage_manager = StorageManager("local", base_path="storage")
        self.version_manager = ModelVersionManager("experiments")
        
        # Initialize monitoring
        self.metrics_collector = MetricsCollector()
        self.alert_manager = AlertManager(self.metrics_collector)
        self.monitor = PipelineMonitor(self.metrics_collector, self.alert_manager)
        
        # Initialize triggers
        self.trigger_manager = TriggerManager()
        self.upload_trigger = UploadTrigger(self.trigger_manager, Path("uploads"))
        self.training_trigger = TrainingTrigger(self.trigger_manager)
        
        # Initialize pipeline
        pipeline_config = PipelineConfig()
        self.pipeline = TrainingPipeline(self.config, pipeline_config, self.db_manager)
        
        # Setup authentication
        self.auth = APIAuth()
        
        # Logger
        self.logger = logging.getLogger(__name__)

    async def startup(self) -> None:
        """Server startup tasks."""
        self.logger.info("Starting OhanaAI API server...")
        
        # Start monitoring
        await self.monitor.start_monitoring()
        
        # Start pipeline
        asyncio.create_task(self.pipeline.start())
        
        # Start upload monitoring
        asyncio.create_task(self.upload_trigger.start_monitoring())
        
        # Register event callbacks
        self.trigger_manager.register_callback("gedcom_upload", self._handle_upload_event)
        self.trigger_manager.register_callback("training_trigger", self._handle_training_event)
        
        self.logger.info("API server started successfully")

    async def shutdown(self) -> None:
        """Server shutdown tasks."""
        self.logger.info("Shutting down OhanaAI API server...")
        
        # Stop components
        await self.monitor.stop_monitoring()
        await self.pipeline.stop()
        await self.upload_trigger.stop_monitoring()
        
        self.logger.info("API server shutdown complete")

    async def _handle_upload_event(self, event) -> None:
        """Handle GEDCOM upload event."""
        file_path = Path(event.data["file_path"])
        await self.pipeline.add_gedcom_file(file_path, event.data["filename"])

    async def _handle_training_event(self, event) -> None:
        """Handle training trigger event."""
        self.logger.info(f"Training triggered: {event.data.get('trigger_reason', 'unknown')}")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager."""
    # Startup
    await app.state.server.startup()
    yield
    # Shutdown
    await app.state.server.shutdown()


def create_app(config_path: str = "config.yaml") -> FastAPI:
    """Create FastAPI application.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        FastAPI application instance
    """
    # Create server instance
    server = OhanaAPIServer(config_path)
    
    # Create FastAPI app
    app = FastAPI(
        title="OhanaAI MLOps API",
        description="API for continuous GEDCOM processing and model training",
        version="1.0.0",
        lifespan=lifespan
    )
    
    # Store server instance in app state
    app.state.server = server
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],  # Configure appropriately for production
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    # Setup routes
    setup_routes(app, server)
    
    return app


def setup_routes(app: FastAPI, server: OhanaAPIServer) -> None:
    """Setup API routes.
    
    Args:
        app: FastAPI application
        server: Server instance
    """
    
    @app.get("/health")
    async def health_check():
        """Health check endpoint."""
        health_status = server.monitor.get_health_status()
        return JSONResponse(content=health_status)
    
    @app.get("/status")
    async def get_status():
        """Get detailed system status."""
        pipeline_status = await server.pipeline.get_pipeline_status()
        db_stats = server.db_manager.get_database_stats()
        storage_stats = await server.storage_manager.get_storage_stats()
        
        return {
            "timestamp": pipeline_status["running"],
            "pipeline": pipeline_status,
            "database": db_stats,
            "storage": storage_stats
        }
    
    @app.post("/upload")
    async def upload_gedcom(
        background_tasks: BackgroundTasks,
        file: UploadFile = File(...),
        auth_user: dict = Depends(require_auth)
    ):
        """Upload a GEDCOM file for processing."""
        if not file.filename.lower().endswith(('.ged', '.gedcom')):
            raise HTTPException(status_code=400, detail="Only GEDCOM files (.ged, .gedcom) are allowed")
        
        # Save uploaded file
        upload_dir = Path("uploads")
        upload_dir.mkdir(exist_ok=True)
        
        file_path = upload_dir / file.filename
        
        try:
            with open(file_path, "wb") as f:
                content = await file.read()
                f.write(content)
            
            # Add to processing pipeline
            file_id = await server.pipeline.add_gedcom_file(file_path, file.filename)
            
            return {
                "message": "File uploaded successfully",
                "file_id": file_id,
                "filename": file.filename,
                "size": len(content)
            }
            
        except Exception as e:
            server.logger.error(f"Upload error: {e}")
            raise HTTPException(status_code=500, detail=str(e))
    
    @app.get("/files")
    async def list_files(
        status: str = None,
        auth_user: dict = Depends(require_auth)
    ):
        """List GEDCOM files."""
        if status:
            try:
                from ..mlops.database import ProcessingStatus
                status_enum = ProcessingStatus(status)
                files = server.db_manager.get_files_by_status(status_enum)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status")
        else:
            # Get all files (implement method to get all files)
            files = []
        
        return {
            "files": [
                {
                    "id": f.id,
                    "filename": f.original_filename,
                    "status": f.status.value,
                    "upload_time": f.upload_time.isoformat(),
                    "num_individuals": f.num_individuals,
                    "num_families": f.num_families,
                    "processing_time": f.processing_time
                }
                for f in files
            ]
        }
    
    @app.get("/files/{file_id}")
    async def get_file_details(
        file_id: int,
        auth_user: dict = Depends(require_auth)
    ):
        """Get details for a specific file."""
        file_record = server.db_manager.get_gedcom_file(file_id)
        if not file_record:
            raise HTTPException(status_code=404, detail="File not found")
        
        return {
            "id": file_record.id,
            "filename": file_record.original_filename,
            "status": file_record.status.value,
            "upload_time": file_record.upload_time.isoformat(),
            "file_size": file_record.file_size,
            "num_individuals": file_record.num_individuals,
            "num_families": file_record.num_families,
            "date_range": file_record.date_range,
            "processing_time": file_record.processing_time,
            "error_message": file_record.error_message,
            "metadata": file_record.metadata
        }
    
    @app.post("/training/start")
    async def start_training(
        request: TrainingRequest,
        auth_user: dict = Depends(require_auth)
    ):
        """Start a training run."""
        # Queue training with specified files
        run_id = await server.pipeline._queue_training_run(
            request.gedcom_ids, "manual"
        )
        
        return {
            "message": "Training started",
            "run_id": run_id
        }
    
    @app.get("/training/runs")
    async def list_training_runs(
        status: str = None,
        limit: int = 50,
        auth_user: dict = Depends(require_auth)
    ):
        """List training runs."""
        if status:
            try:
                from ..mlops.database import TrainingStatus
                status_enum = TrainingStatus(status)
                runs = server.db_manager.get_training_runs(status_enum, limit)
            except ValueError:
                raise HTTPException(status_code=400, detail="Invalid status")
        else:
            runs = server.db_manager.get_training_runs(limit=limit)
        
        return {
            "runs": [
                {
                    "id": run.id,
                    "run_name": run.run_name,
                    "status": run.status.value,
                    "start_time": run.start_time.isoformat(),
                    "end_time": run.end_time.isoformat() if run.end_time else None,
                    "duration": run.duration,
                    "final_metrics": run.final_metrics,
                    "model_version": run.model_version,
                    "triggered_by": run.triggered_by
                }
                for run in runs
            ]
        }
    
    @app.get("/training/runs/{run_id}")
    async def get_training_run(
        run_id: int,
        auth_user: dict = Depends(require_auth)
    ):
        """Get details for a specific training run."""
        runs = server.db_manager.get_training_runs(limit=1000)
        run = None
        for r in runs:
            if r.id == run_id:
                run = r
                break
        
        if not run:
            raise HTTPException(status_code=404, detail="Training run not found")
        
        return {
            "id": run.id,
            "run_name": run.run_name,
            "status": run.status.value,
            "config": run.config,
            "gedcom_ids": run.gedcom_ids,
            "start_time": run.start_time.isoformat(),
            "end_time": run.end_time.isoformat() if run.end_time else None,
            "duration": run.duration,
            "final_metrics": run.final_metrics,
            "model_path": run.model_path,
            "model_version": run.model_version,
            "error_message": run.error_message,
            "triggered_by": run.triggered_by
        }
    
    @app.get("/models")
    async def list_models(
        limit: int = 20,
        auth_user: dict = Depends(require_auth)
    ):
        """List model versions."""
        models = server.version_manager.list_model_versions(limit)
        
        return {
            "models": [
                {
                    "version": model.version,
                    "created_at": model.created_at.isoformat(),
                    "metrics": model.metrics,
                    "model_size_mb": model.model_size_mb,
                    "training_duration_seconds": model.training_duration_seconds,
                    "is_production": model.is_production,
                    "tags": model.tags,
                    "notes": model.notes
                }
                for model in models
            ]
        }
    
    @app.get("/models/{version}")
    async def get_model_details(
        version: str,
        auth_user: dict = Depends(require_auth)
    ):
        """Get details for a specific model version."""
        model = server.version_manager.get_model_metadata(version)
        if not model:
            raise HTTPException(status_code=404, detail="Model version not found")
        
        return {
            "version": model.version,
            "created_at": model.created_at.isoformat(),
            "config": model.config,
            "metrics": model.metrics,
            "training_data": model.training_data,
            "model_size_mb": model.model_size_mb,
            "training_duration_seconds": model.training_duration_seconds,
            "parent_version": model.parent_version,
            "tags": model.tags,
            "notes": model.notes,
            "is_production": model.is_production
        }
    
    @app.post("/models/{version}/promote")
    async def promote_model(
        version: str,
        auth_user: dict = Depends(require_auth)
    ):
        """Promote a model version to production."""
        success = server.version_manager.promote_to_production(version)
        if not success:
            raise HTTPException(status_code=400, detail="Failed to promote model")
        
        return {"message": f"Model {version} promoted to production"}
    
    @app.get("/metrics")
    async def get_metrics(
        minutes: int = 60,
        auth_user: dict = Depends(require_auth)
    ):
        """Get system metrics."""
        system_metrics = server.metrics_collector.get_system_metrics(minutes)
        
        return {
            "metrics": [
                {
                    "timestamp": m.timestamp.isoformat(),
                    "cpu_percent": m.cpu_percent,
                    "memory_percent": m.memory_percent,
                    "memory_used_gb": m.memory_used_gb,
                    "disk_usage_percent": m.disk_usage_percent,
                    "disk_free_gb": m.disk_free_gb,
                    "network_sent_mb": m.network_sent_mb,
                    "network_recv_mb": m.network_recv_mb
                }
                for m in system_metrics
            ]
        }
    
    @app.get("/alerts")
    async def get_alerts(
        auth_user: dict = Depends(require_auth)
    ):
        """Get active alerts."""
        active_alerts = server.alert_manager.get_active_alerts()
        
        return {
            "alerts": [
                {
                    "rule_name": alert.rule_name,
                    "metric": alert.metric,
                    "current_value": alert.current_value,
                    "threshold": alert.threshold,
                    "severity": alert.severity,
                    "triggered_at": alert.triggered_at.isoformat(),
                    "message": alert.message
                }
                for alert in active_alerts
            ]
        }


async def run_server(config_path: str = "config.yaml", host: str = "0.0.0.0", port: int = 8000):
    """Run the API server.
    
    Args:
        config_path: Path to configuration file
        host: Host to bind to
        port: Port to bind to
    """
    app = create_app(config_path)
    
    config = uvicorn.Config(
        app,
        host=host,
        port=port,
        log_level="info",
        access_log=True
    )
    
    server = uvicorn.Server(config)
    await server.serve()


if __name__ == "__main__":
    import sys
    
    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.yaml"
    asyncio.run(run_server(config_path))