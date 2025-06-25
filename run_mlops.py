#!/usr/bin/env python3
"""
OhanaAI MLOps Server Launcher
Starts the complete MLOps pipeline including API server, monitoring, and training automation.
"""

import asyncio
import logging
import signal
import sys
from pathlib import Path
from typing import List, Optional

import click
import yaml

from ohana_ai.api.server import run_server
from ohana_ai.core.config import setup_logging
from ohana_ai.mlops.database import DatabaseManager
from ohana_ai.mlops.pipeline import TrainingPipeline, PipelineConfig
from ohana_ai.mlops.storage import StorageManager
from ohana_ai.mlops.monitoring import setup_monitoring
from ohana_ai.mlops.triggers import TriggerManager, UploadTrigger, TrainingTrigger, WebhookTrigger, NotificationTrigger


class MLOpsServer:
    """Complete MLOps server for OhanaAI."""
    
    def __init__(self, config_path: str = "mlops_config.yaml"):
        """Initialize MLOps server.
        
        Args:
            config_path: Path to MLOps configuration file
        """
        self.config_path = Path(config_path)
        self.config = self._load_config()
        self.logger = self._setup_logging()
        
        # Initialize components
        self.components = {}
        self._running = False
        
        # Signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _load_config(self) -> dict:
        """Load MLOps configuration."""
        if not self.config_path.exists():
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            return yaml.safe_load(f)

    def _setup_logging(self) -> logging.Logger:
        """Setup logging configuration."""
        log_config = self.config.get("logging", {})
        
        # Create logs directory
        log_file = Path(log_config.get("file", "logs/mlops.log"))
        log_file.parent.mkdir(exist_ok=True)
        
        # Configure logging
        logging.basicConfig(
            level=getattr(logging, log_config.get("level", "INFO")),
            format=log_config.get("format", "%(asctime)s - %(name)s - %(levelname)s - %(message)s"),
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(log_file)
            ]
        )
        
        # Set component-specific levels
        for component, level in log_config.get("levels", {}).items():
            logging.getLogger(component).setLevel(getattr(logging, level))
        
        return logging.getLogger(__name__)

    def _signal_handler(self, signum, frame):
        """Handle shutdown signals."""
        self.logger.info(f"Received signal {signum}, shutting down...")
        self._running = False

    async def start(self) -> None:
        """Start the MLOps server."""
        try:
            self.logger.info("Starting OhanaAI MLOps Server")
            self._running = True
            
            # Initialize database
            await self._init_database()
            
            # Initialize storage
            await self._init_storage()
            
            # Initialize monitoring
            await self._init_monitoring()
            
            # Initialize triggers
            await self._init_triggers()
            
            # Initialize training pipeline
            await self._init_pipeline()
            
            # Start API server
            await self._start_api_server()
            
        except Exception as e:
            self.logger.error(f"Failed to start MLOps server: {e}", exc_info=True)
            await self.stop()
            raise

    async def stop(self) -> None:
        """Stop the MLOps server."""
        self.logger.info("Stopping OhanaAI MLOps Server")
        
        # Stop components in reverse order
        for component_name in reversed(list(self.components.keys())):
            try:
                component = self.components[component_name]
                if hasattr(component, 'stop'):
                    await component.stop()
                self.logger.info(f"Stopped {component_name}")
            except Exception as e:
                self.logger.error(f"Error stopping {component_name}: {e}")
        
        self.logger.info("MLOps server shutdown complete")

    async def _init_database(self) -> None:
        """Initialize database manager."""
        db_config = self.config.get("database", {})
        db_manager = DatabaseManager(db_config.get("path", "ohana_mlops.db"))
        self.components["database"] = db_manager
        self.logger.info("Database manager initialized")

    async def _init_storage(self) -> None:
        """Initialize storage manager."""
        storage_config = self.config.get("storage", {})
        backend = storage_config.get("backend", "local")
        
        if backend == "local":
            backend_config = storage_config.get("local", {})
        elif backend == "s3":
            backend_config = storage_config.get("s3", {})
        elif backend == "gcs":
            backend_config = storage_config.get("gcs", {})
        else:
            raise ValueError(f"Unsupported storage backend: {backend}")
        
        storage_manager = StorageManager(backend, **backend_config)
        self.components["storage"] = storage_manager
        self.logger.info(f"Storage manager initialized ({backend})")

    async def _init_monitoring(self) -> None:
        """Initialize monitoring system."""
        monitor_config = self.config.get("monitoring", {})
        
        monitor = await setup_monitoring()
        self.components["monitor"] = monitor
        
        # Add custom alert rules from config
        for alert_config in monitor_config.get("alerts", []):
            from ohana_ai.mlops.monitoring import AlertRule
            rule = AlertRule(**alert_config)
            monitor.alert_manager.add_alert_rule(rule)
        
        self.logger.info("Monitoring system initialized")

    async def _init_triggers(self) -> None:
        """Initialize trigger system."""
        trigger_config = self.config.get("webhooks", {})
        upload_config = self.config.get("uploads", {})
        
        # Main trigger manager
        trigger_manager = TriggerManager()
        self.components["triggers"] = trigger_manager
        
        # Upload trigger
        upload_trigger = UploadTrigger(
            trigger_manager, 
            Path(upload_config.get("directory", "uploads"))
        )
        self.components["upload_trigger"] = upload_trigger
        
        # Training trigger
        training_trigger = TrainingTrigger(trigger_manager)
        self.components["training_trigger"] = training_trigger
        
        # Webhook trigger (if enabled)
        if trigger_config.get("enabled", False):
            webhook_trigger = WebhookTrigger(
                trigger_manager,
                trigger_config.get("host", "0.0.0.0"),
                trigger_config.get("port", 8080)
            )
            self.components["webhook_trigger"] = webhook_trigger
            await webhook_trigger.start_server()
        
        # Notification trigger
        notification_trigger = NotificationTrigger(trigger_manager)
        for notification in trigger_config.get("notifications", []):
            notification_trigger.add_notification_url(notification["url"])
        
        self.components["notification_trigger"] = notification_trigger
        
        # Register notification callback
        trigger_manager.register_callback("*", notification_trigger.send_notification)
        
        self.logger.info("Trigger system initialized")

    async def _init_pipeline(self) -> None:
        """Initialize training pipeline."""
        pipeline_config_dict = self.config.get("pipeline", {})
        
        # Create pipeline configuration
        pipeline_config = PipelineConfig(**pipeline_config_dict)
        
        # Initialize pipeline with dependencies
        from ohana_ai.core.config import load_config
        ohana_config = load_config("config.yaml")
        
        pipeline = TrainingPipeline(
            ohana_config,
            pipeline_config,
            self.components["database"]
        )
        
        self.components["pipeline"] = pipeline
        
        # Start pipeline
        asyncio.create_task(pipeline.start())
        
        # Register pipeline event callbacks
        trigger_manager = self.components["triggers"]
        trigger_manager.register_callback("gedcom_upload", self._handle_upload)
        trigger_manager.register_callback("training_trigger", self._handle_training)
        
        self.logger.info("Training pipeline initialized")

    async def _start_api_server(self) -> None:
        """Start the API server."""
        api_config = self.config.get("api", {})
        
        # Start API server
        await run_server(
            config_path="config.yaml",
            host=api_config.get("host", "0.0.0.0"),
            port=api_config.get("port", 8000)
        )

    async def _handle_upload(self, event) -> None:
        """Handle file upload event."""
        pipeline = self.components["pipeline"]
        file_path = Path(event.data["file_path"])
        filename = event.data["filename"]
        
        await pipeline.add_gedcom_file(file_path, filename)

    async def _handle_training(self, event) -> None:
        """Handle training trigger event."""
        self.logger.info(f"Training triggered: {event.data}")

    async def health_check(self) -> dict:
        """Perform health check on all components.
        
        Returns:
            Health status dictionary
        """
        health = {
            "status": "healthy",
            "timestamp": asyncio.get_event_loop().time(),
            "components": {}
        }
        
        # Check each component
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_health_status'):
                    component_health = await component.get_health_status()
                elif hasattr(component, 'get_status'):
                    component_health = await component.get_status()
                else:
                    component_health = {"status": "running"}
                
                health["components"][name] = component_health
                
            except Exception as e:
                health["components"][name] = {
                    "status": "error",
                    "error": str(e)
                }
                health["status"] = "degraded"
        
        return health

    async def run_forever(self) -> None:
        """Run the server until stopped."""
        while self._running:
            try:
                await asyncio.sleep(1)
            except KeyboardInterrupt:
                break
        
        await self.stop()


@click.command()
@click.option('--config', '-c', default='mlops_config.yaml', help='Configuration file path')
@click.option('--host', default='0.0.0.0', help='API server host')
@click.option('--port', default=8000, type=int, help='API server port')
@click.option('--debug', is_flag=True, help='Enable debug mode')
def main(config: str, host: str, port: int, debug: bool):
    """Start the OhanaAI MLOps server."""
    
    # Override config with CLI options
    if Path(config).exists():
        with open(config, 'r') as f:
            config_data = yaml.safe_load(f)
    else:
        config_data = {}
    
    if 'api' not in config_data:
        config_data['api'] = {}
    
    config_data['api']['host'] = host
    config_data['api']['port'] = port
    config_data['api']['debug'] = debug
    
    # Write updated config
    with open(config, 'w') as f:
        yaml.safe_dump(config_data, f, indent=2)
    
    # Start server
    server = MLOpsServer(config)
    
    try:
        asyncio.run(server.start())
    except KeyboardInterrupt:
        print("\nShutdown initiated by user")
    except Exception as e:
        print(f"Server error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()