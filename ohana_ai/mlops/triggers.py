"""
Event triggers and webhooks for the MLOps pipeline.
"""

import asyncio
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

import aiohttp
from aiohttp import web


@dataclass
class TriggerEvent:
    """Represents a trigger event."""
    
    event_type: str  # upload, training_complete, model_deploy, etc.
    timestamp: datetime
    data: Dict[str, Any]
    source: str = "system"
    event_id: str = ""
    
    def __post_init__(self):
        if not self.event_id:
            # Generate unique event ID
            event_str = f"{self.event_type}_{self.timestamp.isoformat()}_{self.source}"
            self.event_id = hashlib.md5(event_str.encode()).hexdigest()[:16]


class TriggerManager:
    """Manages event triggers and callbacks."""
    
    def __init__(self):
        """Initialize trigger manager."""
        self.callbacks: Dict[str, List[Callable]] = {}
        self.event_history: List[TriggerEvent] = []
        self.max_history = 1000
        self.logger = logging.getLogger(__name__)

    def register_callback(self, event_type: str, callback: Callable) -> None:
        """Register a callback for an event type.
        
        Args:
            event_type: Type of event to listen for
            callback: Async callback function to execute
        """
        if event_type not in self.callbacks:
            self.callbacks[event_type] = []
        
        self.callbacks[event_type].append(callback)
        self.logger.info(f"Registered callback for event type: {event_type}")

    async def trigger_event(self, event: TriggerEvent) -> None:
        """Trigger an event and execute callbacks.
        
        Args:
            event: Event to trigger
        """
        self.logger.info(f"Triggering event: {event.event_type} (ID: {event.event_id})")
        
        # Add to history
        self.event_history.append(event)
        if len(self.event_history) > self.max_history:
            self.event_history.pop(0)
        
        # Execute callbacks
        if event.event_type in self.callbacks:
            tasks = []
            for callback in self.callbacks[event.event_type]:
                try:
                    tasks.append(asyncio.create_task(callback(event)))
                except Exception as e:
                    self.logger.error(f"Error creating callback task: {e}")
            
            if tasks:
                # Wait for all callbacks to complete
                await asyncio.gather(*tasks, return_exceptions=True)

    def get_event_history(self, event_type: Optional[str] = None,
                         limit: int = 100) -> List[TriggerEvent]:
        """Get event history.
        
        Args:
            event_type: Filter by event type (optional)
            limit: Maximum number of events to return
            
        Returns:
            List of trigger events
        """
        events = self.event_history
        
        if event_type:
            events = [e for e in events if e.event_type == event_type]
        
        # Return most recent events first
        return events[-limit:][::-1]


class UploadTrigger:
    """Handles file upload triggers."""
    
    def __init__(self, trigger_manager: TriggerManager, upload_dir: Path):
        """Initialize upload trigger.
        
        Args:
            trigger_manager: Trigger manager instance
            upload_dir: Directory to monitor for uploads
        """
        self.trigger_manager = trigger_manager
        self.upload_dir = Path(upload_dir)
        self.upload_dir.mkdir(exist_ok=True)
        self.logger = logging.getLogger(__name__)
        
        # Track processed files
        self.processed_files: Dict[str, float] = {}
        self._monitoring = False

    async def start_monitoring(self, check_interval: float = 5.0) -> None:
        """Start monitoring for file uploads.
        
        Args:
            check_interval: Interval between checks in seconds
        """
        if self._monitoring:
            return
        
        self._monitoring = True
        self.logger.info(f"Starting upload monitoring on {self.upload_dir}")
        
        while self._monitoring:
            try:
                await self._check_for_uploads()
                await asyncio.sleep(check_interval)
            except Exception as e:
                self.logger.error(f"Error in upload monitoring: {e}")
                await asyncio.sleep(check_interval)

    async def stop_monitoring(self) -> None:
        """Stop monitoring for uploads."""
        self._monitoring = False
        self.logger.info("Stopped upload monitoring")

    async def _check_for_uploads(self) -> None:
        """Check for new uploads."""
        if not self.upload_dir.exists():
            return
        
        for file_path in self.upload_dir.iterdir():
            if file_path.is_file() and file_path.suffix.lower() in ['.ged', '.gedcom']:
                file_key = str(file_path)
                current_mtime = file_path.stat().st_mtime
                
                # Check if this is a new or modified file
                if (file_key not in self.processed_files or 
                    self.processed_files[file_key] < current_mtime):
                    
                    # Wait a bit to ensure file is fully uploaded
                    await asyncio.sleep(1.0)
                    
                    # Check if file size is stable (not still uploading)
                    if await self._is_upload_complete(file_path):
                        await self._handle_upload(file_path)
                        self.processed_files[file_key] = current_mtime

    async def _is_upload_complete(self, file_path: Path, stability_time: float = 2.0) -> bool:
        """Check if file upload is complete by monitoring size stability.
        
        Args:
            file_path: Path to file being uploaded
            stability_time: Time to wait for size stability
            
        Returns:
            True if upload appears complete
        """
        try:
            initial_size = file_path.stat().st_size
            await asyncio.sleep(stability_time)
            final_size = file_path.stat().st_size
            
            return initial_size == final_size and final_size > 0
        except OSError:
            return False

    async def _handle_upload(self, file_path: Path) -> None:
        """Handle a new file upload.
        
        Args:
            file_path: Path to uploaded file
        """
        self.logger.info(f"New upload detected: {file_path.name}")
        
        # Create upload event
        event = TriggerEvent(
            event_type="gedcom_upload",
            timestamp=datetime.now(),
            data={
                "file_path": str(file_path),
                "filename": file_path.name,
                "file_size": file_path.stat().st_size,
                "file_extension": file_path.suffix.lower()
            },
            source="upload_monitor"
        )
        
        # Trigger event
        await self.trigger_manager.trigger_event(event)


class TrainingTrigger:
    """Handles training pipeline triggers."""
    
    def __init__(self, trigger_manager: TriggerManager):
        """Initialize training trigger.
        
        Args:
            trigger_manager: Trigger manager instance
        """
        self.trigger_manager = trigger_manager
        self.logger = logging.getLogger(__name__)
        
        # Training conditions
        self.min_files_for_training = 5
        self.max_training_interval_hours = 24
        self.min_data_quality_score = 0.7
        
        # State tracking
        self.last_training_time: Optional[datetime] = None
        self.pending_files: List[Dict[str, Any]] = []

    async def check_training_conditions(self) -> bool:
        """Check if training should be triggered.
        
        Returns:
            True if training conditions are met
        """
        # Check minimum file count
        if len(self.pending_files) < self.min_files_for_training:
            return False
        
        # Check time since last training
        if self.last_training_time:
            time_since_training = datetime.now() - self.last_training_time
            if time_since_training < timedelta(hours=1):  # Minimum 1 hour between trainings
                return False
        
        # Check maximum interval
        if self.last_training_time:
            time_since_training = datetime.now() - self.last_training_time
            if time_since_training > timedelta(hours=self.max_training_interval_hours):
                self.logger.info("Max training interval reached, forcing training")
                return True
        
        # Check data quality
        avg_quality = sum(f.get("quality_score", 1.0) for f in self.pending_files) / len(self.pending_files)
        if avg_quality < self.min_data_quality_score:
            self.logger.warning(f"Data quality too low: {avg_quality:.2f}")
            return False
        
        return True

    async def add_processed_file(self, file_info: Dict[str, Any]) -> None:
        """Add a processed file to the training queue.
        
        Args:
            file_info: Information about the processed file
        """
        self.pending_files.append(file_info)
        self.logger.info(f"Added file to training queue: {file_info.get('filename', 'unknown')}")
        
        # Check if training should be triggered
        if await self.check_training_conditions():
            await self._trigger_training()

    async def _trigger_training(self) -> None:
        """Trigger a training run."""
        self.logger.info(f"Triggering training with {len(self.pending_files)} files")
        
        # Create training event
        event = TriggerEvent(
            event_type="training_trigger",
            timestamp=datetime.now(),
            data={
                "file_count": len(self.pending_files),
                "files": self.pending_files.copy(),
                "trigger_reason": "automatic"
            },
            source="training_trigger"
        )
        
        # Reset pending files and update last training time
        self.pending_files.clear()
        self.last_training_time = datetime.now()
        
        # Trigger event
        await self.trigger_manager.trigger_event(event)

    async def force_training(self, reason: str = "manual") -> None:
        """Force a training run regardless of conditions.
        
        Args:
            reason: Reason for forcing training
        """
        if not self.pending_files:
            self.logger.warning("No files available for forced training")
            return
        
        event = TriggerEvent(
            event_type="training_trigger",
            timestamp=datetime.now(),
            data={
                "file_count": len(self.pending_files),
                "files": self.pending_files.copy(),
                "trigger_reason": reason,
                "forced": True
            },
            source="manual_trigger"
        )
        
        self.pending_files.clear()
        self.last_training_time = datetime.now()
        
        await self.trigger_manager.trigger_event(event)


class WebhookTrigger:
    """Handles webhook triggers for external integrations."""
    
    def __init__(self, trigger_manager: TriggerManager, host: str = "0.0.0.0", port: int = 8080):
        """Initialize webhook trigger.
        
        Args:
            trigger_manager: Trigger manager instance
            host: Host to bind webhook server
            port: Port to bind webhook server
        """
        self.trigger_manager = trigger_manager
        self.host = host
        self.port = port
        self.logger = logging.getLogger(__name__)
        
        # Setup web application
        self.app = web.Application()
        self.app.router.add_post("/webhook/upload", self._handle_upload_webhook)
        self.app.router.add_post("/webhook/training", self._handle_training_webhook)
        self.app.router.add_get("/webhook/status", self._handle_status_webhook)
        
        # Authentication (simple token-based)
        self.webhook_token = "ohana_webhook_token_change_me"

    async def start_server(self) -> None:
        """Start the webhook server."""
        runner = web.AppRunner(self.app)
        await runner.setup()
        
        site = web.TCPSite(runner, self.host, self.port)
        await site.start()
        
        self.logger.info(f"Webhook server started on {self.host}:{self.port}")

    def _authenticate_request(self, request: web.Request) -> bool:
        """Authenticate webhook request.
        
        Args:
            request: HTTP request
            
        Returns:
            True if authenticated
        """
        auth_header = request.headers.get("Authorization")
        if not auth_header:
            return False
        
        try:
            scheme, token = auth_header.split(" ", 1)
            return scheme.lower() == "bearer" and token == self.webhook_token
        except ValueError:
            return False

    async def _handle_upload_webhook(self, request: web.Request) -> web.Response:
        """Handle upload webhook."""
        if not self._authenticate_request(request):
            return web.Response(status=401, text="Unauthorized")
        
        try:
            data = await request.json()
            
            # Create upload event
            event = TriggerEvent(
                event_type="webhook_upload",
                timestamp=datetime.now(),
                data=data,
                source="webhook"
            )
            
            await self.trigger_manager.trigger_event(event)
            
            return web.json_response({
                "status": "success",
                "event_id": event.event_id,
                "message": "Upload webhook processed"
            })
            
        except Exception as e:
            self.logger.error(f"Error processing upload webhook: {e}")
            return web.Response(status=500, text=str(e))

    async def _handle_training_webhook(self, request: web.Request) -> web.Response:
        """Handle training webhook."""
        if not self._authenticate_request(request):
            return web.Response(status=401, text="Unauthorized")
        
        try:
            data = await request.json()
            
            # Create training event
            event = TriggerEvent(
                event_type="webhook_training",
                timestamp=datetime.now(),
                data=data,
                source="webhook"
            )
            
            await self.trigger_manager.trigger_event(event)
            
            return web.json_response({
                "status": "success",
                "event_id": event.event_id,
                "message": "Training webhook processed"
            })
            
        except Exception as e:
            self.logger.error(f"Error processing training webhook: {e}")
            return web.Response(status=500, text=str(e))

    async def _handle_status_webhook(self, request: web.Request) -> web.Response:
        """Handle status webhook."""
        try:
            # Get recent events
            recent_events = self.trigger_manager.get_event_history(limit=10)
            
            status_data = {
                "status": "healthy",
                "timestamp": datetime.now().isoformat(),
                "recent_events": [
                    {
                        "event_id": event.event_id,
                        "event_type": event.event_type,
                        "timestamp": event.timestamp.isoformat(),
                        "source": event.source
                    }
                    for event in recent_events
                ]
            }
            
            return web.json_response(status_data)
            
        except Exception as e:
            self.logger.error(f"Error processing status webhook: {e}")
            return web.Response(status=500, text=str(e))


class NotificationTrigger:
    """Handles notifications for important events."""
    
    def __init__(self, trigger_manager: TriggerManager):
        """Initialize notification trigger.
        
        Args:
            trigger_manager: Trigger manager instance
        """
        self.trigger_manager = trigger_manager
        self.logger = logging.getLogger(__name__)
        
        # Notification settings
        self.notification_urls: List[str] = []
        self.notification_events: List[str] = [
            "training_complete",
            "training_failed",
            "model_deployed",
            "system_error"
        ]

    def add_notification_url(self, url: str) -> None:
        """Add a notification URL.
        
        Args:
            url: Webhook URL for notifications
        """
        self.notification_urls.append(url)
        self.logger.info(f"Added notification URL: {url}")

    async def send_notification(self, event: TriggerEvent) -> None:
        """Send notification for an event.
        
        Args:
            event: Event to notify about
        """
        if event.event_type not in self.notification_events:
            return
        
        notification_data = {
            "event_id": event.event_id,
            "event_type": event.event_type,
            "timestamp": event.timestamp.isoformat(),
            "source": event.source,
            "data": event.data
        }
        
        # Send to all notification URLs
        for url in self.notification_urls:
            try:
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        url,
                        json=notification_data,
                        headers={"Content-Type": "application/json"},
                        timeout=aiohttp.ClientTimeout(total=10)
                    ) as response:
                        if response.status == 200:
                            self.logger.info(f"Notification sent to {url}")
                        else:
                            self.logger.warning(f"Notification failed: {url} returned {response.status}")
                            
            except Exception as e:
                self.logger.error(f"Error sending notification to {url}: {e}")