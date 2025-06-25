"""
Monitoring and observability for the OhanaAI MLOps pipeline.
"""

import asyncio
import json
import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import psutil


@dataclass
class SystemMetrics:
    """System performance metrics."""
    
    timestamp: datetime
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    memory_total_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float
    gpu_utilization: float = 0.0  # Placeholder for GPU metrics
    gpu_memory_percent: float = 0.0


@dataclass
class PipelineMetrics:
    """Pipeline-specific metrics."""
    
    timestamp: datetime
    files_processed: int
    files_failed: int
    processing_queue_size: int
    training_queue_size: int
    cache_hit_rate: float
    avg_processing_time: float
    total_training_runs: int
    active_training_runs: int
    models_deployed: int


@dataclass
class AlertRule:
    """Defines an alert rule."""
    
    name: str
    metric: str
    condition: str  # "gt", "lt", "eq", "ne"
    threshold: float
    duration_minutes: int = 5  # How long condition must persist
    severity: str = "warning"  # "info", "warning", "error", "critical"
    enabled: bool = True
    last_triggered: Optional[datetime] = None


@dataclass
class Alert:
    """Represents an active alert."""
    
    rule_name: str
    metric: str
    current_value: float
    threshold: float
    severity: str
    triggered_at: datetime
    message: str
    resolved: bool = False
    resolved_at: Optional[datetime] = None


class MetricsCollector:
    """Collects system and pipeline metrics."""
    
    def __init__(self, collection_interval: float = 30.0):
        """Initialize metrics collector.
        
        Args:
            collection_interval: Interval between metric collections in seconds
        """
        self.collection_interval = collection_interval
        self.logger = logging.getLogger(__name__)
        
        # Metric storage (in-memory with size limits)
        self.system_metrics: deque = deque(maxlen=2880)  # 24 hours at 30s intervals
        self.pipeline_metrics: deque = deque(maxlen=2880)
        
        # Collection state
        self._collecting = False
        self._last_network_stats = None

    async def start_collection(self) -> None:
        """Start metrics collection."""
        if self._collecting:
            return
        
        self._collecting = True
        self.logger.info("Starting metrics collection")
        
        while self._collecting:
            try:
                # Collect system metrics
                system_metrics = await self._collect_system_metrics()
                self.system_metrics.append(system_metrics)
                
                # Collect pipeline metrics (would need pipeline instance)
                # pipeline_metrics = await self._collect_pipeline_metrics()
                # self.pipeline_metrics.append(pipeline_metrics)
                
                await asyncio.sleep(self.collection_interval)
                
            except Exception as e:
                self.logger.error(f"Error collecting metrics: {e}")
                await asyncio.sleep(self.collection_interval)

    async def stop_collection(self) -> None:
        """Stop metrics collection."""
        self._collecting = False
        self.logger.info("Stopped metrics collection")

    async def _collect_system_metrics(self) -> SystemMetrics:
        """Collect system performance metrics."""
        # CPU
        cpu_percent = psutil.cpu_percent(interval=1)
        
        # Memory
        memory = psutil.virtual_memory()
        memory_percent = memory.percent
        memory_used_gb = memory.used / (1024**3)
        memory_total_gb = memory.total / (1024**3)
        
        # Disk
        disk = psutil.disk_usage('/')
        disk_usage_percent = (disk.used / disk.total) * 100
        disk_free_gb = disk.free / (1024**3)
        
        # Network
        network = psutil.net_io_counters()
        if self._last_network_stats:
            sent_diff = network.bytes_sent - self._last_network_stats.bytes_sent
            recv_diff = network.bytes_recv - self._last_network_stats.bytes_recv
            network_sent_mb = sent_diff / (1024**2) / self.collection_interval
            network_recv_mb = recv_diff / (1024**2) / self.collection_interval
        else:
            network_sent_mb = 0.0
            network_recv_mb = 0.0
        
        self._last_network_stats = network
        
        return SystemMetrics(
            timestamp=datetime.now(),
            cpu_percent=cpu_percent,
            memory_percent=memory_percent,
            memory_used_gb=memory_used_gb,
            memory_total_gb=memory_total_gb,
            disk_usage_percent=disk_usage_percent,
            disk_free_gb=disk_free_gb,
            network_sent_mb=network_sent_mb,
            network_recv_mb=network_recv_mb
        )

    def get_system_metrics(self, minutes: int = 60) -> List[SystemMetrics]:
        """Get recent system metrics.
        
        Args:
            minutes: Number of minutes of metrics to return
            
        Returns:
            List of system metrics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.system_metrics if m.timestamp >= cutoff_time]

    def get_pipeline_metrics(self, minutes: int = 60) -> List[PipelineMetrics]:
        """Get recent pipeline metrics.
        
        Args:
            minutes: Number of minutes of metrics to return
            
        Returns:
            List of pipeline metrics
        """
        cutoff_time = datetime.now() - timedelta(minutes=minutes)
        return [m for m in self.pipeline_metrics if m.timestamp >= cutoff_time]

    def get_metrics_summary(self) -> Dict[str, Any]:
        """Get metrics summary.
        
        Returns:
            Dictionary containing metrics summary
        """
        if not self.system_metrics:
            return {"status": "no_data"}
        
        latest_system = self.system_metrics[-1]
        
        # Calculate averages for last hour
        hour_metrics = self.get_system_metrics(60)
        if hour_metrics:
            avg_cpu = sum(m.cpu_percent for m in hour_metrics) / len(hour_metrics)
            avg_memory = sum(m.memory_percent for m in hour_metrics) / len(hour_metrics)
        else:
            avg_cpu = latest_system.cpu_percent
            avg_memory = latest_system.memory_percent
        
        return {
            "timestamp": latest_system.timestamp.isoformat(),
            "current": {
                "cpu_percent": latest_system.cpu_percent,
                "memory_percent": latest_system.memory_percent,
                "memory_used_gb": latest_system.memory_used_gb,
                "disk_usage_percent": latest_system.disk_usage_percent,
                "disk_free_gb": latest_system.disk_free_gb
            },
            "hourly_average": {
                "cpu_percent": avg_cpu,
                "memory_percent": avg_memory
            },
            "collection_status": "active" if self._collecting else "stopped"
        }


class AlertManager:
    """Manages alerts and notifications."""
    
    def __init__(self, metrics_collector: MetricsCollector):
        """Initialize alert manager.
        
        Args:
            metrics_collector: Metrics collector instance
        """
        self.metrics_collector = metrics_collector
        self.logger = logging.getLogger(__name__)
        
        # Alert storage
        self.alert_rules: Dict[str, AlertRule] = {}
        self.active_alerts: Dict[str, Alert] = {}
        self.alert_history: List[Alert] = []
        
        # Alert state tracking
        self.rule_violations: Dict[str, List[datetime]] = defaultdict(list)
        
        # Initialize default alert rules
        self._setup_default_rules()

    def _setup_default_rules(self) -> None:
        """Setup default alert rules."""
        default_rules = [
            AlertRule(
                name="high_cpu_usage",
                metric="cpu_percent",
                condition="gt",
                threshold=80.0,
                duration_minutes=5,
                severity="warning"
            ),
            AlertRule(
                name="high_memory_usage",
                metric="memory_percent",
                condition="gt",
                threshold=85.0,
                duration_minutes=3,
                severity="warning"
            ),
            AlertRule(
                name="low_disk_space",
                metric="disk_free_gb",
                condition="lt",
                threshold=5.0,
                duration_minutes=1,
                severity="error"
            ),
            AlertRule(
                name="very_low_disk_space",
                metric="disk_free_gb",
                condition="lt",
                threshold=1.0,
                duration_minutes=1,
                severity="critical"
            )
        ]
        
        for rule in default_rules:
            self.alert_rules[rule.name] = rule

    def add_alert_rule(self, rule: AlertRule) -> None:
        """Add an alert rule.
        
        Args:
            rule: Alert rule to add
        """
        self.alert_rules[rule.name] = rule
        self.logger.info(f"Added alert rule: {rule.name}")

    def remove_alert_rule(self, rule_name: str) -> bool:
        """Remove an alert rule.
        
        Args:
            rule_name: Name of rule to remove
            
        Returns:
            True if rule was removed, False if not found
        """
        if rule_name in self.alert_rules:
            del self.alert_rules[rule_name]
            # Clean up any active alerts for this rule
            if rule_name in self.active_alerts:
                del self.active_alerts[rule_name]
            self.logger.info(f"Removed alert rule: {rule_name}")
            return True
        return False

    async def check_alerts(self) -> List[Alert]:
        """Check all alert rules and trigger alerts if needed.
        
        Returns:
            List of newly triggered alerts
        """
        new_alerts = []
        current_time = datetime.now()
        
        # Get latest metrics
        if not self.metrics_collector.system_metrics:
            return new_alerts
        
        latest_metrics = self.metrics_collector.system_metrics[-1]
        
        for rule_name, rule in self.alert_rules.items():
            if not rule.enabled:
                continue
            
            try:
                # Get metric value
                metric_value = getattr(latest_metrics, rule.metric, None)
                if metric_value is None:
                    continue
                
                # Check condition
                condition_met = self._evaluate_condition(
                    metric_value, rule.condition, rule.threshold
                )
                
                if condition_met:
                    # Add violation timestamp
                    self.rule_violations[rule_name].append(current_time)
                    
                    # Clean old violations outside duration window
                    cutoff_time = current_time - timedelta(minutes=rule.duration_minutes)
                    self.rule_violations[rule_name] = [
                        t for t in self.rule_violations[rule_name] if t >= cutoff_time
                    ]
                    
                    # Check if we have enough violations to trigger alert
                    violation_count = len(self.rule_violations[rule_name])
                    min_violations = max(1, rule.duration_minutes // 2)  # At least half the duration
                    
                    if (violation_count >= min_violations and 
                        rule_name not in self.active_alerts):
                        
                        # Trigger new alert
                        alert = Alert(
                            rule_name=rule_name,
                            metric=rule.metric,
                            current_value=metric_value,
                            threshold=rule.threshold,
                            severity=rule.severity,
                            triggered_at=current_time,
                            message=self._generate_alert_message(rule, metric_value)
                        )
                        
                        self.active_alerts[rule_name] = alert
                        self.alert_history.append(alert)
                        new_alerts.append(alert)
                        
                        self.logger.warning(f"Alert triggered: {alert.message}")
                        
                        # Update rule last triggered time
                        rule.last_triggered = current_time
                
                else:
                    # Condition not met, resolve alert if active
                    if rule_name in self.active_alerts:
                        alert = self.active_alerts[rule_name]
                        alert.resolved = True
                        alert.resolved_at = current_time
                        del self.active_alerts[rule_name]
                        
                        self.logger.info(f"Alert resolved: {rule_name}")
                    
                    # Clear violations
                    self.rule_violations[rule_name].clear()
                    
            except Exception as e:
                self.logger.error(f"Error checking alert rule {rule_name}: {e}")
        
        return new_alerts

    def _evaluate_condition(self, value: float, condition: str, threshold: float) -> bool:
        """Evaluate alert condition.
        
        Args:
            value: Current metric value
            condition: Condition to evaluate
            threshold: Threshold value
            
        Returns:
            True if condition is met
        """
        if condition == "gt":
            return value > threshold
        elif condition == "lt":
            return value < threshold
        elif condition == "eq":
            return abs(value - threshold) < 0.001  # Float comparison
        elif condition == "ne":
            return abs(value - threshold) >= 0.001
        else:
            return False

    def _generate_alert_message(self, rule: AlertRule, current_value: float) -> str:
        """Generate alert message.
        
        Args:
            rule: Alert rule that triggered
            current_value: Current metric value
            
        Returns:
            Alert message string
        """
        return (
            f"{rule.name}: {rule.metric} is {current_value:.2f} "
            f"({rule.condition} {rule.threshold})"
        )

    def get_active_alerts(self) -> List[Alert]:
        """Get all active alerts.
        
        Returns:
            List of active alerts
        """
        return list(self.active_alerts.values())

    def get_alert_history(self, hours: int = 24) -> List[Alert]:
        """Get alert history.
        
        Args:
            hours: Number of hours of history to return
            
        Returns:
            List of historical alerts
        """
        cutoff_time = datetime.now() - timedelta(hours=hours)
        return [a for a in self.alert_history if a.triggered_at >= cutoff_time]


class PipelineMonitor:
    """Comprehensive monitoring for the MLOps pipeline."""
    
    def __init__(self, metrics_collector: MetricsCollector, alert_manager: AlertManager):
        """Initialize pipeline monitor.
        
        Args:
            metrics_collector: Metrics collector instance
            alert_manager: Alert manager instance
        """
        self.metrics_collector = metrics_collector
        self.alert_manager = alert_manager
        self.logger = logging.getLogger(__name__)
        
        # Monitoring state
        self._monitoring = False
        self.check_interval = 60.0  # Check alerts every minute

    async def start_monitoring(self) -> None:
        """Start comprehensive monitoring."""
        if self._monitoring:
            return
        
        self._monitoring = True
        self.logger.info("Starting pipeline monitoring")
        
        # Start metrics collection
        await self.metrics_collector.start_collection()
        
        # Start alert checking loop
        while self._monitoring:
            try:
                new_alerts = await self.alert_manager.check_alerts()
                
                # Log new alerts
                for alert in new_alerts:
                    self.logger.warning(f"New alert: {alert.message}")
                
                await asyncio.sleep(self.check_interval)
                
            except Exception as e:
                self.logger.error(f"Error in monitoring loop: {e}")
                await asyncio.sleep(self.check_interval)

    async def stop_monitoring(self) -> None:
        """Stop monitoring."""
        self._monitoring = False
        await self.metrics_collector.stop_collection()
        self.logger.info("Stopped pipeline monitoring")

    def get_health_status(self) -> Dict[str, Any]:
        """Get overall health status.
        
        Returns:
            Dictionary containing health status
        """
        active_alerts = self.alert_manager.get_active_alerts()
        metrics_summary = self.metrics_collector.get_metrics_summary()
        
        # Determine overall health
        critical_alerts = [a for a in active_alerts if a.severity == "critical"]
        error_alerts = [a for a in active_alerts if a.severity == "error"]
        warning_alerts = [a for a in active_alerts if a.severity == "warning"]
        
        if critical_alerts:
            health_status = "critical"
        elif error_alerts:
            health_status = "error"
        elif warning_alerts:
            health_status = "warning"
        else:
            health_status = "healthy"
        
        return {
            "status": health_status,
            "timestamp": datetime.now().isoformat(),
            "alerts": {
                "active": len(active_alerts),
                "critical": len(critical_alerts),
                "error": len(error_alerts),
                "warning": len(warning_alerts)
            },
            "metrics": metrics_summary,
            "monitoring_active": self._monitoring
        }

    def export_metrics(self, format: str = "json", file_path: Optional[Path] = None) -> Union[str, bool]:
        """Export metrics data.
        
        Args:
            format: Export format ("json", "csv")
            file_path: File path to save to (optional)
            
        Returns:
            Exported data as string or True if saved to file
        """
        if format == "json":
            data = {
                "system_metrics": [asdict(m) for m in self.metrics_collector.system_metrics],
                "pipeline_metrics": [asdict(m) for m in self.metrics_collector.pipeline_metrics],
                "alert_history": [asdict(a) for a in self.alert_manager.alert_history],
                "export_timestamp": datetime.now().isoformat()
            }
            
            json_data = json.dumps(data, indent=2, default=str)
            
            if file_path:
                with open(file_path, "w") as f:
                    f.write(json_data)
                return True
            else:
                return json_data
        
        elif format == "csv":
            # Implement CSV export if needed
            raise NotImplementedError("CSV export not yet implemented")
        
        else:
            raise ValueError(f"Unsupported export format: {format}")


# Example usage and integration
async def setup_monitoring() -> PipelineMonitor:
    """Setup and start monitoring."""
    # Initialize components
    metrics_collector = MetricsCollector(collection_interval=30.0)
    alert_manager = AlertManager(metrics_collector)
    pipeline_monitor = PipelineMonitor(metrics_collector, alert_manager)
    
    # Add custom alert rules
    alert_manager.add_alert_rule(AlertRule(
        name="high_queue_size",
        metric="processing_queue_size",
        condition="gt",
        threshold=100.0,
        duration_minutes=10,
        severity="warning"
    ))
    
    # Start monitoring
    await pipeline_monitor.start_monitoring()
    
    return pipeline_monitor