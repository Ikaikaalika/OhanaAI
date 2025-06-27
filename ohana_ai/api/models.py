"""
Pydantic models for API requests and responses.
"""

from datetime import datetime
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class UploadResponse(BaseModel):
    """Response for file upload."""
    
    message: str
    file_id: int
    filename: str
    size: int


class FileInfo(BaseModel):
    """GEDCOM file information."""
    
    id: int
    filename: str
    status: str
    upload_time: str
    num_individuals: int
    num_families: int
    processing_time: float


class FileDetails(BaseModel):
    """Detailed GEDCOM file information."""
    
    id: int
    filename: str
    status: str
    upload_time: str
    file_size: int
    num_individuals: int
    num_families: int
    date_range: List[int]
    processing_time: float
    error_message: str
    metadata: Dict[str, Any]


class TrainingRequest(BaseModel):
    """Request to start training."""
    
    gedcom_ids: List[int] = Field(..., description="List of GEDCOM file IDs to train on")
    config_overrides: Optional[Dict[str, Any]] = Field(None, description="Configuration overrides")
    tags: Optional[List[str]] = Field(None, description="Tags for the training run")
    notes: Optional[str] = Field(None, description="Notes about the training run")


class TrainingRunInfo(BaseModel):
    """Training run information."""
    
    id: int
    run_name: str
    status: str
    start_time: str
    end_time: Optional[str]
    duration: float
    final_metrics: Dict[str, float]
    model_version: str
    triggered_by: str


class TrainingRunDetails(BaseModel):
    """Detailed training run information."""
    
    id: int
    run_name: str
    status: str
    config: Dict[str, Any]
    gedcom_ids: List[int]
    start_time: str
    end_time: Optional[str]
    duration: float
    final_metrics: Dict[str, float]
    model_path: str
    model_version: str
    error_message: str
    triggered_by: str


class ModelInfo(BaseModel):
    """Model version information."""
    
    version: str
    created_at: str
    metrics: Dict[str, float]
    model_size_mb: float
    training_duration_seconds: float
    is_production: bool
    tags: List[str]
    notes: str


class ModelDetails(BaseModel):
    """Detailed model information."""
    
    version: str
    created_at: str
    config: Dict[str, Any]
    metrics: Dict[str, float]
    training_data: Dict[str, Any]
    model_size_mb: float
    training_duration_seconds: float
    parent_version: Optional[str]
    tags: List[str]
    notes: str
    is_production: bool


class SystemMetric(BaseModel):
    """System performance metric."""
    
    timestamp: str
    cpu_percent: float
    memory_percent: float
    memory_used_gb: float
    disk_usage_percent: float
    disk_free_gb: float
    network_sent_mb: float
    network_recv_mb: float


class AlertInfo(BaseModel):
    """Alert information."""
    
    rule_name: str
    metric: str
    current_value: float
    threshold: float
    severity: str
    triggered_at: str
    message: str


class HealthStatus(BaseModel):
    """System health status."""
    
    status: str  # healthy, warning, error, critical
    timestamp: str
    alerts: Dict[str, int]  # alert counts by type
    metrics: Dict[str, Any]
    monitoring_active: bool


class PipelineStatus(BaseModel):
    """Pipeline status information."""
    
    running: bool
    processing_queue_size: int
    training_queue_size: int
    database_stats: Dict[str, Any]
    config: Dict[str, Any]


class StatusResponse(BaseModel):
    """Overall system status response."""
    
    timestamp: str
    pipeline: PipelineStatus
    database: Dict[str, Any]
    storage: Dict[str, Any]


class PredictionRequest(BaseModel):
    """Request for parent predictions."""
    
    gedcom_file: str = Field(..., description="GEDCOM file content as a string")
    candidate_pairs: List[List[int]] = Field(..., description="List of candidate parent-child pairs")


class ExperimentRequest(BaseModel):
    """Request to create an experiment."""
    
    name: str = Field(..., description="Experiment name")
    config: Dict[str, Any] = Field(..., description="Experiment configuration")
    gedcom_ids: List[int] = Field(..., description="GEDCOM files to use")
    parent_run_id: Optional[str] = Field(None, description="Parent experiment run ID")
    tags: Optional[List[str]] = Field(None, description="Experiment tags")


class ExperimentInfo(BaseModel):
    """Experiment run information."""
    
    id: str
    name: str
    status: str
    created_at: str
    duration_seconds: float
    metrics: Dict[str, float]
    tags: List[str]


class ModelComparisonRequest(BaseModel):
    """Request to compare models."""
    
    version1: str = Field(..., description="First model version")
    version2: str = Field(..., description="Second model version")


class ModelComparison(BaseModel):
    """Model comparison result."""
    
    versions: List[str]
    metrics_comparison: Dict[str, Dict[str, float]]
    config_differences: List[Dict[str, Any]]
    performance_delta: Dict[str, float]


class WebhookEvent(BaseModel):
    """Webhook event payload."""
    
    event_type: str
    timestamp: str
    data: Dict[str, Any]
    source: str


class APIError(BaseModel):
    """API error response."""
    
    error: str
    detail: str
    timestamp: str
    request_id: Optional[str] = None


class BatchUploadRequest(BaseModel):
    """Request for batch file upload."""
    
    files: List[str] = Field(..., description="List of file paths to upload")
    tags: Optional[List[str]] = Field(None, description="Tags for uploaded files")
    auto_train: Optional[bool] = Field(True, description="Automatically trigger training")


class BatchUploadResponse(BaseModel):
    """Response for batch upload."""
    
    message: str
    uploaded_files: List[Dict[str, Any]]
    failed_files: List[Dict[str, Any]]
    total_files: int
    successful_uploads: int