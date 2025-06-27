"""
Database management for OhanaAI MLOps pipeline.
Handles GEDCOM files, processed graphs, and training metadata.
"""

import json
import os
import pickle
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import psycopg2
from sqlalchemy import create_engine, text
from dataclasses import dataclass

import numpy as np


class ProcessingStatus(Enum):
    """Status of GEDCOM file processing."""
    UPLOADED = "uploaded"
    PROCESSING = "processing"
    PROCESSED = "processed"
    FAILED = "failed"


class TrainingStatus(Enum):
    """Status of training runs."""
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class GedcomRecord:
    """Record for a GEDCOM file in the database."""
    id: Optional[int] = None
    filename: str = ""
    original_filename: str = ""
    file_hash: str = ""
    file_size: int = 0
    upload_time: datetime = None
    status: ProcessingStatus = ProcessingStatus.UPLOADED
    num_individuals: int = 0
    num_families: int = 0
    date_range: Tuple[int, int] = (0, 0)
    processing_time: float = 0.0
    error_message: str = ""
    metadata: Dict[str, Any] = None

    def __post_init__(self):
        if self.upload_time is None:
            self.upload_time = datetime.now()
        if self.metadata is None:
            self.metadata = {}


@dataclass
class GraphCache:
    """Cache record for processed graph data."""
    id: Optional[int] = None
    gedcom_id: int = 0
    graph_hash: str = ""
    node_features: bytes = b""
    edge_indices: bytes = b""
    edge_types: bytes = b""
    node_metadata: str = ""
    created_time: datetime = None
    access_count: int = 0
    last_accessed: datetime = None

    def __post_init__(self):
        if self.created_time is None:
            self.created_time = datetime.now()
        if self.last_accessed is None:
            self.last_accessed = datetime.now()


@dataclass
class TrainingRun:
    """Record for a training run."""
    id: Optional[int] = None
    run_name: str = ""
    status: TrainingStatus = TrainingStatus.QUEUED
    config: Dict[str, Any] = None
    gedcom_ids: List[int] = None
    start_time: datetime = None
    end_time: Optional[datetime] = None
    duration: float = 0.0
    final_metrics: Dict[str, float] = None
    model_path: str = ""
    model_version: str = ""
    error_message: str = ""
    triggered_by: str = "manual"  # manual, upload, scheduled

    def __post_init__(self):
        if self.start_time is None:
            self.start_time = datetime.now()
        if self.config is None:
            self.config = {}
        if self.gedcom_ids is None:
            self.gedcom_ids = []
        if self.final_metrics is None:
            self.final_metrics = {}


class DatabaseManager:
    """Manages PostgreSQL database for MLOps pipeline."""

    def __init__(self):
        self.db_url = os.environ.get("DATABASE_URL")
        if not self.db_url:
            raise ValueError("DATABASE_URL environment variable not set.")
        self.engine = create_engine(self.db_url)
        self._init_database()

    def _init_database(self) -> None:
        with self.engine.connect() as connection:
            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS gedcom_files (
                    id SERIAL PRIMARY KEY,
                    filename TEXT NOT NULL,
                    original_filename TEXT NOT NULL,
                    file_hash TEXT UNIQUE NOT NULL,
                    file_size INTEGER NOT NULL,
                    upload_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    status TEXT NOT NULL DEFAULT 'uploaded',
                    num_individuals INTEGER DEFAULT 0,
                    num_families INTEGER DEFAULT 0,
                    date_range_start INTEGER DEFAULT 0,
                    date_range_end INTEGER DEFAULT 0,
                    processing_time REAL DEFAULT 0.0,
                    error_message TEXT DEFAULT '',
                    metadata JSONB DEFAULT '{}'
                )
            """))

            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS graph_cache (
                    id SERIAL PRIMARY KEY,
                    gedcom_id INTEGER NOT NULL,
                    graph_hash TEXT UNIQUE NOT NULL,
                    node_features BYTEA NOT NULL,
                    edge_indices BYTEA NOT NULL,
                    edge_types BYTEA NOT NULL,
                    node_metadata JSONB DEFAULT '{}',
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (gedcom_id) REFERENCES gedcom_files (id)
                )
            """))

            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id SERIAL PRIMARY KEY,
                    run_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    config JSONB NOT NULL DEFAULT '{}',
                    gedcom_ids JSONB NOT NULL DEFAULT '[]',
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP NULL,
                    duration REAL DEFAULT 0.0,
                    final_metrics JSONB DEFAULT '{}',
                    model_path TEXT DEFAULT '',
                    model_version TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    triggered_by TEXT DEFAULT 'manual'
                )
            """))

            connection.execute(text("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id SERIAL PRIMARY KEY,
                    run_id INTEGER NOT NULL,
                    epoch INTEGER NOT NULL,
                    train_loss REAL NOT NULL,
                    train_accuracy REAL NOT NULL,
                    val_loss REAL NOT NULL,
                    val_accuracy REAL NOT NULL,
                    learning_rate REAL NOT NULL,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (run_id) REFERENCES training_runs (id)
                )
            """))

            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_gedcom_hash ON gedcom_files (file_hash)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_gedcom_status ON gedcom_files (status)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_graph_hash ON graph_cache (graph_hash)"))
            connection.execute(text("CREATE INDEX IF NOT EXISTS idx_training_status ON training_runs (status)"))
            connection.commit()

    def add_gedcom_file(self, record: GedcomRecord) -> int:
        with self.engine.connect() as connection:
            result = connection.execute(text("""
                INSERT INTO gedcom_files (
                    filename, original_filename, file_hash, file_size, upload_time,
                    status, num_individuals, num_families, date_range_start, date_range_end,
                    processing_time, error_message, metadata
                ) VALUES (:filename, :original_filename, :file_hash, :file_size, :upload_time,
                    :status, :num_individuals, :num_families, :date_range_start, :date_range_end,
                    :processing_time, :error_message, :metadata)
                RETURNING id
            """), {
                "filename": record.filename,
                "original_filename": record.original_filename,
                "file_hash": record.file_hash,
                "file_size": record.file_size,
                "upload_time": record.upload_time,
                "status": record.status.value,
                "num_individuals": record.num_individuals,
                "num_families": record.num_families,
                "date_range_start": record.date_range[0],
                "date_range_end": record.date_range[1],
                "processing_time": record.processing_time,
                "error_message": record.error_message,
                "metadata": json.dumps(record.metadata)
            })
            connection.commit()
            return result.scalar_one()

    def get_gedcom_file(self, file_id: int) -> Optional[GedcomRecord]:
        with self.engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM gedcom_files WHERE id = :file_id"), {"file_id": file_id})
            row = result.fetchone()
            
            if row:
                return GedcomRecord(
                    id=row.id,
                    filename=row.filename,
                    original_filename=row.original_filename,
                    file_hash=row.file_hash,
                    file_size=row.file_size,
                    upload_time=row.upload_time,
                    status=ProcessingStatus(row.status),
                    num_individuals=row.num_individuals,
                    num_families=row.num_families,
                    date_range=(row.date_range_start, row.date_range_end),
                    processing_time=row.processing_time,
                    error_message=row.error_message,
                    metadata=json.loads(row.metadata)
                )
        return None

    def update_gedcom_status(self, file_id: int, status: ProcessingStatus, 
                           error_message: str = "") -> None:
        with self.engine.connect() as connection:
            connection.execute(text("""
                UPDATE gedcom_files 
                SET status = :status, error_message = :error_message
                WHERE id = :file_id
            """), {"status": status.value, "error_message": error_message, "file_id": file_id})
            connection.commit()

    def get_files_by_status(self, status: ProcessingStatus) -> List[GedcomRecord]:
        records = []
        with self.engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM gedcom_files WHERE status = :status"), {"status": status.value})
            
            for row in result.fetchall():
                records.append(GedcomRecord(
                    id=row.id,
                    filename=row.filename,
                    original_filename=row.original_filename,
                    file_hash=row.file_hash,
                    file_size=row.file_size,
                    upload_time=row.upload_time,
                    status=ProcessingStatus(row.status),
                    num_individuals=row.num_individuals,
                    num_families=row.num_families,
                    date_range=(row.date_range_start, row.date_range_end),
                    processing_time=row.processing_time,
                    error_message=row.error_message,
                    metadata=json.loads(row.metadata)
                ))
        return records

    def cache_graph(self, cache: GraphCache) -> int:
        with self.engine.connect() as connection:
            result = connection.execute(text("""
                INSERT INTO graph_cache (
                    gedcom_id, graph_hash, node_features, edge_indices, edge_types,
                    node_metadata, created_time, access_count, last_accessed
                ) VALUES (:gedcom_id, :graph_hash, :node_features, :edge_indices, :edge_types,
                    :node_metadata, :created_time, :access_count, :last_accessed)
                RETURNING id
            """), {
                "gedcom_id": cache.gedcom_id,
                "graph_hash": cache.graph_hash,
                "node_features": cache.node_features,
                "edge_indices": cache.edge_indices,
                "edge_types": cache.edge_types,
                "node_metadata": cache.node_metadata,
                "created_time": cache.created_time,
                "access_count": cache.access_count,
                "last_accessed": cache.last_accessed
            })
            connection.commit()
            return result.scalar_one()

    def get_cached_graph(self, graph_hash: str) -> Optional[GraphCache]:
        with self.engine.connect() as connection:
            result = connection.execute(text("SELECT * FROM graph_cache WHERE graph_hash = :graph_hash"), {"graph_hash": graph_hash})
            row = result.fetchone()
            
            if row:
                # Update access count
                connection.execute(text("""
                    UPDATE graph_cache 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id = :id
                """), {"id": row.id})
                connection.commit()
                
                return GraphCache(
                    id=row.id,
                    gedcom_id=row.gedcom_id,
                    graph_hash=row.graph_hash,
                    node_features=row.node_features,
                    edge_indices=row.edge_indices,
                    edge_types=row.edge_types,
                    node_metadata=row.node_metadata,
                    created_time=row.created_time,
                    access_count=row.access_count + 1,
                    last_accessed=datetime.now()
                )
        return None

    def add_training_run(self, run: TrainingRun) -> int:
        with self.engine.connect() as connection:
            result = connection.execute(text("""
                INSERT INTO training_runs (
                    run_name, status, config, gedcom_ids, start_time, end_time,
                    duration, final_metrics, model_path, model_version,
                    error_message, triggered_by
                ) VALUES (:run_name, :status, :config, :gedcom_ids, :start_time, :end_time,
                    :duration, :final_metrics, :model_path, :model_version,
                    :error_message, :triggered_by)
                RETURNING id
            """), {
                "run_name": run.run_name,
                "status": run.status.value,
                "config": json.dumps(run.config),
                "gedcom_ids": json.dumps(run.gedcom_ids),
                "start_time": run.start_time,
                "end_time": run.end_time,
                "duration": run.duration,
                "final_metrics": json.dumps(run.final_metrics),
                "model_path": run.model_path,
                "model_version": run.model_version,
                "error_message": run.error_message,
                "triggered_by": run.triggered_by
            })
            connection.commit()
            return result.scalar_one()

    def update_training_run(self, run_id: int, status: TrainingStatus,
                           end_time: Optional[datetime] = None,
                           duration: float = 0.0,
                           final_metrics: Optional[Dict[str, float]] = None,
                           model_path: str = "",
                           model_version: str = "",
                           error_message: str = "") -> None:
        if final_metrics is None:
            final_metrics = {}
            
        with self.engine.connect() as connection:
            connection.execute(text("""
                UPDATE training_runs 
                SET status = :status, end_time = :end_time, duration = :duration, final_metrics = :final_metrics,
                    model_path = :model_path, model_version = :model_version, error_message = :error_message
                WHERE id = :run_id
            """), {
                "status": status.value,
                "end_time": end_time,
                "duration": duration,
                "final_metrics": json.dumps(final_metrics),
                "model_path": model_path,
                "model_version": model_version,
                "error_message": error_message,
                "run_id": run_id
            })
            connection.commit()

    def log_training_metrics(self, run_id: int, epoch: int, train_loss: float,
                           train_accuracy: float, val_loss: float, val_accuracy: float,
                           learning_rate: float) -> None:
        with self.engine.connect() as connection:
            connection.execute(text("""
                INSERT INTO training_metrics (
                    run_id, epoch, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate
                ) VALUES (:run_id, :epoch, :train_loss, :train_accuracy, :val_loss, :val_accuracy, :learning_rate)
            """), {
                "run_id": run_id,
                "epoch": epoch,
                "train_loss": train_loss,
                "train_accuracy": train_accuracy,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "learning_rate": learning_rate
            })
            connection.commit()

    def get_training_runs(self, status: Optional[TrainingStatus] = None,
                         limit: int = 100) -> List[TrainingRun]:
        runs = []
        with self.engine.connect() as connection:
            if status:
                result = connection.execute(text("""
                    SELECT * FROM training_runs WHERE status = :status
                    ORDER BY start_time DESC LIMIT :limit
                """), {"status": status.value, "limit": limit})
            else:
                result = connection.execute(text("""
                    SELECT * FROM training_runs 
                    ORDER BY start_time DESC LIMIT :limit
                """), {"limit": limit})
            
            for row in result.fetchall():
                end_time = None
                if row.end_time:
                    end_time = row.end_time
                
                runs.append(TrainingRun(
                    id=row.id,
                    run_name=row.run_name,
                    status=TrainingStatus(row.status),
                    config=json.loads(row.config),
                    gedcom_ids=json.loads(row.gedcom_ids),
                    start_time=row.start_time,
                    end_time=end_time,
                    duration=row.duration,
                    final_metrics=json.loads(row.final_metrics),
                    model_path=row.model_path,
                    model_version=row.model_version,
                    error_message=row.error_message,
                    triggered_by=row.triggered_by
                ))
        return runs

    def cleanup_old_cache(self, max_age_days: int = 30) -> int:
        with self.engine.connect() as connection:
            result = connection.execute(text("""
                DELETE FROM graph_cache 
                WHERE created_time < NOW() - INTERVAL ':max_age_days days'
            """), {"max_age_days": max_age_days})
            connection.commit()
            return result.rowcount

    def get_database_stats(self) -> Dict[str, Any]:
        stats = {}
        with self.engine.connect() as connection:
            result = connection.execute(text("""
                SELECT status, COUNT(*) as count 
                FROM gedcom_files 
                GROUP BY status
            """))
            stats["gedcom_files"] = {row.status: row.count for row in result.fetchall()}
            
            result = connection.execute(text("""
                SELECT status, COUNT(*) as count 
                FROM training_runs 
                GROUP BY status
            """))
            stats["training_runs"] = {row.status: row.count for row in result.fetchall()}
            
            result = connection.execute(text("""
                SELECT COUNT(*) as count, SUM(access_count) as total_accesses
                FROM graph_cache
            """))
            row = result.fetchone()
            stats["graph_cache"] = {
                "count": row.count,
                "total_accesses": row.total_accesses or 0
            }
            
            # Database size (approximation for PostgreSQL)
            result = connection.execute(text("SELECT pg_database_size(current_database())"))
            stats["database_size"] = result.scalar_one()
            
        return stats