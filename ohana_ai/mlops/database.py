"""
Database management for OhanaAI MLOps pipeline.
Handles GEDCOM files, processed graphs, and training metadata.
"""

import json
import pickle
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import sqlite3
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
    """Manages SQLite database for MLOps pipeline."""

    def __init__(self, db_path: str = "ohana_mlops.db"):
        """Initialize database manager.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_database()

    def _init_database(self) -> None:
        """Initialize database tables."""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("PRAGMA foreign_keys = ON")
            
            # GEDCOM files table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS gedcom_files (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
                    metadata TEXT DEFAULT '{}'
                )
            """)

            # Graph cache table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS graph_cache (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    gedcom_id INTEGER NOT NULL,
                    graph_hash TEXT UNIQUE NOT NULL,
                    node_features BLOB NOT NULL,
                    edge_indices BLOB NOT NULL,
                    edge_types BLOB NOT NULL,
                    node_metadata TEXT DEFAULT '{}',
                    created_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    access_count INTEGER DEFAULT 0,
                    last_accessed TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (gedcom_id) REFERENCES gedcom_files (id)
                )
            """)

            # Training runs table
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_runs (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    run_name TEXT NOT NULL,
                    status TEXT NOT NULL DEFAULT 'queued',
                    config TEXT NOT NULL DEFAULT '{}',
                    gedcom_ids TEXT NOT NULL DEFAULT '[]',
                    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    end_time TIMESTAMP NULL,
                    duration REAL DEFAULT 0.0,
                    final_metrics TEXT DEFAULT '{}',
                    model_path TEXT DEFAULT '',
                    model_version TEXT DEFAULT '',
                    error_message TEXT DEFAULT '',
                    triggered_by TEXT DEFAULT 'manual'
                )
            """)

            # Training metrics table (for detailed tracking)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS training_metrics (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
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
            """)

            # Indexes for performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gedcom_hash ON gedcom_files (file_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_gedcom_status ON gedcom_files (status)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_graph_hash ON graph_cache (graph_hash)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_training_status ON training_runs (status)")

            conn.commit()

    def add_gedcom_file(self, record: GedcomRecord) -> int:
        """Add a GEDCOM file record to database.
        
        Args:
            record: GEDCOM record to add
            
        Returns:
            Database ID of inserted record
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO gedcom_files (
                    filename, original_filename, file_hash, file_size, upload_time,
                    status, num_individuals, num_families, date_range_start, date_range_end,
                    processing_time, error_message, metadata
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                record.filename, record.original_filename, record.file_hash, record.file_size,
                record.upload_time, record.status.value, record.num_individuals, record.num_families,
                record.date_range[0], record.date_range[1], record.processing_time,
                record.error_message, json.dumps(record.metadata)
            ))
            return cursor.lastrowid

    def get_gedcom_file(self, file_id: int) -> Optional[GedcomRecord]:
        """Get GEDCOM file record by ID.
        
        Args:
            file_id: Database ID of record
            
        Returns:
            GEDCOM record or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM gedcom_files WHERE id = ?", (file_id,))
            row = cursor.fetchone()
            
            if row:
                return GedcomRecord(
                    id=row["id"],
                    filename=row["filename"],
                    original_filename=row["original_filename"],
                    file_hash=row["file_hash"],
                    file_size=row["file_size"],
                    upload_time=datetime.fromisoformat(row["upload_time"]),
                    status=ProcessingStatus(row["status"]),
                    num_individuals=row["num_individuals"],
                    num_families=row["num_families"],
                    date_range=(row["date_range_start"], row["date_range_end"]),
                    processing_time=row["processing_time"],
                    error_message=row["error_message"],
                    metadata=json.loads(row["metadata"])
                )
        return None

    def update_gedcom_status(self, file_id: int, status: ProcessingStatus, 
                           error_message: str = "") -> None:
        """Update GEDCOM file processing status.
        
        Args:
            file_id: Database ID of record
            status: New processing status
            error_message: Error message if status is FAILED
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE gedcom_files 
                SET status = ?, error_message = ?
                WHERE id = ?
            """, (status.value, error_message, file_id))
            conn.commit()

    def get_files_by_status(self, status: ProcessingStatus) -> List[GedcomRecord]:
        """Get all GEDCOM files with a specific status.
        
        Args:
            status: Processing status to filter by
            
        Returns:
            List of GEDCOM records
        """
        records = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("SELECT * FROM gedcom_files WHERE status = ?", (status.value,))
            
            for row in cursor.fetchall():
                records.append(GedcomRecord(
                    id=row["id"],
                    filename=row["filename"],
                    original_filename=row["original_filename"],
                    file_hash=row["file_hash"],
                    file_size=row["file_size"],
                    upload_time=datetime.fromisoformat(row["upload_time"]),
                    status=ProcessingStatus(row["status"]),
                    num_individuals=row["num_individuals"],
                    num_families=row["num_families"],
                    date_range=(row["date_range_start"], row["date_range_end"]),
                    processing_time=row["processing_time"],
                    error_message=row["error_message"],
                    metadata=json.loads(row["metadata"])
                ))
        return records

    def cache_graph(self, cache: GraphCache) -> int:
        """Cache processed graph data.
        
        Args:
            cache: Graph cache record
            
        Returns:
            Database ID of cached graph
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO graph_cache (
                    gedcom_id, graph_hash, node_features, edge_indices, edge_types,
                    node_metadata, created_time, access_count, last_accessed
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                cache.gedcom_id, cache.graph_hash, cache.node_features, cache.edge_indices,
                cache.edge_types, cache.node_metadata, cache.created_time,
                cache.access_count, cache.last_accessed
            ))
            return cursor.lastrowid

    def get_cached_graph(self, graph_hash: str) -> Optional[GraphCache]:
        """Get cached graph by hash.
        
        Args:
            graph_hash: Hash of graph data
            
        Returns:
            Graph cache record or None if not found
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            cursor = conn.execute("""
                SELECT * FROM graph_cache WHERE graph_hash = ?
            """, (graph_hash,))
            row = cursor.fetchone()
            
            if row:
                # Update access count
                conn.execute("""
                    UPDATE graph_cache 
                    SET access_count = access_count + 1, last_accessed = CURRENT_TIMESTAMP
                    WHERE id = ?
                """, (row["id"],))
                conn.commit()
                
                return GraphCache(
                    id=row["id"],
                    gedcom_id=row["gedcom_id"],
                    graph_hash=row["graph_hash"],
                    node_features=row["node_features"],
                    edge_indices=row["edge_indices"],
                    edge_types=row["edge_types"],
                    node_metadata=row["node_metadata"],
                    created_time=datetime.fromisoformat(row["created_time"]),
                    access_count=row["access_count"] + 1,
                    last_accessed=datetime.now()
                )
        return None

    def add_training_run(self, run: TrainingRun) -> int:
        """Add a training run record.
        
        Args:
            run: Training run record
            
        Returns:
            Database ID of training run
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                INSERT INTO training_runs (
                    run_name, status, config, gedcom_ids, start_time, end_time,
                    duration, final_metrics, model_path, model_version,
                    error_message, triggered_by
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                run.run_name, run.status.value, json.dumps(run.config),
                json.dumps(run.gedcom_ids), run.start_time, run.end_time,
                run.duration, json.dumps(run.final_metrics), run.model_path,
                run.model_version, run.error_message, run.triggered_by
            ))
            return cursor.lastrowid

    def update_training_run(self, run_id: int, status: TrainingStatus,
                           end_time: Optional[datetime] = None,
                           duration: float = 0.0,
                           final_metrics: Optional[Dict[str, float]] = None,
                           model_path: str = "",
                           model_version: str = "",
                           error_message: str = "") -> None:
        """Update training run status and results.
        
        Args:
            run_id: Database ID of training run
            status: New training status
            end_time: Training end time
            duration: Training duration in seconds
            final_metrics: Final training metrics
            model_path: Path to saved model
            model_version: Model version string
            error_message: Error message if training failed
        """
        if final_metrics is None:
            final_metrics = {}
            
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                UPDATE training_runs 
                SET status = ?, end_time = ?, duration = ?, final_metrics = ?,
                    model_path = ?, model_version = ?, error_message = ?
                WHERE id = ?
            """, (
                status.value, end_time, duration, json.dumps(final_metrics),
                model_path, model_version, error_message, run_id
            ))
            conn.commit()

    def log_training_metrics(self, run_id: int, epoch: int, train_loss: float,
                           train_accuracy: float, val_loss: float, val_accuracy: float,
                           learning_rate: float) -> None:
        """Log training metrics for an epoch.
        
        Args:
            run_id: Database ID of training run
            epoch: Training epoch number
            train_loss: Training loss
            train_accuracy: Training accuracy
            val_loss: Validation loss
            val_accuracy: Validation accuracy
            learning_rate: Current learning rate
        """
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO training_metrics (
                    run_id, epoch, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate
                ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (run_id, epoch, train_loss, train_accuracy, val_loss, val_accuracy, learning_rate))
            conn.commit()

    def get_training_runs(self, status: Optional[TrainingStatus] = None,
                         limit: int = 100) -> List[TrainingRun]:
        """Get training runs, optionally filtered by status.
        
        Args:
            status: Training status to filter by (optional)
            limit: Maximum number of runs to return
            
        Returns:
            List of training run records
        """
        runs = []
        with sqlite3.connect(self.db_path) as conn:
            conn.row_factory = sqlite3.Row
            
            if status:
                cursor = conn.execute("""
                    SELECT * FROM training_runs WHERE status = ?
                    ORDER BY start_time DESC LIMIT ?
                """, (status.value, limit))
            else:
                cursor = conn.execute("""
                    SELECT * FROM training_runs 
                    ORDER BY start_time DESC LIMIT ?
                """, (limit,))
            
            for row in cursor.fetchall():
                end_time = None
                if row["end_time"]:
                    end_time = datetime.fromisoformat(row["end_time"])
                
                runs.append(TrainingRun(
                    id=row["id"],
                    run_name=row["run_name"],
                    status=TrainingStatus(row["status"]),
                    config=json.loads(row["config"]),
                    gedcom_ids=json.loads(row["gedcom_ids"]),
                    start_time=datetime.fromisoformat(row["start_time"]),
                    end_time=end_time,
                    duration=row["duration"],
                    final_metrics=json.loads(row["final_metrics"]),
                    model_path=row["model_path"],
                    model_version=row["model_version"],
                    error_message=row["error_message"],
                    triggered_by=row["triggered_by"]
                ))
        return runs

    def cleanup_old_cache(self, max_age_days: int = 30) -> int:
        """Clean up old cached graphs.
        
        Args:
            max_age_days: Maximum age of cached graphs in days
            
        Returns:
            Number of cache entries deleted
        """
        with sqlite3.connect(self.db_path) as conn:
            cursor = conn.execute("""
                DELETE FROM graph_cache 
                WHERE created_time < datetime('now', '-{} days')
            """.format(max_age_days))
            conn.commit()
            return cursor.rowcount

    def get_database_stats(self) -> Dict[str, Any]:
        """Get database statistics.
        
        Returns:
            Dictionary containing database statistics
        """
        stats = {}
        with sqlite3.connect(self.db_path) as conn:
            # GEDCOM file counts by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM gedcom_files 
                GROUP BY status
            """)
            stats["gedcom_files"] = dict(cursor.fetchall())
            
            # Training run counts by status
            cursor = conn.execute("""
                SELECT status, COUNT(*) as count 
                FROM training_runs 
                GROUP BY status
            """)
            stats["training_runs"] = dict(cursor.fetchall())
            
            # Cache statistics
            cursor = conn.execute("""
                SELECT COUNT(*) as count, SUM(access_count) as total_accesses
                FROM graph_cache
            """)
            row = cursor.fetchone()
            stats["graph_cache"] = {
                "count": row[0],
                "total_accesses": row[1] or 0
            }
            
            # Database size
            stats["database_size"] = self.db_path.stat().st_size
            
        return stats