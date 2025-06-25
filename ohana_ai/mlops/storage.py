"""
Storage management for GEDCOM files and models.
Supports local filesystem, AWS S3, and Google Cloud Storage.
"""

import asyncio
import logging
import shutil
from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

# Optional cloud storage imports
try:
    import boto3
    from botocore.exceptions import ClientError
    HAS_BOTO3 = True
except ImportError:
    HAS_BOTO3 = False

try:
    from google.cloud import storage as gcs
    HAS_GCS = True
except ImportError:
    HAS_GCS = False


class StorageBackend(ABC):
    """Abstract base class for storage backends."""

    @abstractmethod
    async def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload a file to storage.
        
        Args:
            local_path: Local file path
            remote_key: Remote storage key/path
            
        Returns:
            URL or path to uploaded file
        """
        pass

    @abstractmethod
    async def download_file(self, remote_key: str, local_path: Path) -> bool:
        """Download a file from storage.
        
        Args:
            remote_key: Remote storage key/path
            local_path: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def delete_file(self, remote_key: str) -> bool:
        """Delete a file from storage.
        
        Args:
            remote_key: Remote storage key/path
            
        Returns:
            True if successful, False otherwise
        """
        pass

    @abstractmethod
    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in storage.
        
        Args:
            prefix: Prefix to filter files
            
        Returns:
            List of file keys/paths
        """
        pass

    @abstractmethod
    async def file_exists(self, remote_key: str) -> bool:
        """Check if a file exists in storage.
        
        Args:
            remote_key: Remote storage key/path
            
        Returns:
            True if file exists, False otherwise
        """
        pass


class LocalStorageBackend(StorageBackend):
    """Local filesystem storage backend."""

    def __init__(self, base_path: str = "storage"):
        """Initialize local storage backend.
        
        Args:
            base_path: Base directory for storage
        """
        self.base_path = Path(base_path)
        self.base_path.mkdir(parents=True, exist_ok=True)
        self.logger = logging.getLogger(__name__)

    async def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload file to local storage."""
        dest_path = self.base_path / remote_key
        dest_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        await asyncio.get_event_loop().run_in_executor(
            None, shutil.copy2, str(local_path), str(dest_path)
        )
        
        self.logger.info(f"Uploaded {local_path} to {dest_path}")
        return str(dest_path)

    async def download_file(self, remote_key: str, local_path: Path) -> bool:
        """Download file from local storage."""
        source_path = self.base_path / remote_key
        
        if not source_path.exists():
            return False
        
        local_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Copy file
        await asyncio.get_event_loop().run_in_executor(
            None, shutil.copy2, str(source_path), str(local_path)
        )
        
        self.logger.info(f"Downloaded {source_path} to {local_path}")
        return True

    async def delete_file(self, remote_key: str) -> bool:
        """Delete file from local storage."""
        file_path = self.base_path / remote_key
        
        if file_path.exists():
            await asyncio.get_event_loop().run_in_executor(
                None, file_path.unlink
            )
            self.logger.info(f"Deleted {file_path}")
            return True
        
        return False

    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in local storage."""
        prefix_path = self.base_path / prefix if prefix else self.base_path
        
        if not prefix_path.exists():
            return []
        
        files = []
        for file_path in prefix_path.rglob("*"):
            if file_path.is_file():
                relative_path = file_path.relative_to(self.base_path)
                files.append(str(relative_path))
        
        return sorted(files)

    async def file_exists(self, remote_key: str) -> bool:
        """Check if file exists in local storage."""
        file_path = self.base_path / remote_key
        return file_path.exists()


class S3StorageBackend(StorageBackend):
    """AWS S3 storage backend."""

    def __init__(self, bucket_name: str, aws_access_key_id: Optional[str] = None,
                 aws_secret_access_key: Optional[str] = None, region: str = "us-east-1"):
        """Initialize S3 storage backend.
        
        Args:
            bucket_name: S3 bucket name
            aws_access_key_id: AWS access key ID (optional, uses default credentials)
            aws_secret_access_key: AWS secret access key (optional)
            region: AWS region
        """
        if not HAS_BOTO3:
            raise ImportError("boto3 is required for S3 storage backend")
        
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize S3 client
        session = boto3.Session(
            aws_access_key_id=aws_access_key_id,
            aws_secret_access_key=aws_secret_access_key,
            region_name=region
        )
        self.s3_client = session.client("s3")

    async def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload file to S3."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.upload_file, str(local_path), self.bucket_name, remote_key
            )
            
            url = f"s3://{self.bucket_name}/{remote_key}"
            self.logger.info(f"Uploaded {local_path} to {url}")
            return url
            
        except ClientError as e:
            self.logger.error(f"Failed to upload {local_path} to S3: {e}")
            raise

    async def download_file(self, remote_key: str, local_path: Path) -> bool:
        """Download file from S3."""
        try:
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.download_file, self.bucket_name, remote_key, str(local_path)
            )
            
            self.logger.info(f"Downloaded s3://{self.bucket_name}/{remote_key} to {local_path}")
            return True
            
        except ClientError as e:
            self.logger.error(f"Failed to download {remote_key} from S3: {e}")
            return False

    async def delete_file(self, remote_key: str) -> bool:
        """Delete file from S3."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.delete_object, {"Bucket": self.bucket_name, "Key": remote_key}
            )
            
            self.logger.info(f"Deleted s3://{self.bucket_name}/{remote_key}")
            return True
            
        except ClientError as e:
            self.logger.error(f"Failed to delete {remote_key} from S3: {e}")
            return False

    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in S3."""
        try:
            response = await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.list_objects_v2, {"Bucket": self.bucket_name, "Prefix": prefix}
            )
            
            files = []
            if "Contents" in response:
                files = [obj["Key"] for obj in response["Contents"]]
            
            return sorted(files)
            
        except ClientError as e:
            self.logger.error(f"Failed to list files in S3: {e}")
            return []

    async def file_exists(self, remote_key: str) -> bool:
        """Check if file exists in S3."""
        try:
            await asyncio.get_event_loop().run_in_executor(
                None, self.s3_client.head_object, {"Bucket": self.bucket_name, "Key": remote_key}
            )
            return True
            
        except ClientError:
            return False


class GCSStorageBackend(StorageBackend):
    """Google Cloud Storage backend."""

    def __init__(self, bucket_name: str, credentials_path: Optional[str] = None):
        """Initialize GCS storage backend.
        
        Args:
            bucket_name: GCS bucket name
            credentials_path: Path to service account credentials JSON file
        """
        if not HAS_GCS:
            raise ImportError("google-cloud-storage is required for GCS backend")
        
        self.bucket_name = bucket_name
        self.logger = logging.getLogger(__name__)
        
        # Initialize GCS client
        if credentials_path:
            self.client = gcs.Client.from_service_account_json(credentials_path)
        else:
            self.client = gcs.Client()
        
        self.bucket = self.client.bucket(bucket_name)

    async def upload_file(self, local_path: Path, remote_key: str) -> str:
        """Upload file to GCS."""
        try:
            blob = self.bucket.blob(remote_key)
            
            await asyncio.get_event_loop().run_in_executor(
                None, blob.upload_from_filename, str(local_path)
            )
            
            url = f"gs://{self.bucket_name}/{remote_key}"
            self.logger.info(f"Uploaded {local_path} to {url}")
            return url
            
        except Exception as e:
            self.logger.error(f"Failed to upload {local_path} to GCS: {e}")
            raise

    async def download_file(self, remote_key: str, local_path: Path) -> bool:
        """Download file from GCS."""
        try:
            blob = self.bucket.blob(remote_key)
            local_path.parent.mkdir(parents=True, exist_ok=True)
            
            await asyncio.get_event_loop().run_in_executor(
                None, blob.download_to_filename, str(local_path)
            )
            
            self.logger.info(f"Downloaded gs://{self.bucket_name}/{remote_key} to {local_path}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to download {remote_key} from GCS: {e}")
            return False

    async def delete_file(self, remote_key: str) -> bool:
        """Delete file from GCS."""
        try:
            blob = self.bucket.blob(remote_key)
            
            await asyncio.get_event_loop().run_in_executor(
                None, blob.delete
            )
            
            self.logger.info(f"Deleted gs://{self.bucket_name}/{remote_key}")
            return True
            
        except Exception as e:
            self.logger.error(f"Failed to delete {remote_key} from GCS: {e}")
            return False

    async def list_files(self, prefix: str = "") -> List[str]:
        """List files in GCS."""
        try:
            blobs = await asyncio.get_event_loop().run_in_executor(
                None, list, self.bucket.list_blobs(prefix=prefix)
            )
            
            files = [blob.name for blob in blobs]
            return sorted(files)
            
        except Exception as e:
            self.logger.error(f"Failed to list files in GCS: {e}")
            return []

    async def file_exists(self, remote_key: str) -> bool:
        """Check if file exists in GCS."""
        try:
            blob = self.bucket.blob(remote_key)
            return await asyncio.get_event_loop().run_in_executor(
                None, blob.exists
            )
            
        except Exception:
            return False


class StorageManager:
    """Manages file storage with multiple backend support."""

    def __init__(self, backend_type: str = "local", **backend_kwargs):
        """Initialize storage manager.
        
        Args:
            backend_type: Storage backend type (local, s3, gcs)
            **backend_kwargs: Backend-specific configuration
        """
        self.backend_type = backend_type
        self.logger = logging.getLogger(__name__)
        
        # Initialize backend
        if backend_type == "local":
            self.backend = LocalStorageBackend(**backend_kwargs)
        elif backend_type == "s3":
            self.backend = S3StorageBackend(**backend_kwargs)
        elif backend_type == "gcs":
            self.backend = GCSStorageBackend(**backend_kwargs)
        else:
            raise ValueError(f"Unsupported storage backend: {backend_type}")

    async def store_gedcom_file(self, local_path: Path, file_hash: str) -> str:
        """Store a GEDCOM file.
        
        Args:
            local_path: Local path to GEDCOM file
            file_hash: File hash for unique storage key
            
        Returns:
            Storage URL/path
        """
        remote_key = f"gedcom/{file_hash[:2]}/{file_hash}.ged"
        return await self.backend.upload_file(local_path, remote_key)

    async def store_model(self, local_path: Path, model_version: str) -> str:
        """Store a trained model.
        
        Args:
            local_path: Local path to model file
            model_version: Model version identifier
            
        Returns:
            Storage URL/path
        """
        remote_key = f"models/{model_version}/model.npz"
        return await self.backend.upload_file(local_path, remote_key)

    async def retrieve_gedcom_file(self, file_hash: str, local_path: Path) -> bool:
        """Retrieve a GEDCOM file.
        
        Args:
            file_hash: File hash
            local_path: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        remote_key = f"gedcom/{file_hash[:2]}/{file_hash}.ged"
        return await self.backend.download_file(remote_key, local_path)

    async def retrieve_model(self, model_version: str, local_path: Path) -> bool:
        """Retrieve a trained model.
        
        Args:
            model_version: Model version identifier
            local_path: Local destination path
            
        Returns:
            True if successful, False otherwise
        """
        remote_key = f"models/{model_version}/model.npz"
        return await self.backend.download_file(remote_key, local_path)

    async def list_models(self) -> List[str]:
        """List available model versions.
        
        Returns:
            List of model version identifiers
        """
        files = await self.backend.list_files("models/")
        
        # Extract version identifiers
        versions = []
        for file_path in files:
            parts = file_path.split("/")
            if len(parts) >= 2 and parts[1] not in versions:
                versions.append(parts[1])
        
        return sorted(versions, reverse=True)

    async def cleanup_old_files(self, max_age_days: int = 90) -> int:
        """Clean up old files from storage.
        
        Args:
            max_age_days: Maximum age of files to keep
            
        Returns:
            Number of files deleted
        """
        # This would implement cleanup logic based on file metadata
        # For now, return 0 as placeholder
        self.logger.info(f"Cleanup requested for files older than {max_age_days} days")
        return 0

    async def get_storage_stats(self) -> Dict[str, Any]:
        """Get storage statistics.
        
        Returns:
            Dictionary containing storage statistics
        """
        gedcom_files = await self.backend.list_files("gedcom/")
        model_files = await self.backend.list_files("models/")
        
        return {
            "backend_type": self.backend_type,
            "gedcom_files_count": len(gedcom_files),
            "model_files_count": len(model_files),
            "total_files": len(gedcom_files) + len(model_files)
        }