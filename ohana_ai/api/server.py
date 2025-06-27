"""
FastAPI server for OhanaAI MLOps API.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
import hashlib
from typing import Dict, Any, List

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, PlainTextResponse
from fastapi.staticfiles import StaticFiles
import uvicorn
from vercel_blob import BlobClient

from ..core.config import OhanaConfig, load_config
from ..graph_builder import GraphBuilder
from .gedcom_parser import parse_gedcom_file
from .models import PredictionRequest, UserCreate, UserLogin, Token, UserResponse, GedcomUploadResponse, GedcomFile
from .auth import APIAuth, require_auth
from fastapi.security import OAuth2PasswordRequestForm
from ..mlops.database import User, GedcomRecord, ProcessingStatus
from ..predictor import OhanaAIPredictor


class OhanaAPIServer:
    """OhanaAI API Server with TensorFlow model."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.model = tf.keras.models.load_model(self.config.paths.models)
        self.graph_builder = GraphBuilder(config_path)
        self.auth = APIAuth()
        self.logger = logging.getLogger(__name__)
        self.blob_client = BlobClient()

    async def startup(self) -> None:
        self.logger.info("Starting OhanaAI API server...")
        self.logger.info("API server started successfully")

    async def shutdown(self) -> None:
        self.logger.info("Shutting down OhanaAI API server...")
        self.logger.info("API server shutdown complete")


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
    @app.post("/register", response_model=UserResponse)
    async def register_user(user_create: UserCreate):
        existing_user = server.auth.db_manager.get_user(user_create.username)
        if existing_user:
            raise HTTPException(status_code=400, detail="Username already registered")
        
        hashed_password = server.auth.get_password_hash(user_create.password)
        new_user = User(username=user_create.username, password_hash=hashed_password)
        user_id = server.auth.db_manager.add_user(new_user)
        new_user.id = user_id
        return new_user

    @app.post("/token", response_model=Token)
    async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
        user = server.auth.authenticate_user(form_data.username, form_data.password)
        if not user:
            raise HTTPException(status_code=400, detail="Incorrect username or password")
        access_token = server.auth.create_access_token(data={"sub": user.username})
        return {"access_token": access_token, "token_type": "bearer"}

    @app.get("/users/me", response_model=UserResponse)
    async def read_users_me(current_user: User = Depends(require_auth)):
        return current_user

    from ..predictor import OhanaAIPredictor

    @app.post("/predict")
    async def predict(
        request: PredictionRequest,
        current_user: User = Depends(require_auth)
    ):
        try:
            predictor = OhanaAIPredictor(config_path=server.config.config_file)
            predictor.load_model(server.config.paths.models)
            
            individuals, families = parse_gedcom_file(request.gedcom_file)
            predictor.prepare_data(individuals, families)
            
            predictions = predictor.predict_missing_parents()
            
            return {"predictions": predictions}
        except Exception as e:
            server.logger.error(f"Prediction error: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.post("/gedcom/upload", response_model=GedcomUploadResponse)
    async def upload_gedcom(
        file: UploadFile = File(...),
        current_user: User = Depends(require_auth)
    ):
        if not file.filename.lower().endswith(('.ged', '.gedcom')):
            raise HTTPException(status_code=400, detail="Only GEDCOM files (.ged, .gedcom) are allowed")
        
        file_content = await file.read()
        file_hash = hashlib.sha256(file_content).hexdigest()
        
        # Upload to Vercel Blob
        blob_url = await server.blob_client.upload(file.filename, file_content, {'access': 'public'})
        
        # Save metadata to DB
        gedcom_record = GedcomRecord(
            user_id=current_user.id,
            filename=file.filename,
            original_filename=file.filename,
            file_hash=file_hash,
            file_size=len(file_content),
            status=ProcessingStatus.UPLOADED,
            metadata={"blob_url": blob_url}
        )
        file_id = server.auth.db_manager.add_gedcom_file(gedcom_record)
        
        return {"message": "File uploaded successfully", "file_id": file_id, "filename": file.filename, "size": len(file_content), "blob_url": blob_url}

    @app.get("/gedcom/list", response_model=List[GedcomFile])
    async def list_gedcom_files(current_user: User = Depends(require_auth)):
        files = server.auth.db_manager.get_all_gedcom_files(current_user.id)
        return [
            GedcomFile(
                id=f.id,
                filename=f.filename,
                upload_time=f.upload_time.isoformat(),
                status=f.status.value,
                blob_url=f.metadata.get("blob_url")
            ) for f in files
        ]

    @app.get("/gedcom/{file_id}")
    async def get_gedcom_file_content(file_id: int, current_user: User = Depends(require_auth)):
        gedcom_record = server.auth.db_manager.get_gedcom_file(file_id, current_user.id)
        if not gedcom_record:
            raise HTTPException(status_code=404, detail="GEDCOM file not found")
        
        blob_url = gedcom_record.metadata.get("blob_url")
        if not blob_url:
            raise HTTPException(status_code=404, detail="GEDCOM file content not found in blob storage")

        try:
            # Fetch content from Vercel Blob
            file_content = await server.blob_client.download(blob_url)
            return PlainTextResponse(file_content.decode('utf-8'))
        except Exception as e:
            server.logger.error(f"Error fetching GEDCOM from blob: {e}")
            raise HTTPException(status_code=500, detail="Failed to retrieve GEDCOM file content")

    @app.delete("/gedcom/delete_all")
    async def delete_all_gedcom_files(current_user: User = Depends(require_auth)):
        try:
            deleted_blob_urls = server.auth.db_manager.delete_gedcom_files_for_user(current_user.id)
            for blob_url in deleted_blob_urls:
                await server.blob_client.delete(blob_url)
            return {"message": f"Successfully deleted {len(deleted_blob_urls)} GEDCOM files."}
        except Exception as e:
            server.logger.error(f"Error deleting GEDCOM files: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.delete("/users/delete_me")
    async def delete_my_account(current_user: User = Depends(require_auth)):
        try:
            # Delete all GEDCOM files associated with the user
            deleted_blob_urls = server.auth.db_manager.delete_gedcom_files_for_user(current_user.id)
            for blob_url in deleted_blob_urls:
                await server.blob_client.delete(blob_url)
            
            # Delete the user account from the database
            server.auth.db_manager.delete_user(current_user.id)
            
            return {"message": "Account and all associated data deleted successfully."}
        except Exception as e:
            server.logger.error(f"Error deleting user account: {e}")
            raise HTTPException(status_code=500, detail=str(e))

    @app.get("/gedcom/{file_id}/filled", response_class=PlainTextResponse)
    async def get_filled_gedcom_file(file_id: int, current_user: User = Depends(require_auth)):
        gedcom_record = server.auth.db_manager.get_gedcom_file(file_id, current_user.id)
        if not gedcom_record:
            raise HTTPException(status_code=404, detail="GEDCOM file not found")
        
        blob_url = gedcom_record.metadata.get("blob_url")
        if not blob_url:
            raise HTTPException(status_code=404, detail="GEDCOM file content not found in blob storage")

        try:
            original_gedcom_content = await server.blob_client.download(blob_url)
            original_gedcom_content = original_gedcom_content.decode('utf-8')

            # Predict missing parents
            predictor = OhanaAIPredictor(config_path=server.config.config_file)
            predictor.load_model(server.config.paths.models)
            predictions = predictor.predict_parents(original_gedcom_content)

            # Generate filled GEDCOM content
            filled_gedcom_content = predictor.export_predictions_gedcom_string(predictions, original_gedcom_content)
            
            return PlainTextResponse(filled_gedcom_content, media_type="text/plain")
        except Exception as e:
            server.logger.error(f"Error generating filled GEDCOM: {e}")
            raise HTTPException(status_code=500, detail=str(e))


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