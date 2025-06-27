"""
FastAPI server for OhanaAI MLOps API.
"""

import asyncio
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Dict, Any

import numpy as np
import tensorflow as tf
from fastapi import FastAPI, HTTPException, UploadFile, File, Depends, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
import uvicorn

from ..core.config import OhanaConfig, load_config
from ..graph_builder import GraphBuilder
from ..gedcom_parser import parse_gedcom_file
from .models import *
from .auth import APIAuth, require_auth
from ..gedcom_parser import parse_gedcom_file


class OhanaAPIServer:
    """OhanaAI API Server with TensorFlow model."""
    
    def __init__(self, config_path: str = "config.yaml"):
        self.config = load_config(config_path)
        self.model = tf.keras.models.load_model(self.config.paths.models)
        self.graph_builder = GraphBuilder(config_path)
        self.auth = APIAuth()
        self.logger = logging.getLogger(__name__)

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
    @app.post("/predict")
    async def predict(
        request: PredictionRequest,
        auth_user: dict = Depends(require_auth)
    ):
        try:
            individuals, families = parse_gedcom_file(request.gedcom_file)
            graph_data = server.graph_builder.build_graph(individuals, families)
            
            predictions = server.model.predict([graph_data.node_features, graph_data.edge_index, graph_data.edge_types, np.array(request.candidate_pairs)])
            
            return {"predictions": predictions.tolist()}
        except Exception as e:
            server.logger.error(f"Prediction error: {e}")
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