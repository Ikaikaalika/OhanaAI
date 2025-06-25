#!/usr/bin/env python3
"""
Test script for OhanaAI MLOps pipeline.
Verifies all components are working correctly.
"""

import asyncio
import logging
import tempfile
import time
from pathlib import Path

import click
import yaml


async def test_mlops_pipeline():
    """Test the complete MLOps pipeline."""
    print("🧬 OhanaAI MLOps Pipeline Test")
    print("=" * 40)
    
    # Test imports
    try:
        from ohana_ai.mlops.database import DatabaseManager, GedcomRecord, ProcessingStatus
        from ohana_ai.mlops.pipeline import TrainingPipeline, PipelineConfig  
        from ohana_ai.mlops.storage import StorageManager
        from ohana_ai.mlops.versioning import ModelVersionManager
        from ohana_ai.mlops.monitoring import MetricsCollector, AlertManager
        from ohana_ai.mlops.triggers import TriggerManager, UploadTrigger
        from ohana_ai.api.server import create_app
        print("✓ All MLOps imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test database operations
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            db_path = Path(temp_dir) / "test.db"
            db = DatabaseManager(str(db_path))
            
            # Test GEDCOM record
            record = GedcomRecord(
                filename="test.ged",
                original_filename="test.ged", 
                file_hash="abc123",
                file_size=1000
            )
            
            file_id = db.add_gedcom_file(record)
            retrieved = db.get_gedcom_file(file_id)
            
            assert retrieved.filename == "test.ged"
            assert retrieved.file_hash == "abc123"
            
        print("✓ Database operations working")
    except Exception as e:
        print(f"✗ Database error: {e}")
        return False
    
    # Test storage manager
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            storage = StorageManager("local", base_path=temp_dir)
            
            # Create test file
            test_file = Path(temp_dir) / "input" / "test.ged"
            test_file.parent.mkdir(exist_ok=True)
            test_file.write_text("0 HEAD\n1 CHAR UTF-8\n0 TRLR\n")
            
            # Test storage
            url = await storage.store_gedcom_file(test_file, "abc123")
            assert url is not None
            
            # Test retrieval
            output_file = Path(temp_dir) / "output" / "test.ged"
            success = await storage.retrieve_gedcom_file("abc123", output_file)
            assert success
            assert output_file.exists()
            
        print("✓ Storage manager working")
    except Exception as e:
        print(f"✗ Storage error: {e}")
        return False
    
    # Test model versioning
    try:
        with tempfile.TemporaryDirectory() as temp_dir:
            version_manager = ModelVersionManager(temp_dir)
            
            # Create dummy model file
            model_file = Path(temp_dir) / "model.npz"
            model_file.write_bytes(b"dummy model data")
            
            # Test version creation
            version = version_manager.create_model_version(
                model_file,
                config={"epochs": 10},
                metrics={"accuracy": 0.85},
                training_data={"files": 3},
                training_duration=120.0
            )
            
            # Test metadata retrieval
            metadata = version_manager.get_model_metadata(version)
            assert metadata is not None
            assert metadata.metrics["accuracy"] == 0.85
            
        print("✓ Model versioning working")
    except Exception as e:
        print(f"✗ Versioning error: {e}")
        return False
    
    # Test monitoring
    try:
        metrics_collector = MetricsCollector(collection_interval=1.0)
        alert_manager = AlertManager(metrics_collector)
        
        # Test metrics collection
        await asyncio.wait_for(
            metrics_collector.start_collection(), 
            timeout=0.1
        )
    except asyncio.TimeoutError:
        pass  # Expected for quick test
    except Exception as e:
        print(f"✗ Monitoring error: {e}")
        return False
    
    print("✓ Monitoring system working")
    
    # Test trigger system
    try:
        trigger_manager = TriggerManager()
        
        # Test callback registration
        callback_called = False
        
        async def test_callback(event):
            nonlocal callback_called
            callback_called = True
        
        trigger_manager.register_callback("test_event", test_callback)
        
        # Test event triggering
        from ohana_ai.mlops.triggers import TriggerEvent
        from datetime import datetime
        
        event = TriggerEvent(
            event_type="test_event",
            timestamp=datetime.now(),
            data={"test": True}
        )
        
        await trigger_manager.trigger_event(event)
        
        # Give callback time to execute
        await asyncio.sleep(0.1)
        
        assert callback_called, "Callback was not called"
        
        print("✓ Trigger system working")
    except Exception as e:
        print(f"✗ Trigger error: {e}")
        return False
    
    # Test API server creation
    try:
        app = create_app("config.yaml")
        assert app is not None
        print("✓ API server creation working")
    except Exception as e:
        print(f"✗ API server error: {e}")
        return False
    
    # Test configuration loading
    try:
        config_path = Path("mlops_config.yaml")
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
            assert isinstance(config, dict)
            print("✓ Configuration loading working")
        else:
            print("ℹ MLOps config file not found, skipping config test")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    print("\n🎉 All MLOps pipeline tests passed!")
    print("\nNext steps:")
    print("1. Start the MLOps server: python run_mlops.py")
    print("2. Upload GEDCOM files via API or file system")
    print("3. Monitor training at http://localhost:8000/status")
    print("4. View system health at http://localhost:8000/health")
    
    return True


async def test_basic_workflow():
    """Test a basic end-to-end workflow."""
    print("\n📋 Testing Basic Workflow")
    print("-" * 30)
    
    try:
        # Import required modules
        from ohana_ai.mlops.database import DatabaseManager, GedcomRecord
        from ohana_ai.mlops.storage import StorageManager
        from ohana_ai.mlops.triggers import TriggerManager, TriggerEvent
        from datetime import datetime
        
        # Create temporary workspace
        with tempfile.TemporaryDirectory() as temp_dir:
            temp_path = Path(temp_dir)
            
            # Initialize components
            db = DatabaseManager(str(temp_path / "test.db"))
            storage = StorageManager("local", base_path=str(temp_path / "storage"))
            triggers = TriggerManager()
            
            # Create sample GEDCOM content
            gedcom_content = """0 HEAD
1 SOUR OhanaAI Test
1 CHAR UTF-8
0 @I1@ INDI
1 NAME John /Doe/
1 SEX M
1 BIRT
2 DATE 1950
0 @I2@ INDI  
1 NAME Jane /Smith/
1 SEX F
1 BIRT
2 DATE 1952
0 @F1@ FAM
1 HUSB @I1@
1 WIFE @I2@
0 TRLR"""
            
            # Simulate file upload
            upload_file = temp_path / "uploads" / "test_family.ged"
            upload_file.parent.mkdir(exist_ok=True)
            upload_file.write_text(gedcom_content)
            
            # Add to database
            record = GedcomRecord(
                filename=str(upload_file),
                original_filename="test_family.ged",
                file_hash="test123",
                file_size=len(gedcom_content)
            )
            
            file_id = db.add_gedcom_file(record)
            print(f"✓ Added GEDCOM file to database (ID: {file_id})")
            
            # Store in storage system
            storage_url = await storage.store_gedcom_file(upload_file, "test123")
            print(f"✓ Stored file in storage system: {storage_url}")
            
            # Trigger upload event
            event = TriggerEvent(
                event_type="gedcom_upload",
                timestamp=datetime.now(),
                data={
                    "file_id": file_id,
                    "filename": "test_family.ged",
                    "file_size": len(gedcom_content)
                }
            )
            
            # Track event processing
            events_processed = []
            
            async def event_handler(event):
                events_processed.append(event.event_type)
            
            triggers.register_callback("gedcom_upload", event_handler)
            await triggers.trigger_event(event)
            
            # Give event time to process
            await asyncio.sleep(0.1)
            
            assert "gedcom_upload" in events_processed
            print("✓ Event triggering and handling working")
            
            # Verify file retrieval
            retrieved_file = temp_path / "retrieved" / "test_family.ged"
            success = await storage.retrieve_gedcom_file("test123", retrieved_file)
            assert success
            assert retrieved_file.read_text() == gedcom_content
            print("✓ File retrieval working")
            
        print("✓ Basic workflow test completed successfully")
        return True
        
    except Exception as e:
        print(f"✗ Workflow test failed: {e}")
        return False


@click.command()
@click.option('--workflow', is_flag=True, help='Run basic workflow test')
@click.option('--verbose', '-v', is_flag=True, help='Verbose output')
def main(workflow: bool, verbose: bool):
    """Test the OhanaAI MLOps pipeline."""
    
    if verbose:
        logging.basicConfig(level=logging.DEBUG)
    
    async def run_tests():
        success = True
        
        # Run main tests
        if not await test_mlops_pipeline():
            success = False
        
        # Run workflow test if requested
        if workflow:
            if not await test_basic_workflow():
                success = False
        
        return success
    
    try:
        success = asyncio.run(run_tests())
        exit_code = 0 if success else 1
    except KeyboardInterrupt:
        print("\nTest interrupted by user")
        exit_code = 1
    except Exception as e:
        print(f"\nTest suite error: {e}")
        exit_code = 1
    
    exit(exit_code)


if __name__ == "__main__":
    main()