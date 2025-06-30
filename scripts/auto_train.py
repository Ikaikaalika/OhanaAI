#!/usr/bin/env python3
"""
Automated Training Pipeline for Ohana AI
Fetches new GEDCOM data from the web app and retrains the model
Designed for M1 Mac with automated deployment
"""

import os
import sys
import json
import requests
import subprocess
import shutil
from datetime import datetime
from pathlib import Path
import logging

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class OhanaTrainingPipeline:
    def __init__(self, config_file='training_config.json'):
        self.config = self.load_config(config_file)
        self.project_root = Path(__file__).parent.parent
        self.models_dir = self.project_root / 'models' / 'parent_predictor'
        
    def load_config(self, config_file):
        """Load training configuration"""
        default_config = {
            'web_app_url': 'https://your-app.vercel.app',  # Replace with your URL
            'api_key': os.getenv('ML_EXPORT_API_KEY', 'your-secret-key'),
            'min_new_examples': 50,  # Minimum new examples to trigger retraining
            'backup_models': True,
            'auto_deploy': False,  # Set to True for automatic deployment
            'notification_webhook': None,  # Optional Slack/Discord webhook
            'training_script': 'train_model_m1.py'
        }
        
        try:
            if os.path.exists(config_file):
                with open(config_file, 'r') as f:
                    user_config = json.load(f)
                default_config.update(user_config)
        except Exception as e:
            logger.warning(f"Could not load config file: {e}")
            
        return default_config
    
    def check_for_new_data(self):
        """Check if new training data is available"""
        logger.info("Checking for new training data...")
        
        try:
            response = requests.post(
                f"{self.config['web_app_url']}/api/ml/export-user-data",
                json={
                    'apiKey': self.config['api_key'],
                    'includeMetadata': True
                },
                timeout=30
            )
            
            if response.status_code == 200:
                data = response.json()
                return data
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to check for new data: {e}")
            return None
    
    def download_training_data(self, webhook_url, filename):
        """Download the training data file"""
        logger.info(f"Downloading training data: {filename}")
        
        try:
            response = requests.get(webhook_url, timeout=60)
            
            if response.status_code == 200:
                # Save to local directory
                data_dir = self.project_root / 'training_data'
                data_dir.mkdir(exist_ok=True)
                
                filepath = data_dir / filename
                with open(filepath, 'w') as f:
                    f.write(response.text)
                
                logger.info(f"Downloaded training data to: {filepath}")
                return filepath
            else:
                logger.error(f"Download failed: {response.status_code}")
                return None
                
        except Exception as e:
            logger.error(f"Failed to download data: {e}")
            return None
    
    def backup_current_model(self):
        """Backup the current model before retraining"""
        if not self.config['backup_models']:
            return
            
        logger.info("Backing up current model...")
        
        try:
            backup_dir = self.project_root / 'model_backups'
            backup_dir.mkdir(exist_ok=True)
            
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            backup_path = backup_dir / f"model_backup_{timestamp}"
            
            if self.models_dir.exists():
                shutil.copytree(self.models_dir, backup_path)
                logger.info(f"Model backed up to: {backup_path}")
            
        except Exception as e:
            logger.error(f"Failed to backup model: {e}")
    
    def run_training(self, data_file):
        """Run the training script with new data"""
        logger.info("Starting model training...")
        
        try:
            # Ensure we're in the right directory
            os.chdir(self.project_root)
            
            # Run the training script
            cmd = [
                sys.executable,
                self.config['training_script'],
                '--data-file', str(data_file),
                '--output-dir', str(self.models_dir)
            ]
            
            logger.info(f"Running command: {' '.join(cmd)}")
            
            result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=3600  # 1 hour timeout
            )
            
            if result.returncode == 0:
                logger.info("Training completed successfully!")
                logger.info(f"Training output: {result.stdout}")
                return True
            else:
                logger.error(f"Training failed with code {result.returncode}")
                logger.error(f"Error output: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            logger.error("Training timed out after 1 hour")
            return False
        except Exception as e:
            logger.error(f"Failed to run training: {e}")
            return False
    
    def convert_to_tensorflowjs(self):
        """Convert the trained model to TensorFlow.js format"""
        logger.info("Converting model to TensorFlow.js format...")
        
        try:
            model_file = self.models_dir / 'ohana_model_m1.h5'
            if not model_file.exists():
                logger.error(f"Model file not found: {model_file}")
                return False
            
            # Convert using tensorflowjs_converter
            cmd = [
                'tensorflowjs_converter',
                '--input_format=keras',
                str(model_file),
                str(self.models_dir)
            ]
            
            result = subprocess.run(cmd, capture_output=True, text=True)
            
            if result.returncode == 0:
                logger.info("Model conversion completed!")
                return True
            else:
                logger.error(f"Conversion failed: {result.stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Failed to convert model: {e}")
            return False
    
    def deploy_model(self):
        """Deploy the updated model (if auto_deploy is enabled)"""
        if not self.config['auto_deploy']:
            logger.info("Auto-deploy disabled. Manual deployment required.")
            return True
            
        logger.info("Deploying updated model...")
        
        try:
            # You can implement various deployment strategies here:
            
            # Option 1: Copy to a shared directory that your web app monitors
            # deploy_dir = Path('/shared/models')
            # shutil.copytree(self.models_dir, deploy_dir / 'parent_predictor', dirs_exist_ok=True)
            
            # Option 2: Upload to cloud storage (S3, GCS, etc.)
            # self.upload_to_cloud_storage()
            
            # Option 3: Trigger a webhook to notify your web app
            self.notify_model_update()
            
            logger.info("Model deployment completed!")
            return True
            
        except Exception as e:
            logger.error(f"Failed to deploy model: {e}")
            return False
    
    def notify_model_update(self):
        """Notify about model update via webhook"""
        if not self.config['notification_webhook']:
            return
            
        try:
            payload = {
                'text': f"🤖 Ohana AI model updated at {datetime.now().isoformat()}",
                'timestamp': datetime.now().isoformat(),
                'model_path': str(self.models_dir)
            }
            
            requests.post(
                self.config['notification_webhook'],
                json=payload,
                timeout=10
            )
            
        except Exception as e:
            logger.warning(f"Failed to send notification: {e}")
    
    def update_training_status(self, status, message):
        """Update the web app about training status"""
        try:
            requests.post(
                f"{self.config['web_app_url']}/api/ml/training-status",
                json={
                    'apiKey': self.config['api_key'],
                    'status': status,
                    'message': message,
                    'timestamp': datetime.now().isoformat()
                },
                timeout=10
            )
        except Exception as e:
            logger.warning(f"Failed to update training status: {e}")
    
    def run_pipeline(self):
        """Run the complete training pipeline"""
        logger.info("=== Ohana AI Automated Training Pipeline ===")
        logger.info(f"Started at: {datetime.now()}")
        
        try:
            # 1. Check for new data
            data_info = self.check_for_new_data()
            
            if not data_info:
                logger.info("No new data available or API error")
                return False
            
            if data_info['count'] < self.config['min_new_examples']:
                logger.info(f"Only {data_info['count']} new examples available. "
                           f"Minimum {self.config['min_new_examples']} required.")
                return False
            
            logger.info(f"Found {data_info['count']} new training examples!")
            
            # 2. Download training data
            data_file = self.download_training_data(
                data_info['webhookUrl'], 
                data_info['filename']
            )
            
            if not data_file:
                logger.error("Failed to download training data")
                return False
            
            # 3. Backup current model
            self.backup_current_model()
            
            # 4. Update status
            self.update_training_status('training', 'Starting model training...')
            
            # 5. Run training
            training_success = self.run_training(data_file)
            
            if not training_success:
                self.update_training_status('failed', 'Model training failed')
                return False
            
            # 6. Convert to TensorFlow.js
            conversion_success = self.convert_to_tensorflowjs()
            
            if not conversion_success:
                self.update_training_status('failed', 'Model conversion failed')
                return False
            
            # 7. Deploy model
            deploy_success = self.deploy_model()
            
            if deploy_success:
                self.update_training_status('completed', f'Model updated with {data_info["count"]} new examples')
                logger.info("🎉 Training pipeline completed successfully!")
            else:
                self.update_training_status('deploy_failed', 'Training completed but deployment failed')
                logger.warning("Training completed but deployment failed")
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed: {e}")
            self.update_training_status('error', f'Pipeline error: {str(e)}')
            return False
        
        finally:
            logger.info(f"Pipeline finished at: {datetime.now()}")

def main():
    """Main entry point"""
    pipeline = OhanaTrainingPipeline()
    
    # Check if this is a manual run or scheduled
    if len(sys.argv) > 1 and sys.argv[1] == '--manual':
        logger.info("Running manual training pipeline...")
    else:
        logger.info("Running scheduled training pipeline...")
    
    success = pipeline.run_pipeline()
    
    if success:
        logger.info("✅ Training pipeline completed successfully!")
        sys.exit(0)
    else:
        logger.error("❌ Training pipeline failed!")
        sys.exit(1)

if __name__ == "__main__":
    main()