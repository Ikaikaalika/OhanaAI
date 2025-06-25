# OhanaAI MLOps Pipeline

**Continuous Learning System for Genealogical Parent Prediction**

This MLOps pipeline provides automated training, model versioning, and deployment for the OhanaAI genealogical prediction system. When users upload GEDCOM files, the system automatically processes them, builds graph representations, and triggers training runs to continuously improve the model.

## 🏗️ Architecture Overview

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   File Upload   │───▶│  Processing      │───▶│   Training      │
│   (GEDCOM)      │    │  Pipeline        │    │   Pipeline      │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         ▼                       ▼                       ▼
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   Storage       │    │   Graph Cache    │    │   Model         │
│   Manager       │    │   Database       │    │   Versioning    │
└─────────────────┘    └──────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 ▼
                    ┌─────────────────────┐
                    │   Monitoring &      │
                    │   Alerting          │
                    └─────────────────────┘
```

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements-mlops.txt
```

### 2. Configure the System

Edit `mlops_config.yaml` to customize:
- API server settings
- Storage backend (local/S3/GCS)
- Training parameters
- Monitoring thresholds

### 3. Start the MLOps Server

```bash
python run_mlops.py --config mlops_config.yaml
```

This starts:
- **API Server** (default: http://localhost:8000)
- **File Upload Monitoring**
- **Training Pipeline**
- **System Monitoring**
- **Webhook Endpoints**

### 4. Upload GEDCOM Files

**Via Web Interface:**
```bash
curl -X POST "http://localhost:8000/upload" \
  -H "Authorization: Bearer your_token" \
  -F "file=@family.ged"
```

**Via File System:**
Drop `.ged` files into the `uploads/` directory - they'll be processed automatically.

## 📡 API Endpoints

### Authentication
```bash
# Get access token (default credentials)
curl -X POST "http://localhost:8000/auth/login" \
  -d "username=admin&password=admin123"
```

### File Management
```bash
# Upload GEDCOM file
POST /upload

# List files
GET /files?status=processed

# Get file details
GET /files/{file_id}
```

### Training Management
```bash
# Start training run
POST /training/start
{
  "gedcom_ids": [1, 2, 3],
  "config_overrides": {"epochs": 50}
}

# List training runs
GET /training/runs?status=running

# Get training details
GET /training/runs/{run_id}
```

### Model Management
```bash
# List model versions
GET /models

# Get model details
GET /models/{version}

# Promote to production
POST /models/{version}/promote
```

### Monitoring
```bash
# System health
GET /health

# Detailed status
GET /status

# System metrics
GET /metrics?minutes=60

# Active alerts
GET /alerts
```

## 🔄 Automated Workflows

### 1. File Upload → Processing

1. **File Detection**: Monitor `uploads/` directory
2. **Validation**: Check file format and integrity
3. **Parsing**: Extract individuals and families from GEDCOM
4. **Graph Building**: Create graph representation
5. **Storage**: Cache processed data in database
6. **Notification**: Trigger downstream processes

### 2. Training Triggers

**Automatic Training** occurs when:
- Minimum files threshold reached (default: 3)
- Scheduled intervals (default: 2 AM, 2 PM)
- Manual trigger via API

**Training Process:**
1. **Data Preparation**: Load cached graphs
2. **Model Training**: Train Graph Neural Network
3. **Validation**: Evaluate on held-out data
4. **Versioning**: Save model with metadata
5. **Deployment**: Auto-deploy if improvement threshold met

### 3. Model Lifecycle

```
┌─────────────┐    ┌─────────────┐    ┌─────────────┐
│  Training   │───▶│   Staging   │───▶│ Production  │
│             │    │             │    │             │
└─────────────┘    └─────────────┘    └─────────────┘
       │                   │                   │
       ▼                   ▼                   ▼
   Experiment         Validation         Deployment
   Tracking          & Testing           & Monitoring
```

## 🗄️ Database Schema

### Core Tables

**gedcom_files**: Uploaded GEDCOM files
- `id`, `filename`, `file_hash`, `status`
- `num_individuals`, `num_families`, `processing_time`

**graph_cache**: Processed graph data
- `gedcom_id`, `graph_hash`, `node_features`, `edge_indices`
- `created_time`, `access_count`

**training_runs**: Training execution records
- `run_name`, `status`, `config`, `gedcom_ids`
- `start_time`, `duration`, `final_metrics`

**training_metrics**: Detailed training metrics
- `run_id`, `epoch`, `train_loss`, `val_accuracy`

## 📊 Monitoring & Alerting

### System Metrics
- **CPU Usage**: Process and system utilization
- **Memory**: RAM usage and available memory
- **Disk**: Storage usage and free space
- **Network**: Data transfer rates

### Pipeline Metrics
- **Queue Sizes**: Processing and training backlogs
- **Processing Times**: File parsing and graph building
- **Success Rates**: Processing and training success rates
- **Cache Performance**: Hit rates and access patterns

### Default Alerts
- High CPU usage (>80% for 5 minutes)
- High memory usage (>85% for 3 minutes)
- Low disk space (<5GB)
- Processing queue backup (>50 files for 10 minutes)

## 🔧 Configuration

### Environment Variables
```bash
export OHANA_SECRET_KEY="your-secret-key"
export AWS_ACCESS_KEY_ID="your-aws-key"
export AWS_SECRET_ACCESS_KEY="your-aws-secret"
export OHANA_DB_PATH="/path/to/database.db"
```

### Storage Backends

**Local Storage:**
```yaml
storage:
  backend: "local"
  local:
    base_path: "storage"
```

**AWS S3:**
```yaml
storage:
  backend: "s3"
  s3:
    bucket_name: "ohanaai-storage"
    region: "us-east-1"
```

**Google Cloud Storage:**
```yaml
storage:
  backend: "gcs"
  gcs:
    bucket_name: "ohanaai-storage"
    credentials_path: "/path/to/service-account.json"
```

## 🔐 Security

### Authentication
- **JWT Tokens**: Secure API access
- **API Keys**: Long-term service authentication
- **Role-Based Access**: Admin and user permissions

### Data Protection
- **File Validation**: Check uploaded files for safety
- **Rate Limiting**: Prevent abuse
- **Audit Logging**: Track all operations

### Production Considerations
```yaml
security:
  encrypt_at_rest: true
  audit_log_enabled: true
  allowed_ips: ["10.0.0.0/8"]
  scan_uploads: true
```

## 📈 Scaling & Performance

### Horizontal Scaling
- **Load Balancing**: Multiple API server instances
- **Distributed Processing**: Separate worker nodes
- **Database Clustering**: High availability databases

### Performance Optimization
```yaml
performance:
  db_connection_pool_size: 10
  parallel_processing_workers: 4
  graph_batch_size: 32
  enable_graph_cache: true
```

### Resource Monitoring
- Automatic scaling based on queue sizes
- Memory usage optimization
- Disk cleanup and rotation

## 🚨 Troubleshooting

### Common Issues

**Training Not Starting:**
```bash
# Check pipeline status
curl http://localhost:8000/status

# Check file processing
curl http://localhost:8000/files?status=failed

# Manual training trigger
curl -X POST http://localhost:8000/training/start \
  -H "Authorization: Bearer $TOKEN" \
  -d '{"gedcom_ids": [1,2,3]}'
```

**High Memory Usage:**
- Reduce `graph_batch_size` in config
- Enable graph cache compression
- Increase cache cleanup frequency

**Processing Failures:**
- Check GEDCOM file format and encoding
- Verify sufficient disk space
- Review error logs in database

### Logs & Debugging
```bash
# View logs
tail -f logs/mlops.log

# Database inspection
sqlite3 ohana_mlops.db ".tables"

# Health check
curl http://localhost:8000/health
```

## 🔄 Deployment for ohanaai.com

### Vercel Deployment Structure
```
ohanaai.com/
├── app/                    # Next.js frontend
│   ├── upload/            # File upload interface  
│   ├── dashboard/         # Training monitoring
│   └── predictions/       # Results visualization
├── api/                   # API routes
│   ├── upload.ts         # File upload endpoint
│   ├── training.ts       # Training management
│   └── status.ts         # System status
└── mlops/                 # MLOps backend (separate deployment)
    ├── run_mlops.py      # Main server
    ├── Dockerfile        # Container deployment
    └── vercel.json       # Vercel configuration
```

### Environment Setup
```bash
# Vercel environment variables
OHANA_SECRET_KEY=your-production-secret
DATABASE_URL=postgresql://user:pass@host:port/db
STORAGE_BACKEND=s3
AWS_S3_BUCKET=ohanaai-production
WEBHOOK_TOKEN=your-webhook-token
```

This MLOps pipeline provides a complete solution for continuous genealogical model training and deployment, ready for integration with your ohanaai.com website! 🌺