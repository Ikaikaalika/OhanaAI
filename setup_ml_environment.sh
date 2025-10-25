#!/bin/bash
# Ohana AI - M1 Mac ML Environment Setup
# Sets up the complete machine learning environment for Apple Silicon

set -e  # Exit on any error

echo "🚀 Setting up Ohana AI ML Environment for M1 Mac..."

# Check if we're on macOS
if [[ "$OSTYPE" != "darwin"* ]]; then
    echo "❌ This script is designed for macOS (M1/M2 Mac)"
    exit 1
fi

# Check for Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "⚠️  Warning: This script is optimized for Apple Silicon (M1/M2)"
    echo "   It may still work on Intel Macs but performance will be different"
fi

# Create Python virtual environment
echo "📦 Creating Python virtual environment..."
python3 -m venv venv_ohana_ml
source venv_ohana_ml/bin/activate

# Upgrade pip
echo "⬆️  Upgrading pip..."
pip install --upgrade pip

# Install M1-optimized packages
echo "🔧 Installing MLX and ONNX tooling..."
pip install mlx mlx-data onnx numpy scikit-learn

# Create necessary directories
echo "📁 Creating project directories..."
mkdir -p models/parent_predictor
mkdir -p training_data
mkdir -p exports/ml_training
mkdir -p model_backups
mkdir -p scripts
mkdir -p logs

# Set up environment variables
echo "🔐 Setting up environment variables..."
if [ ! -f .env.local ]; then
    echo "Creating .env.local template..."
    cat > .env.local << EOF
# Database
DATABASE_URL="postgresql://username:password@localhost:5432/ohana_ai"

# NextAuth
NEXTAUTH_SECRET="$(openssl rand -base64 32)"
NEXTAUTH_URL="http://localhost:3000"

# ML Training
ML_EXPORT_API_KEY="$(openssl rand -base64 32)"
EXPORT_SECRET="$(openssl rand -base64 32)"

# Optional: Notification webhooks
# SLACK_WEBHOOK_URL=""
# DISCORD_WEBHOOK_URL=""
EOF
    echo "✅ Created .env.local with generated secrets"
    echo "⚠️  Please update the DATABASE_URL with your actual database connection"
else
    echo "✅ .env.local already exists"
fi

# Update training config with generated API key
if [ -f .env.local ]; then
    API_KEY=$(grep ML_EXPORT_API_KEY .env.local | cut -d'=' -f2 | tr -d '"')
    if [ ! -z "$API_KEY" ]; then
        # Update training_config.json with the API key
        python3 -c "
import json
try:
    with open('training_config.json', 'r') as f:
        config = json.load(f)
    config['api_key'] = '$API_KEY'
    with open('training_config.json', 'w') as f:
        json.dump(config, f, indent=2)
    print('✅ Updated training_config.json with API key')
except Exception as e:
    print(f'⚠️  Could not update training_config.json: {e}')
"
    fi
fi

# Test MLX installation
echo "🧪 Testing MLX installation..."
python3 -c "
import mlx.core as mx
import mlx.nn as nn
x = mx.random.uniform(shape=(4, 12))
m = nn.Linear(12, 3)
y = m(x)
mx.eval(y)
print('✅ MLX operational. Output shape:', y.shape)
"

# Create a simple test script
echo "📝 Creating test script..."
cat > test_ml_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script for Ohana AI MLX setup"""

import mlx.core as mx
import mlx.nn as nn
import numpy as np
from datetime import datetime

def test_setup():
    print("=== Ohana AI MLX Setup Test ===")
    print(f"Test started: {datetime.now()}")
    x = mx.random.uniform(shape=(8, 12))
    model = nn.Sequential(nn.Linear(12, 32), nn.relu, nn.Linear(32, 3), nn.sigmoid)
    y = model(x)
    mx.eval(y)
    print("Output sample:", np.array(y)[0])
    print("✅ MLX forward pass succeeded")

if __name__ == "__main__":
    test_setup()
EOF

chmod +x test_ml_setup.py

# Run the test
echo "🧪 Running setup test..."
python3 test_ml_setup.py

# Set up cron job template
echo "⏰ Creating cron job template..."
cat > setup_cron.sh << 'EOF'
#!/bin/bash
# Set up automated training cron job

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_ENV="$SCRIPT_DIR/venv_ohana_ml/bin/python"

# Add to crontab (runs daily at 2 AM)
CRON_JOB="0 2 * * * cd $SCRIPT_DIR && $PYTHON_ENV scripts/auto_train.py >> logs/cron.log 2>&1"

echo "To set up automated training, run:"
echo "crontab -e"
echo "Then add this line:"
echo "$CRON_JOB"
echo ""
echo "Or run this command:"
echo "(crontab -l 2>/dev/null; echo \"$CRON_JOB\") | crontab -"
EOF

chmod +x setup_cron.sh

# Create quick start script
echo "🚀 Creating quick start script..."
cat > quick_start.sh << 'EOF'
#!/bin/bash
# Quick start script for Ohana AI training

echo "🚀 Ohana AI Quick Start"

# Activate virtual environment
source venv_ohana_ml/bin/activate

# Check what we can do
if [ -f "Hussey Ohana.ged.txt" ]; then
    echo "📁 Found GEDCOM file: Hussey Ohana.ged.txt"
    echo "🤖 You can train the initial model with:"
    echo "   python train_model_m1.py"
    echo ""
fi

echo "🌐 To start the web app:"
echo "   npm run dev"
echo ""

echo "🔄 To set up automated training:"
echo "   ./setup_cron.sh"
echo ""

echo "📊 To manually fetch new data and retrain:"
echo "   python scripts/auto_train.py --manual"
echo ""

echo "🧪 To test the ML setup:"
echo "   python test_ml_setup.py"
EOF

chmod +x quick_start.sh

echo ""
echo "🎉 Setup complete!"
echo ""
echo "📋 Next steps:"
echo "1. Update .env.local with your database URL"
echo "2. Deploy your web app to Vercel"
echo "3. Update training_config.json with your app URL"
echo "4. Run: ./quick_start.sh"
echo ""
echo "🔧 Available commands:"
echo "   ./quick_start.sh         - Show available options"
echo "   python train_model_m1.py - Train initial model"
echo "   ./setup_cron.sh          - Set up automated training"
echo "   python test_ml_setup.py  - Test ML environment"
echo ""
echo "🚀 Your AI model training environment is ready!"
