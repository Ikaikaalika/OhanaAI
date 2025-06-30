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
echo "🔧 Installing TensorFlow with Metal support..."
pip install tensorflow-macos tensorflow-metal

echo "📊 Installing scientific computing packages..."
pip install -r requirements_m1.txt

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

# Test TensorFlow installation
echo "🧪 Testing TensorFlow installation..."
python3 -c "
import tensorflow as tf
print(f'TensorFlow version: {tf.__version__}')
print(f'GPU available: {len(tf.config.list_physical_devices(\"GPU\")) > 0}')
if len(tf.config.list_physical_devices('GPU')) > 0:
    print('✅ Metal GPU acceleration available!')
else:
    print('⚠️  GPU not detected, using CPU only')
print('✅ TensorFlow installation successful!')
"

# Create a simple test script
echo "📝 Creating test script..."
cat > test_ml_setup.py << 'EOF'
#!/usr/bin/env python3
"""Test script for Ohana AI ML setup"""

import tensorflow as tf
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from datetime import datetime

def test_setup():
    print("=== Ohana AI ML Setup Test ===")
    print(f"Test started: {datetime.now()}")
    
    # Test TensorFlow
    print(f"\n1. TensorFlow: {tf.__version__}")
    print(f"   GPU devices: {len(tf.config.list_physical_devices('GPU'))}")
    
    # Test basic operations
    print("\n2. Testing basic TensorFlow operations...")
    a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
    b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
    c = tf.matmul(a, b)
    print(f"   Matrix multiplication result: {c.numpy()}")
    
    # Test NetworkX
    print(f"\n3. NetworkX: {nx.__version__}")
    G = nx.Graph()
    G.add_edges_from([(1, 2), (2, 3), (3, 1)])
    print(f"   Created test graph with {G.number_of_nodes()} nodes and {G.number_of_edges()} edges")
    
    # Test a simple neural network
    print("\n4. Testing neural network creation...")
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(10, activation='relu', input_shape=(5,)),
        tf.keras.layers.Dense(3, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy')
    print(f"   Created model with {model.count_params()} parameters")
    
    # Test with sample data
    X = np.random.random((100, 5))
    y = np.random.randint(2, size=(100, 3))
    
    print("\n5. Testing model training...")
    history = model.fit(X, y, epochs=3, verbose=0, validation_split=0.2)
    final_loss = history.history['loss'][-1]
    print(f"   Final training loss: {final_loss:.4f}")
    
    print("\n✅ All tests passed! ML environment is ready.")
    print("🚀 You can now run: python train_model_m1.py")

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