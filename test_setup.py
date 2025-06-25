#!/usr/bin/env python3
"""
Quick setup test for OhanaAI.
This script verifies that the application is properly configured and ready to use.
"""

import sys
from pathlib import Path

def test_ohana_setup():
    """Test OhanaAI setup and configuration."""
    print("🧬 OhanaAI Setup Test")
    print("=" * 40)
    
    # Test imports
    try:
        import ohana_ai
        from ohana_ai.core.config import load_config
        from ohana_ai.main import main
        print("✓ Core imports successful")
    except ImportError as e:
        print(f"✗ Import error: {e}")
        return False
    
    # Test configuration
    try:
        config_path = Path("config.yaml")
        if not config_path.exists():
            print(f"✗ Configuration file not found: {config_path}")
            return False
        
        config = load_config(config_path)
        print(f"✓ Configuration loaded from {config_path}")
        print(f"  - Model architecture: {config.num_layers} layers, {config.hidden_dim} hidden dim")
        print(f"  - Training: {config.epochs} epochs, LR {config.learning_rate}")
        print(f"  - Output paths configured: {config.outputs_dir}")
    except Exception as e:
        print(f"✗ Configuration error: {e}")
        return False
    
    # Test directory structure
    required_dirs = ["ohana_ai", "logs", "checkpoints", "outputs"]
    for dir_name in required_dirs:
        dir_path = Path(dir_name)
        if dir_path.exists():
            print(f"✓ Directory exists: {dir_name}/")
        else:
            print(f"ℹ Creating directory: {dir_name}/")
            dir_path.mkdir(exist_ok=True)
    
    # Test CLI
    try:
        import subprocess
        result = subprocess.run([sys.executable, "-m", "ohana_ai.main", "--help"], 
                              capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print("✓ CLI interface functional")
        else:
            print(f"✗ CLI error: {result.stderr}")
            return False
    except Exception as e:
        print(f"✗ CLI test error: {e}")
        return False
    
    print("\n🎉 OhanaAI setup test completed successfully!")
    print("\nNext steps:")
    print("1. Place GEDCOM files in the uploads/ directory")
    print("2. Run 'python -m ohana_ai.main gui' to launch the GUI")
    print("3. Or use CLI: 'python -m ohana_ai.main info <gedcom_file>' for file analysis")
    
    return True

if __name__ == "__main__":
    success = test_ohana_setup()
    sys.exit(0 if success else 1)