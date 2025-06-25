# OhanaAI 🌺

**AI-Powered Genealogical Parent Prediction**

A sophisticated Graph Neural Network system that predicts missing parents in genealogical GEDCOM files using Apple's MLX framework. Combines cutting-edge machine learning with an intuitive GUI for genealogy research.

[![GitHub](https://img.shields.io/badge/GitHub-ikaikaalika%2Fohanaai-blue?logo=github)](https://github.com/ikaikaalika/ohanaai)
[![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python)](https://python.org)
[![MLX](https://img.shields.io/badge/MLX-Apple%20Silicon-orange?logo=apple)](https://github.com/ml-explore/mlx)

## ✨ Features

- 🧬 **GEDCOM Parser**: Parse genealogy files extracting individual records and family relationships
- 🧠 **Graph Neural Network**: Graph Attention Network (GAT) with edge-type aware message passing
- 👨‍👩‍👧‍👦 **Parent Prediction**: Binary classifier for parent-child relationships with confidence scoring
- 🔍 **Smart Deduplication**: Detect and merge duplicate individuals across multiple GEDCOM files
- 💻 **Modern GUI**: User-friendly Tkinter interface for visualization and predictions
- ⚡ **CLI Tools**: Command-line interface for training, prediction, and analysis

## 🚀 Quick Start

### Prerequisites

- Python 3.8+
- Apple Silicon Mac (recommended for MLX optimization)
- GEDCOM genealogy files

### Installation

1. **Clone the repository**:
```bash
git clone https://github.com/ikaikaalika/ohanaai.git
cd ohanaai
```

2. **Install dependencies**:
```bash
pip install -r requirements.txt
```

3. **Launch the GUI**:
```bash
python -m ohana_ai.main gui
```

## 🖥️ Usage

### GUI Mode (Recommended)

Launch the graphical interface for the easiest experience:

```bash
python -m ohana_ai.main gui
```

**Features:**
- 📁 Interactive file loading for GEDCOM files
- 🎯 One-click training and prediction
- 📊 Results visualization with confidence scores
- 🔄 Duplicate detection and review interface
- 📤 Export predictions as CSV or GEDCOM

### Command Line Interface

#### View GEDCOM Information
```bash
python -m ohana_ai.main info family.ged
```

#### Train a Model
```bash
python -m ohana_ai.main train family1.ged family2.ged --epochs 50
```

#### Generate Predictions
```bash
python -m ohana_ai.main predict family.ged --model checkpoints/ohana_model.npz
```

#### Detect Duplicates
```bash
python -m ohana_ai.main deduplicate family.ged --threshold 0.9
```

## 🏗️ Architecture

### Core Components

| Component | File | Purpose |
|-----------|------|---------|
| **GEDCOM Parser** | `gedcom_parser.py` | Parse GEDCOM files into structured data |
| **Graph Builder** | `graph_builder.py` | Convert family trees to MLX graph format |
| **GNN Model** | `gnn_model.py` | Graph Attention Network implementation |
| **Trainer** | `trainer.py` | Model training with early stopping |
| **Predictor** | `predictor.py` | Parent prediction with constraint validation |
| **Deduplication** | `data_deduplication.py` | Smart duplicate detection and merging |
| **GUI** | `gui.py` | Tkinter user interface |

### Model Architecture

- **Graph Attention Network (GAT)** with 4-head multi-attention
- **Edge-type aware message passing** for different relationship types
- **3-layer deep architecture** with 256 hidden dimensions
- **128-dimensional node embeddings**
- **Binary classification head** for parent prediction

## 📊 Data Format

### Input: GEDCOM Files
Standard GEDCOM 5.5.1 format supporting:
- Individual records (`INDI`) with names, dates, places
- Family records (`FAM`) with relationships
- Proper date formats (e.g., "25 DEC 1850")

### Node Features
Each person is encoded with:
- **Demographics**: Gender, birth/death years
- **Names**: Character-level and statistical features
- **Geography**: Location hash encoding
- **Connectivity**: Family relationship counts

### Edge Types
- **Parent → Child** (Type 0)
- **Child → Parent** (Type 1)
- **Spouse ↔ Spouse** (Type 2)
- **Sibling ↔ Sibling** (Type 3)

## 🎯 Prediction System

### Smart Constraints
- **Age validation**: Parents 12-70 years older than children
- **Temporal logic**: Parents alive when child was born
- **Relationship rules**: No conflicting existing relationships

### Output
- **Confidence scores** (0.0 - 1.0)
- **Age difference analysis**
- **Constraint validation results**
- **Export formats**: CSV, enhanced GEDCOM

## ⚙️ Configuration

Customize behavior via `config.yaml`:

```yaml
model:
  hidden_dim: 256
  num_heads: 4
  num_layers: 3

training:
  learning_rate: 0.001
  batch_size: 64
  epochs: 100

constraints:
  min_parent_age_diff: 12
  max_parent_age_diff: 70
  date_tolerance_years: 5
```

## 📁 Project Structure

```
ohanaai/
├── ohana_ai/
│   ├── gedcom_parser.py      # GEDCOM file parsing
│   ├── graph_builder.py      # Graph construction
│   ├── gnn_model.py         # GAT model (MLX)
│   ├── trainer.py           # Training pipeline
│   ├── predictor.py         # Inference engine
│   ├── data_deduplication.py # Duplicate handling
│   ├── gui.py               # Tkinter interface
│   └── main.py              # CLI entry point
├── config.yaml              # Configuration
├── requirements.txt         # Dependencies
├── logs/                    # Training logs
├── checkpoints/            # Model saves
├── outputs/               # Results
└── uploads/              # Input files
```

## 🧪 Example Workflow

1. **Load Data**: Import GEDCOM files via GUI or CLI
2. **Train Model**: Build GNN on your genealogical data
3. **Generate Predictions**: Find missing parent relationships
4. **Review Results**: Examine confidence scores and constraints
5. **Export Findings**: Save as CSV or enhanced GEDCOM

## 🔧 Technical Details

### MLX Integration
- **Apple Silicon optimized** tensor operations
- **Automatic differentiation** for gradient computation
- **Efficient batching** for large family trees
- **Native GPU acceleration** on Mac

### Graph Neural Network
- **Multi-head attention** mechanism (4 heads)
- **Message passing** with edge-type embeddings
- **Residual connections** for training stability
- **Contrastive learning** with balanced sampling

## 🚨 Troubleshooting

### Common Issues

| Issue | Solution |
|-------|----------|
| MLX installation fails | Ensure Apple Silicon Mac |
| Memory errors | Reduce batch size in config |
| GEDCOM parsing errors | Check file encoding (UTF-8/Latin-1) |
| Poor predictions | Increase training data or epochs |

### Performance Tips
- Use MLX GPU acceleration on Apple Silicon
- Batch small families together for efficiency
- Filter individuals with missing data
- Enable early stopping to prevent overfitting

## 🤝 Contributing

We welcome contributions! Please:

1. Fork the repository
2. Create a feature branch
3. Submit a pull request

## 📄 License

[License details to be added]

## 📚 Citation

If you use OhanaAI in your research:

```bibtex
@software{ohanaai2024,
  title={OhanaAI: Graph Neural Networks for Genealogical Parent Prediction},
  author={Tyler Gee},
  url={https://github.com/ikaikaalika/ohanaai},
  year={2024}
}
```

---

**OhanaAI** - *Reconnecting families through AI* 🌺

*"Ohana means family. Family means nobody gets left behind or forgotten."*