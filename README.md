# OhanaAI - Genealogical Parent Prediction

A complete Graph Neural Network system for predicting missing parents in genealogical GEDCOM files using Apple's MLX framework, integrated with a Tkinter GUI for user interaction.

## Features

- **GEDCOM Parser**: Parse .gedcom genealogy files extracting individual records and family relationships
- **Graph Neural Network**: Graph Attention Network (GAT) with edge-type aware message passing using MLX
- **Parent Prediction**: Binary classifier for parent-child relationships with confidence scoring
- **Deduplication System**: Detect and merge duplicate individuals across multiple GEDCOM files
- **Tkinter GUI**: User-friendly interface for file loading, visualization, and predictions
- **CLI Interface**: Command-line tools for training, prediction, and analysis

## Architecture

### Core Components

- **GEDCOM Parser** (`gedcom_parser.py`): Parse GEDCOM files into Individual/Family dataclasses
- **Graph Builder** (`graph_builder.py`): Convert family trees to MLX graph format with node features and edges
- **GNN Model** (`gnn_model.py`): GAT layers and OhanaAI model implementation in MLX
- **Trainer** (`trainer.py`): Training loop with MLX optimizers and early stopping
- **Predictor** (`predictor.py`): Inference and parent prediction logic with constraint validation
- **Deduplication** (`data_deduplication.py`): Handle duplicates across files with similarity scoring
- **GUI** (`gui.py`): Tkinter interface for visualization and user interaction

### Model Architecture

- **Graph Attention Network (GAT)** with multi-head attention (4 heads)
- **Edge-type aware message passing** for different relationship types
- **3-layer GNN** with hidden dimensions of 256
- **Node embeddings** of dimension 128
- **Parent prediction head** with binary classification

## Installation

### Requirements

- Python 3.8+
- Apple Silicon Mac (for MLX) or compatible system
- MLX framework

### Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd OhanaAI
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Install MLX (if not already installed):
```bash
pip install mlx
```

## Usage

### GUI Mode (Recommended)

Launch the graphical interface:

```bash
python -m ohana_ai.main gui
```

The GUI provides:
- File loading dialog for GEDCOM files
- Interactive training and prediction controls
- Results visualization with confidence scores
- Duplicate detection and review interface
- Export options for predictions

### Command Line Interface

#### Show GEDCOM file information:
```bash
python -m ohana_ai.main info family1.ged family2.ged
```

#### Train a model:
```bash
python -m ohana_ai.main train family1.ged family2.ged --epochs 50
```

#### Run predictions:
```bash
python -m ohana_ai.main predict family1.ged family2.ged --model checkpoints/ohana_model.npz
```

#### Detect duplicates:
```bash
python -m ohana_ai.main deduplicate family1.ged family2.ged --threshold 0.9
```

## Configuration

Edit `config.yaml` to customize:

- **Model parameters**: Hidden dimensions, attention heads, layers
- **Training settings**: Learning rate, batch size, epochs
- **Data constraints**: Age differences, date tolerances
- **Output paths**: Checkpoints, logs, results

## Input Data Format

### GEDCOM Files

Standard GEDCOM 5.5.1 format with:
- Individual records (`INDI`) with names, dates, places
- Family records (`FAM`) with parent-child and spouse relationships
- Proper date formats (e.g., "DD MMM YYYY")

### Node Features

Each individual is represented with:
- **Gender**: One-hot encoded (M, F, U)
- **Birth/Death years**: Normalized temporal features
- **Name statistics**: Character-level and length features
- **Location features**: Hash-based encoding
- **Family connectivity**: Number of relationships

### Edge Types

- **0**: Parent → Child
- **1**: Child → Parent  
- **2**: Spouse (bidirectional)
- **3**: Sibling (bidirectional)

## Output

### Predictions

Generated predictions include:
- Child and candidate parent information
- Confidence scores (0.0 - 1.0)
- Age differences and constraint validation
- Export formats: CSV, GEDCOM supplement

### Constraints Validated

- **Age constraints**: Parents 12-70 years older than children
- **Temporal constraints**: Parents alive when child was born
- **Relationship constraints**: No existing parent-child relationships

## Deduplication

The system can detect and merge duplicate individuals across files using:
- **Name similarity**: Levenshtein distance and fuzzy matching
- **Date similarity**: Birth/death year tolerance (±5 years)
- **Location matching**: Place name normalization and comparison
- **User confirmation**: GUI interface for reviewing potential merges

## Model Training

### Training Process

1. **Data Preparation**: Parse GEDCOM files and build graph representation
2. **Training Pairs**: Generate positive (true parent-child) and negative pairs
3. **Contrastive Loss**: Train with class imbalance handling
4. **Early Stopping**: Validation-based stopping with patience=5 epochs
5. **Checkpointing**: Save model state every 10 epochs

### Training Data

- Uses existing parent-child relationships as positive examples
- Generates negative examples with age and temporal constraints
- 3:1 negative to positive ratio for balanced training
- Train/validation split: 80/20

## File Structure

```
OhanaAI/
├── ohana_ai/
│   ├── __init__.py              # Package initialization
│   ├── gedcom_parser.py         # GEDCOM file parsing
│   ├── graph_builder.py         # Graph construction
│   ├── gnn_model.py            # GAT model implementation
│   ├── trainer.py              # Training pipeline
│   ├── predictor.py            # Inference logic
│   ├── data_deduplication.py   # Duplicate handling
│   ├── gui.py                  # Tkinter interface
│   └── main.py                 # CLI entry point
├── config.yaml                 # Configuration file
├── requirements.txt            # Dependencies
├── README.md                   # Documentation
├── logs/                       # Training and inference logs
├── checkpoints/               # Model checkpoints
└── outputs/                   # Prediction results
```

## Example Usage

### Basic Workflow

1. **Load GEDCOM files** through GUI or CLI
2. **Train model** on your genealogical data
3. **Run predictions** to find missing parents
4. **Review results** with confidence scores and constraints
5. **Export predictions** as CSV or GEDCOM supplement

### Advanced Features

- **Batch Processing**: Handle large datasets efficiently
- **Incremental Training**: Add new data to existing models
- **Confidence Filtering**: Adjust thresholds for prediction quality
- **Constraint Validation**: Ensure genealogically valid predictions

## Technical Details

### Graph Neural Network

- **Message Passing**: Node features aggregated via attention mechanism
- **Edge Type Embeddings**: Different relationship types have learned representations
- **Multi-head Attention**: 4 attention heads for robust feature learning
- **Residual Connections**: Skip connections for training stability

### MLX Integration

- All tensor operations use MLX arrays for Apple Silicon optimization
- Automatic differentiation for gradient computation
- Efficient batching for large graph processing
- Native support for Mac GPU acceleration

## Troubleshooting

### Common Issues

1. **MLX Installation**: Ensure Apple Silicon Mac or compatible system
2. **Memory Usage**: Reduce batch size for large genealogies
3. **GEDCOM Parsing**: Check file encoding (UTF-8, Latin-1 supported)
4. **Training Convergence**: Adjust learning rate or model capacity

### Performance Tips

- Use GPU acceleration with MLX on Apple Silicon
- Batch multiple small families for efficient processing
- Filter individuals with insufficient data before training
- Use early stopping to prevent overfitting

## Contributing

Contributions welcome! Please submit issues and pull requests.

## License

[License information to be added]

## Citation

If you use OhanaAI in your research, please cite:

```
OhanaAI: Graph Neural Networks for Genealogical Parent Prediction
[Citation details to be added]
```