#!/usr/bin/env python3
"""
Ohana AI - Parent Prediction Model Training (M1 Mac Optimized)
Optimized for Apple Silicon M1/M2 Macs with Metal GPU acceleration
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings

import json
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from sklearn.preprocessing import StandardScaler
import matplotlib
matplotlib.use('Agg')  # Use non-interactive backend for M1 compatibility
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
import multiprocessing as mp
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Configure TensorFlow for Apple Silicon
print("Configuring TensorFlow for Apple Silicon...")
print(f"TensorFlow version: {tf.__version__}")

# Check for Apple Silicon GPU
if tf.config.list_physical_devices('GPU'):
    print("GPU (Metal) detected and available")
    # Enable memory growth for GPU
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
else:
    print("No GPU detected, using CPU")

# Set optimal thread counts for M1
if hasattr(tf.config.threading, 'set_intra_op_parallelism_threads'):
    tf.config.threading.set_intra_op_parallelism_threads(8)  # M1 has 8 performance cores
    tf.config.threading.set_inter_op_parallelism_threads(2)

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GedcomParser:
    """Parse GEDCOM files and extract family relationships - optimized for large files"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.individuals = {}
        self.families = {}
        self.relationships = defaultdict(list)
        
    def parse(self):
        """Parse the GEDCOM file with progress tracking"""
        print(f"Parsing GEDCOM file: {self.filepath}")
        
        # Get file size for progress tracking
        file_size = os.path.getsize(self.filepath)
        print(f"File size: {file_size / (1024*1024):.1f} MB")
        
        current_record = None
        current_type = None
        current_data = {}
        bytes_processed = 0
        
        with open(self.filepath, 'r', encoding='utf-8', buffering=8192) as f:
            for line_num, line in enumerate(f):
                bytes_processed += len(line.encode('utf-8'))
                
                if line_num % 100000 == 0 and line_num > 0:
                    progress = (bytes_processed / file_size) * 100
                    print(f"Progress: {progress:.1f}% ({line_num:,} lines)")
                
                line = line.strip()
                if not line:
                    continue
                
                try:
                    parts = line.split(' ', 2)
                    level = int(parts[0])
                    tag = parts[1] if len(parts) > 1 else ''
                    value = parts[2] if len(parts) > 2 else ''
                except (ValueError, IndexError):
                    continue  # Skip malformed lines
                
                if level == 0:
                    # Save previous record
                    self._save_record(current_record, current_type, current_data)
                    
                    # Start new record
                    if '@' in tag and tag.endswith('@'):
                        current_record = tag
                        current_type = value
                        current_data = {'id': current_record}
                    else:
                        current_record = None
                        current_type = None
                        current_data = {}
                
                elif level == 1 and current_record:
                    self._process_level1_tag(tag, value, current_data)
                
                elif level == 2 and current_record:
                    self._process_level2_tag(tag, value, current_data)
        
        # Save last record
        self._save_record(current_record, current_type, current_data)
        
        print(f"\\nParsed {len(self.individuals):,} individuals and {len(self.families):,} families")
        self._build_relationships()
        
        return self.individuals, self.families, dict(self.relationships)
    
    def _save_record(self, record_id, record_type, data):
        """Save a record to the appropriate collection"""
        if record_id and record_type and data:
            if record_type == 'INDI':
                self.individuals[record_id] = data
            elif record_type == 'FAM':
                self.families[record_id] = data
    
    def _process_level1_tag(self, tag, value, current_data):
        """Process level 1 GEDCOM tags"""
        if tag == 'NAME':
            name_clean = value.replace('/', '').strip()
            current_data['name'] = name_clean
            name_parts = name_clean.split()
            current_data['first_name'] = name_parts[0] if name_parts else ''
            current_data['last_name'] = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
        elif tag == 'SEX':
            current_data['gender'] = value
        elif tag in ['BIRT', 'DEAT', 'MARR']:
            current_data[f'_{tag.lower()}_event'] = True
        elif tag == 'HUSB':
            current_data['husband'] = value
        elif tag == 'WIFE':
            current_data['wife'] = value
        elif tag == 'CHIL':
            if 'children' not in current_data:
                current_data['children'] = []
            current_data['children'].append(value)
        elif tag == 'FAMC':
            current_data['family_child'] = value
        elif tag == 'FAMS':
            if 'family_spouse' not in current_data:
                current_data['family_spouse'] = []
            current_data['family_spouse'].append(value)
    
    def _process_level2_tag(self, tag, value, current_data):
        """Process level 2 GEDCOM tags"""
        if tag == 'DATE':
            if current_data.get('_birt_event'):
                current_data['birth_date'] = value
                current_data.pop('_birt_event', None)
            elif current_data.get('_deat_event'):
                current_data['death_date'] = value
                current_data.pop('_deat_event', None)
            elif current_data.get('_marr_event'):
                current_data['marriage_date'] = value
                current_data.pop('_marr_event', None)
        elif tag == 'PLAC':
            if '_birt_place' not in current_data and 'birth_date' in current_data:
                current_data['birth_place'] = value
            elif '_deat_place' not in current_data and 'death_date' in current_data:
                current_data['death_place'] = value
    
    def _build_relationships(self):
        """Build parent-child and spousal relationships"""
        print("Building relationships...")
        
        # Process families in parallel for better performance
        family_items = list(self.families.items())
        
        # Parent-child relationships
        parent_child_count = 0
        spousal_count = 0
        sibling_count = 0
        
        for fam_id, family in family_items:
            children = family.get('children', [])
            husband = family.get('husband')
            wife = family.get('wife')
            
            # Parent-child relationships
            for child in children:
                if husband and husband in self.individuals:
                    self.relationships['parent_child'].append({
                        'parent': husband,
                        'child': child,
                        'parent_type': 'father'
                    })
                    parent_child_count += 1
                    
                if wife and wife in self.individuals:
                    self.relationships['parent_child'].append({
                        'parent': wife,
                        'child': child,
                        'parent_type': 'mother'
                    })
                    parent_child_count += 1
            
            # Spousal relationships
            if husband and wife and husband in self.individuals and wife in self.individuals:
                self.relationships['spousal'].append({
                    'spouse1': husband,
                    'spouse2': wife
                })
                spousal_count += 1
            
            # Sibling relationships (limit to avoid explosion)
            if len(children) > 1 and len(children) <= 20:  # Reasonable family size limit
                for i, child1 in enumerate(children):
                    for child2 in children[i+1:]:
                        if child1 in self.individuals and child2 in self.individuals:
                            self.relationships['siblings'].append({
                                'sibling1': child1,
                                'sibling2': child2
                            })
                            sibling_count += 1
        
        print(f"Built {parent_child_count:,} parent-child relationships")
        print(f"Built {spousal_count:,} spousal relationships")
        print(f"Built {sibling_count:,} sibling relationships")

class FamilyGraphBuilder:
    """Build graph representation optimized for Apple Silicon"""
    
    def __init__(self, individuals, families, relationships):
        self.individuals = individuals
        self.families = families
        self.relationships = relationships
        self.graph = nx.Graph()
        self.node_features = {}
        self.edge_features = {}
        
    def build_graph(self):
        """Build NetworkX graph with optimized processing"""
        print("Building family graph...")
        
        # Add nodes with progress tracking
        node_count = 0
        for person_id, person_data in self.individuals.items():
            self.graph.add_node(person_id)
            self.node_features[person_id] = self._extract_person_features(person_data)
            node_count += 1
            
            if node_count % 5000 == 0:
                print(f"Processed {node_count:,} nodes...")
        
        print(f"Added {node_count:,} nodes")
        
        # Add edges efficiently
        edge_count = 0
        
        # Parent-child edges
        for rel in self.relationships.get('parent_child', []):
            if rel['parent'] in self.graph and rel['child'] in self.graph:
                edge = (rel['parent'], rel['child'])
                if not self.graph.has_edge(*edge):
                    self.graph.add_edge(*edge)
                    self.edge_features[edge] = {
                        'type': 'parent',
                        'parent_type': rel['parent_type'],
                        'weight': 1.0
                    }
                    edge_count += 1
        
        # Spousal edges
        for rel in self.relationships.get('spousal', []):
            if rel['spouse1'] in self.graph and rel['spouse2'] in self.graph:
                edge = (rel['spouse1'], rel['spouse2'])
                if not self.graph.has_edge(*edge):
                    self.graph.add_edge(*edge)
                    self.edge_features[edge] = {
                        'type': 'spouse',
                        'weight': 0.8
                    }
                    edge_count += 1
        
        # Sibling edges (sample to avoid memory issues)
        sibling_rels = self.relationships.get('siblings', [])
        if len(sibling_rels) > 50000:  # Limit sibling edges
            sibling_rels = np.random.choice(sibling_rels, 50000, replace=False)
        
        for rel in sibling_rels:
            if rel['sibling1'] in self.graph and rel['sibling2'] in self.graph:
                edge = (rel['sibling1'], rel['sibling2'])
                if not self.graph.has_edge(*edge):
                    self.graph.add_edge(*edge)
                    self.edge_features[edge] = {
                        'type': 'sibling',
                        'weight': 0.6
                    }
                    edge_count += 1
        
        print(f"Built graph with {self.graph.number_of_nodes():,} nodes and {edge_count:,} edges")
        return self.graph, self.node_features, self.edge_features
    
    def _extract_person_features(self, person_data):
        """Extract numerical features optimized for vectorization"""
        features = np.zeros(12, dtype=np.float32)
        
        # Gender encoding
        gender = person_data.get('gender', 'U')
        features[0] = 1.0 if gender == 'M' else 2.0 if gender == 'F' else 0.0
        
        # Date features
        features[1] = 1.0 if person_data.get('birth_date') else 0.0
        features[2] = 1.0 if person_data.get('death_date') else 0.0
        
        # Birth year (normalized)
        if person_data.get('birth_date'):
            year_match = re.search(r'\\b(\\d{4})\\b', person_data['birth_date'])
            if year_match:
                year = int(year_match.group(1))
                if 1500 <= year <= 2024:
                    features[3] = (year - 1700) / 324  # Normalize to ~0-1
        
        # Name features
        first_name = person_data.get('first_name', '')
        last_name = person_data.get('last_name', '')
        
        features[4] = min(len(first_name) / 20.0, 1.0)
        features[5] = min(len(last_name) / 30.0, 1.0)
        features[6] = 1.0 if first_name else 0.0
        features[7] = 1.0 if last_name else 0.0
        
        # Name complexity
        if first_name:
            vowels = sum(1 for c in first_name.lower() if c in 'aeiou')
            features[8] = vowels / len(first_name)
        
        if last_name:
            vowels = sum(1 for c in last_name.lower() if c in 'aeiou')
            features[9] = vowels / len(last_name)
        
        # Geographic features
        birth_place = person_data.get('birth_place', '').lower()
        features[10] = 1.0 if any(term in birth_place for term in ['usa', 'united states', 'america']) else 0.0
        features[11] = 1.0 if any(state in birth_place for state in ['california', 'utah', 'hawaii', 'nevada']) else 0.0
        
        return features

@tf.keras.utils.register_keras_serializable()
class MultiHeadGraphAttention(layers.Layer):
    """Optimized Multi-Head Graph Attention Layer for Apple Silicon"""
    
    def __init__(self, units, num_heads=8, dropout_rate=0.1, **kwargs):
        super().__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        self.head_dim = units // num_heads
        
        self.query_dense = layers.Dense(units, use_bias=False)
        self.key_dense = layers.Dense(units, use_bias=False)
        self.value_dense = layers.Dense(units, use_bias=False)
        self.output_dense = layers.Dense(units)
        self.dropout = layers.Dropout(dropout_rate)
        
    def build(self, input_shape):
        super().build(input_shape)
        
    def call(self, inputs, training=None):
        node_features, adjacency_matrix = inputs
        batch_size = tf.shape(node_features)[0]
        seq_len = tf.shape(node_features)[1]
        
        # Linear projections
        query = self.query_dense(node_features)
        key = self.key_dense(node_features)
        value = self.value_dense(node_features)
        
        # Reshape for multi-head attention
        query = tf.reshape(query, [batch_size, seq_len, self.num_heads, self.head_dim])
        key = tf.reshape(key, [batch_size, seq_len, self.num_heads, self.head_dim])
        value = tf.reshape(value, [batch_size, seq_len, self.num_heads, self.head_dim])
        
        # Transpose for attention computation
        query = tf.transpose(query, [0, 2, 1, 3])  # [batch, heads, seq, head_dim]
        key = tf.transpose(key, [0, 2, 1, 3])
        value = tf.transpose(value, [0, 2, 1, 3])
        
        # Scaled dot-product attention
        attention_scores = tf.matmul(query, key, transpose_b=True)
        attention_scores = attention_scores / tf.math.sqrt(tf.cast(self.head_dim, tf.float32))
        
        # Apply adjacency mask
        mask = tf.expand_dims(adjacency_matrix, axis=1)  # [batch, 1, seq, seq]
        mask = tf.tile(mask, [1, self.num_heads, 1, 1])  # [batch, heads, seq, seq]
        
        # Apply mask (set non-connected nodes to large negative value)
        attention_scores = tf.where(mask > 0, attention_scores, -1e9)
        
        # Softmax
        attention_weights = tf.nn.softmax(attention_scores, axis=-1)
        attention_weights = self.dropout(attention_weights, training=training)
        
        # Apply attention to values
        context = tf.matmul(attention_weights, value)  # [batch, heads, seq, head_dim]
        
        # Concatenate heads
        context = tf.transpose(context, [0, 2, 1, 3])  # [batch, seq, heads, head_dim]
        context = tf.reshape(context, [batch_size, seq_len, self.units])
        
        # Final linear projection
        output = self.output_dense(context)
        
        return output
    
    def get_config(self):
        config = super().get_config()
        config.update({
            'units': self.units,
            'num_heads': self.num_heads,
            'dropout_rate': self.dropout_rate
        })
        return config

def create_gat_model(max_nodes, feature_dim=12):
    """Create optimized GAT model for Apple Silicon"""
    print(f"Building GAT model for {max_nodes} nodes, {feature_dim} features...")
    
    # Input layers
    node_features = keras.Input(shape=(max_nodes, feature_dim), name='node_features')
    adjacency = keras.Input(shape=(max_nodes, max_nodes), name='adjacency')
    
    # Graph Attention Layers
    x = MultiHeadGraphAttention(64, num_heads=8, dropout_rate=0.1)(
        [node_features, adjacency]
    )
    x = layers.LayerNormalization()(x)
    x = layers.Activation('gelu')(x)  # GELU works well on Apple Silicon
    
    x = MultiHeadGraphAttention(32, num_heads=4, dropout_rate=0.1)(
        [x, adjacency]
    )
    x = layers.LayerNormalization()(x)
    x = layers.Activation('gelu')(x)
    
    # Global pooling
    x = layers.GlobalAveragePooling1D()(x)
    
    # Final prediction layers
    x = layers.Dense(64, activation='gelu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='gelu')(x)
    x = layers.Dropout(0.1)(x)
    
    # Output layer: [has_missing_parent, missing_father, missing_mother]
    outputs = layers.Dense(3, activation='sigmoid', name='predictions')(x)
    
    model = keras.Model(inputs=[node_features, adjacency], outputs=outputs)
    
    # Use optimized optimizer for Apple Silicon
    optimizer = keras.optimizers.AdamW(
        learning_rate=0.001,
        weight_decay=0.01,
        beta_1=0.9,
        beta_2=0.999
    )
    
    model.compile(
        optimizer=optimizer,
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def main():
    """Main training pipeline optimized for M1 Mac"""
    print("=== Ohana AI Model Training (M1 Mac Optimized) ===")
    print(f"Started at: {datetime.now()}")
    print(f"Running on {mp.cpu_count()} CPU cores")
    
    # Parse GEDCOM file
    print("\\n1. Parsing GEDCOM file...")
    parser = GedcomParser("Hussey Ohana.ged.txt")
    individuals, families, relationships = parser.parse()
    
    # Build graph
    print("\\n2. Building family graph...")
    graph_builder = FamilyGraphBuilder(individuals, families, relationships)
    graph, node_features, edge_features = graph_builder.build_graph()
    
    # Create training data
    print("\\n3. Creating training data...")
    training_examples = create_training_data(individuals, relationships)
    
    # Prepare graph data
    print("\\n4. Preparing graph data...")
    X_features, X_adjacency, y = prepare_graph_data(graph, node_features, training_examples, max_nodes=200)
    
    if len(X_features) == 0:
        print("ERROR: No training data generated!")
        return
    
    print(f"\\nDataset Summary:")
    print(f"- Examples: {X_features.shape[0]}")
    print(f"- Max nodes per graph: {X_features.shape[1]}")
    print(f"- Features per node: {X_features.shape[2]}")
    print(f"- Label distribution: {np.mean(y, axis=0)}")
    
    # Split data
    X_feat_train, X_feat_test, X_adj_train, X_adj_test, y_train, y_test = train_test_split(
        X_features, X_adjacency, y, test_size=0.2, random_state=42
    )
    
    print(f"\\nTraining: {X_feat_train.shape[0]} examples")
    print(f"Testing: {X_feat_test.shape[0]} examples")
    
    # Build model
    print("\\n5. Building model...")
    model = create_gat_model(max_nodes=X_features.shape[1])
    
    print("\\nModel Summary:")
    model.summary()
    
    # Callbacks optimized for M1
    callbacks = [
        keras.callbacks.EarlyStopping(
            patience=15, 
            restore_best_weights=True, 
            monitor='val_loss',
            verbose=1
        ),
        keras.callbacks.ReduceLROnPlateau(
            patience=8, 
            factor=0.5, 
            monitor='val_loss',
            verbose=1,
            min_lr=1e-6
        ),
        keras.callbacks.ModelCheckpoint(
            'models/parent_predictor/best_ohana_model.h5',
            save_best_only=True,
            monitor='val_loss',
            verbose=1
        )
    ]
    
    # Train model
    print("\\n6. Training model...")
    os.makedirs('models/parent_predictor', exist_ok=True)
    
    history = model.fit(
        [X_feat_train, X_adj_train], y_train,
        validation_data=([X_feat_test, X_adj_test], y_test),
        epochs=50,
        batch_size=8,  # Smaller batch size for M1 efficiency
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate
    print("\\n7. Evaluating model...")
    test_results = model.evaluate([X_feat_test, X_adj_test], y_test, verbose=0)
    
    print(f"\\nTest Results:")
    print(f"Loss: {test_results[0]:.4f}")
    print(f"Accuracy: {test_results[1]:.4f}")
    print(f"Precision: {test_results[2]:.4f}")
    print(f"Recall: {test_results[3]:.4f}")
    
    # Save model
    print("\\n8. Saving model...")
    model.save('models/parent_predictor/ohana_model_m1.h5')
    
    # Save training metadata
    metadata = {
        'training_date': datetime.now().isoformat(),
        'individuals_count': len(individuals),
        'families_count': len(families),
        'training_examples': len(X_features),
        'model_architecture': 'GAT',
        'test_accuracy': float(test_results[1]),
        'test_precision': float(test_results[2]),
        'test_recall': float(test_results[3])
    }
    
    with open('models/parent_predictor/training_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\\n✅ Training completed successfully!")
    print("Model saved to: models/parent_predictor/ohana_model_m1.h5")
    print(f"Completed at: {datetime.now()}")

# Additional helper functions
def create_training_data(individuals, relationships):
    """Create training examples from family data"""
    print("Creating training examples...")
    
    parent_child_rels = relationships.get('parent_child', [])
    child_to_parents = defaultdict(list)
    
    for rel in parent_child_rels:
        child_to_parents[rel['child']].append(rel)
    
    training_examples = []
    stats = {'complete': 0, 'missing_father': 0, 'missing_mother': 0, 'missing_both': 0}
    
    for person_id in individuals:
        parents = child_to_parents[person_id]
        has_father = any(p['parent_type'] == 'father' for p in parents)
        has_mother = any(p['parent_type'] == 'mother' for p in parents)
        
        # Update statistics
        if has_father and has_mother:
            stats['complete'] += 1
        elif has_father and not has_mother:
            stats['missing_mother'] += 1
        elif not has_father and has_mother:
            stats['missing_father'] += 1
        else:
            stats['missing_both'] += 1
        
        label = [
            1 if not has_father or not has_mother else 0,
            1 if not has_father else 0,
            1 if not has_mother else 0
        ]
        
        training_examples.append({
            'person_id': person_id,
            'label': label,
            'has_father': has_father,
            'has_mother': has_mother
        })
    
    print(f"Training examples: {len(training_examples):,}")
    for key, value in stats.items():
        print(f"  {key}: {value:,} ({value/len(training_examples)*100:.1f}%)")
    
    return training_examples

def prepare_graph_data(graph, node_features, training_examples, max_nodes=200):
    """Prepare graph data for training - memory optimized"""
    print(f"Preparing graph data (max {max_nodes} nodes per component)...")
    
    # Get connected components
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)
    
    print(f"Found {len(components)} connected components")
    if components:
        print(f"Largest: {len(components[0])}, Smallest: {len(components[-1])}")
    
    X_features = []
    X_adjacency = []
    y = []
    
    # Process components
    processed = 0
    for component in components:
        if len(component) < 5:  # Skip tiny components
            continue
        if processed >= 100:  # Limit to prevent memory issues
            break
            
        # Limit component size
        if len(component) > max_nodes:
            component = list(component)[:max_nodes]
        
        subgraph = graph.subgraph(component)
        nodes = list(subgraph.nodes())
        
        # Create features matrix
        features_matrix = np.zeros((max_nodes, 12), dtype=np.float32)
        for i, node in enumerate(nodes):
            if node in node_features:
                features_matrix[i] = node_features[node]
        
        # Create adjacency matrix
        adj_matrix = np.zeros((max_nodes, max_nodes), dtype=np.float32)
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        for edge in subgraph.edges():
            if edge[0] in node_to_idx and edge[1] in node_to_idx:
                i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
                adj_matrix[i, j] = 1.0
                adj_matrix[j, i] = 1.0
        
        # Component-level labels
        component_labels = [0, 0, 0]
        for example in training_examples:
            if example['person_id'] in component:
                for i in range(3):
                    if example['label'][i]:
                        component_labels[i] = 1
        
        X_features.append(features_matrix)
        X_adjacency.append(adj_matrix)
        y.append(component_labels)
        processed += 1
        
        if processed % 20 == 0:
            print(f"Processed {processed} components...")
    
    print(f"Final dataset: {len(X_features)} graph examples")
    
    return np.array(X_features), np.array(X_adjacency), np.array(y)

if __name__ == "__main__":
    main()