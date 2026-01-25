#!/usr/bin/env python3
"""
Ohana AI - Parent Prediction Model Training
Trains a Graph Attention Network (GAT) on the Hussey Ohana family tree data
to predict missing parent relationships.
"""

import os
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
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict
import re
from datetime import datetime

# Set random seeds for reproducibility
np.random.seed(42)
tf.random.set_seed(42)

class GedcomParser:
    """Parse GEDCOM files and extract family relationships"""
    
    def __init__(self, filepath):
        self.filepath = filepath
        self.individuals = {}
        self.families = {}
        self.relationships = defaultdict(list)
        
    def parse(self):
        """Parse the GEDCOM file"""
        print(f"Parsing GEDCOM file: {self.filepath}")
        
        current_record = None
        current_type = None
        current_data = {}
        
        with open(self.filepath, 'r', encoding='utf-8') as f:
            for line_num, line in enumerate(f):
                if line_num % 50000 == 0:
                    print(f"Processed {line_num} lines...")
                
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split(' ', 2)
                level = int(parts[0])
                tag = parts[1] if len(parts) > 1 else ''
                value = parts[2] if len(parts) > 2 else ''
                
                if level == 0:
                    # Save previous record
                    if current_record and current_type:
                        if current_type == 'INDI':
                            self.individuals[current_record] = current_data
                        elif current_type == 'FAM':
                            self.families[current_record] = current_data
                    
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
                    if tag == 'NAME':
                        current_data['name'] = value.replace('/', '')
                        # Extract first and last names
                        name_parts = current_data['name'].split()
                        current_data['first_name'] = name_parts[0] if name_parts else ''
                        current_data['last_name'] = ' '.join(name_parts[1:]) if len(name_parts) > 1 else ''
                    elif tag == 'SEX':
                        current_data['gender'] = value
                    elif tag == 'BIRT':
                        current_data['_birth_event'] = True
                    elif tag == 'DEAT':
                        current_data['_death_event'] = True
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
                
                elif level == 2 and current_record:
                    if tag == 'DATE':
                        if current_data.get('_birth_event'):
                            current_data['birth_date'] = value
                            current_data.pop('_birth_event', None)
                        elif current_data.get('_death_event'):
                            current_data['death_date'] = value
                            current_data.pop('_death_event', None)
                    elif tag == 'PLAC':
                        if 'birth_date' in current_data and '_birth_place' not in current_data:
                            current_data['birth_place'] = value
                        elif 'death_date' in current_data and '_death_place' not in current_data:
                            current_data['death_place'] = value
        
        # Save last record
        if current_record and current_type:
            if current_type == 'INDI':
                self.individuals[current_record] = current_data
            elif current_type == 'FAM':
                self.families[current_record] = current_data
        
        print(f"Parsed {len(self.individuals)} individuals and {len(self.families)} families")
        self._build_relationships()
        
        return self.individuals, self.families, dict(self.relationships)
    
    def _build_relationships(self):
        """Build parent-child and spousal relationships"""
        print("Building relationships...")
        
        # Parent-child relationships from families
        for fam_id, family in self.families.items():
            children = family.get('children', [])
            husband = family.get('husband')
            wife = family.get('wife')
            
            for child in children:
                if husband:
                    self.relationships['parent_child'].append({
                        'parent': husband,
                        'child': child,
                        'parent_type': 'father'
                    })
                if wife:
                    self.relationships['parent_child'].append({
                        'parent': wife,
                        'child': child,
                        'parent_type': 'mother'
                    })
            
            # Spousal relationships
            if husband and wife:
                self.relationships['spousal'].append({
                    'spouse1': husband,
                    'spouse2': wife
                })
        
        # Sibling relationships
        for fam_id, family in self.families.items():
            children = family.get('children', [])
            for i, child1 in enumerate(children):
                for child2 in children[i+1:]:
                    self.relationships['siblings'].append({
                        'sibling1': child1,
                        'sibling2': child2
                    })
        
        print(f"Built {len(self.relationships['parent_child'])} parent-child relationships")
        print(f"Built {len(self.relationships['spousal'])} spousal relationships")
        print(f"Built {len(self.relationships['siblings'])} sibling relationships")

class FamilyGraphBuilder:
    """Build graph representation of family data"""
    
    def __init__(self, individuals, families, relationships):
        self.individuals = individuals
        self.families = families
        self.relationships = relationships
        self.graph = nx.Graph()
        self.node_features = {}
        self.edge_features = {}
        
    def build_graph(self):
        """Build NetworkX graph with node and edge features"""
        print("Building family graph...")
        
        # Add nodes (individuals)
        for person_id, person_data in self.individuals.items():
            self.graph.add_node(person_id)
            self.node_features[person_id] = self._extract_person_features(person_data)
        
        # Add edges (relationships)
        edge_count = 0
        
        # Parent-child edges
        for rel in self.relationships.get('parent_child', []):
            if rel['parent'] in self.graph and rel['child'] in self.graph:
                self.graph.add_edge(rel['parent'], rel['child'])
                self.edge_features[(rel['parent'], rel['child'])] = {
                    'type': 'parent',
                    'parent_type': rel['parent_type'],
                    'weight': 1.0
                }
                edge_count += 1
        
        # Spousal edges
        for rel in self.relationships.get('spousal', []):
            if rel['spouse1'] in self.graph and rel['spouse2'] in self.graph:
                self.graph.add_edge(rel['spouse1'], rel['spouse2'])
                self.edge_features[(rel['spouse1'], rel['spouse2'])] = {
                    'type': 'spouse',
                    'weight': 0.8
                }
                edge_count += 1
        
        # Sibling edges
        for rel in self.relationships.get('siblings', []):
            if rel['sibling1'] in self.graph and rel['sibling2'] in self.graph:
                self.graph.add_edge(rel['sibling1'], rel['sibling2'])
                self.edge_features[(rel['sibling1'], rel['sibling2'])] = {
                    'type': 'sibling',
                    'weight': 0.6
                }
                edge_count += 1
        
        print(f"Built graph with {self.graph.number_of_nodes()} nodes and {edge_count} edges")
        return self.graph, self.node_features, self.edge_features
    
    def _extract_person_features(self, person_data):
        """Extract numerical features for a person"""
        features = []
        
        # Gender (0=unknown, 1=male, 2=female)
        gender = person_data.get('gender', 'U')
        features.append(1 if gender == 'M' else 2 if gender == 'F' else 0)
        
        # Has birth date
        features.append(1 if person_data.get('birth_date') else 0)
        
        # Has death date
        features.append(1 if person_data.get('death_date') else 0)
        
        # Birth year (normalized)
        birth_year = 0
        if person_data.get('birth_date'):
            year_match = re.search(r'\\b(\\d{4})\\b', person_data['birth_date'])
            if year_match:
                year = int(year_match.group(1))
                if 1700 <= year <= 2024:  # Reasonable year range
                    birth_year = (year - 1800) / 224  # Normalize to ~0-1
        features.append(birth_year)
        
        # Name features
        first_name = person_data.get('first_name', '')
        last_name = person_data.get('last_name', '')
        
        # Name lengths (normalized)
        features.append(min(len(first_name) / 20, 1))
        features.append(min(len(last_name) / 20, 1))
        
        # Has names
        features.append(1 if first_name else 0)
        features.append(1 if last_name else 0)
        
        # Name complexity (vowel ratios)
        vowels = 'aeiouAEIOU'
        first_vowel_ratio = len([c for c in first_name if c in vowels]) / len(first_name) if first_name else 0
        last_vowel_ratio = len([c for c in last_name if c in vowels]) / len(last_name) if last_name else 0
        features.append(first_vowel_ratio)
        features.append(last_vowel_ratio)
        
        # Geographic features (birth place)
        birth_place = person_data.get('birth_place', '')
        features.append(1 if 'USA' in birth_place or 'United States' in birth_place else 0)
        features.append(1 if any(state in birth_place for state in ['California', 'Utah', 'Hawaii']) else 0)
        
        # Ensure fixed feature size
        while len(features) < 12:
            features.append(0)
        
        return np.array(features[:12], dtype=np.float32)

class GATModel:
    """Graph Attention Network for parent prediction"""
    
    def __init__(self, feature_dim=12, hidden_dim=64, num_heads=8, num_classes=3):
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.num_classes = num_classes
        self.model = None
    
    def build_model(self, max_nodes):
        """Build the GAT model architecture"""
        print(f"Building GAT model for max {max_nodes} nodes...")
        
        # Input layers
        node_features = keras.Input(shape=(max_nodes, self.feature_dim), name='node_features')
        adjacency = keras.Input(shape=(max_nodes, max_nodes), name='adjacency')
        
        # Graph Attention Layers
        x = self._gat_layer(self.hidden_dim, self.num_heads, 'gat1')(node_features, adjacency)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)
        
        x = self._gat_layer(self.hidden_dim // 2, self.num_heads // 2, 'gat2')(x, adjacency)
        x = layers.LayerNormalization()(x)
        x = layers.Activation('relu')(x)
        x = layers.Dropout(0.1)(x)
        
        # Global pooling and prediction
        x = layers.GlobalAveragePooling1D()(x)
        x = layers.Dense(64, activation='relu')(x)
        x = layers.Dropout(0.2)(x)
        x = layers.Dense(32, activation='relu')(x)
        
        # Output: [has_missing_parent, missing_father, missing_mother]
        outputs = layers.Dense(self.num_classes, activation='sigmoid', name='predictions')(x)
        
        self.model = keras.Model(inputs=[node_features, adjacency], outputs=outputs)
        
        self.model.compile(
            optimizer=keras.optimizers.Adam(learning_rate=0.001),
            loss='binary_crossentropy',
            metrics=['accuracy', 'precision', 'recall']
        )
        
        return self.model
    
    def _gat_layer(self, units, num_heads, name):
        """Create a Graph Attention Layer"""
        def gat_layer_fn(node_features, adjacency):
            batch_size = tf.shape(node_features)[0]
            num_nodes = tf.shape(node_features)[1]
            
            # Linear projections
            query = layers.Dense(units * num_heads, name=f'{name}_query')(node_features)
            key = layers.Dense(units * num_heads, name=f'{name}_key')(node_features)
            value = layers.Dense(units * num_heads, name=f'{name}_value')(node_features)
            
            # Reshape for multi-head attention
            query = tf.reshape(query, [batch_size, num_nodes, num_heads, units])
            key = tf.reshape(key, [batch_size, num_nodes, num_heads, units])
            value = tf.reshape(value, [batch_size, num_nodes, num_heads, units])
            
            # Attention scores
            attention_scores = tf.matmul(query, key, transpose_b=True) / tf.sqrt(tf.cast(units, tf.float32))
            
            # Apply adjacency mask
            adjacency_mask = tf.expand_dims(adjacency, axis=2)  # [batch, nodes, 1, nodes]
            adjacency_mask = tf.tile(adjacency_mask, [1, 1, num_heads, 1])  # [batch, nodes, heads, nodes]
            
            # Mask attention scores
            attention_scores = tf.where(adjacency_mask > 0, attention_scores, -1e9)
            
            # Softmax
            attention_weights = tf.nn.softmax(attention_scores, axis=-1)
            
            # Apply attention to values
            attended_values = tf.matmul(attention_weights, value)  # [batch, nodes, heads, units]
            
            # Concatenate heads
            output = tf.reshape(attended_values, [batch_size, num_nodes, units * num_heads])
            
            # Final projection
            output = layers.Dense(units, name=f'{name}_output')(output)
            
            return output
        
        return gat_layer_fn

def create_training_data(individuals, relationships):
    """Create training examples from family data"""
    print("Creating training data...")
    
    # Find individuals with missing parents
    training_examples = []
    
    # Get parent-child relationships
    parent_child_rels = relationships.get('parent_child', [])
    
    # Build parent lookup
    child_to_parents = defaultdict(list)
    for rel in parent_child_rels:
        child_to_parents[rel['child']].append({
            'parent': rel['parent'],
            'type': rel['parent_type']
        })
    
    # Create training examples
    for person_id, person_data in individuals.items():
        parents = child_to_parents[person_id]
        
        has_father = any(p['type'] == 'father' for p in parents)
        has_mother = any(p['type'] == 'mother' for p in parents)
        
        # Label: [has_missing_parent, missing_father, missing_mother]
        label = [
            1 if not has_father or not has_mother else 0,  # has missing parent
            1 if not has_father else 0,  # missing father
            1 if not has_mother else 0   # missing mother
        ]
        
        training_examples.append({
            'person_id': person_id,
            'label': label,
            'has_father': has_father,
            'has_mother': has_mother
        })
    
    print(f"Created {len(training_examples)} training examples")
    
    # Statistics
    missing_both = sum(1 for ex in training_examples if not ex['has_father'] and not ex['has_mother'])
    missing_father = sum(1 for ex in training_examples if not ex['has_father'] and ex['has_mother'])
    missing_mother = sum(1 for ex in training_examples if ex['has_father'] and not ex['has_mother'])
    complete = sum(1 for ex in training_examples if ex['has_father'] and ex['has_mother'])
    
    print(f"Missing both parents: {missing_both}")
    print(f"Missing father only: {missing_father}")
    print(f"Missing mother only: {missing_mother}")
    print(f"Complete families: {complete}")
    
    return training_examples

def prepare_graph_data(graph, node_features, training_examples, max_nodes=500):
    """Prepare graph data for training"""
    print(f"Preparing graph data (max {max_nodes} nodes)...")
    
    # Get all connected components
    components = list(nx.connected_components(graph))
    components.sort(key=len, reverse=True)
    
    print(f"Found {len(components)} connected components")
    print(f"Largest component: {len(components[0])} nodes")
    
    X_features = []
    X_adjacency = []
    y = []
    
    # Process each component
    for comp_idx, component in enumerate(components[:50]):  # Limit to 50 largest components
        if len(component) < 3:  # Skip very small components
            continue
        
        # Get subgraph
        subgraph = graph.subgraph(component)
        nodes = list(subgraph.nodes())
        
        # Limit size
        if len(nodes) > max_nodes:
            nodes = nodes[:max_nodes]
            subgraph = graph.subgraph(nodes)
        
        # Create node mapping
        node_to_idx = {node: idx for idx, node in enumerate(nodes)}
        
        # Node features matrix
        features_matrix = np.zeros((max_nodes, 12))
        for i, node in enumerate(nodes):
            if node in node_features:
                features_matrix[i] = node_features[node]
        
        # Adjacency matrix
        adj_matrix = np.zeros((max_nodes, max_nodes))
        for edge in subgraph.edges():
            if edge[0] in node_to_idx and edge[1] in node_to_idx:
                i, j = node_to_idx[edge[0]], node_to_idx[edge[1]]
                adj_matrix[i, j] = 1
                adj_matrix[j, i] = 1
        
        # Labels for this component
        component_labels = [0, 0, 0]  # Default: no missing parents
        
        for example in training_examples:
            if example['person_id'] in component:
                # Update component-level labels
                if example['label'][0]:  # has missing parent
                    component_labels[0] = 1
                if example['label'][1]:  # missing father
                    component_labels[1] = 1
                if example['label'][2]:  # missing mother
                    component_labels[2] = 1
        
        X_features.append(features_matrix)
        X_adjacency.append(adj_matrix)
        y.append(component_labels)
    
    print(f"Prepared {len(X_features)} graph examples")
    
    return np.array(X_features), np.array(X_adjacency), np.array(y)

def main():
    """Main training pipeline"""
    print("=== Ohana AI Model Training ===")
    print(f"Started at: {datetime.now()}")
    
    # Parse GEDCOM file
    parser = GedcomParser("Hussey Ohana.ged.txt")
    individuals, families, relationships = parser.parse()
    
    # Build graph
    graph_builder = FamilyGraphBuilder(individuals, families, relationships)
    graph, node_features, edge_features = graph_builder.build_graph()
    
    # Create training data
    training_examples = create_training_data(individuals, relationships)
    
    # Prepare graph data
    X_features, X_adjacency, y = prepare_graph_data(graph, node_features, training_examples)
    
    print(f"Final dataset: {X_features.shape[0]} examples")
    print(f"Feature shape: {X_features.shape}")
    print(f"Adjacency shape: {X_adjacency.shape}")
    print(f"Label shape: {y.shape}")
    
    # Split data
    X_feat_train, X_feat_test, X_adj_train, X_adj_test, y_train, y_test = train_test_split(
        X_features, X_adjacency, y, test_size=0.2, random_state=42, stratify=y[:, 0]
    )
    
    print(f"Training set: {X_feat_train.shape[0]} examples")
    print(f"Test set: {X_feat_test.shape[0]} examples")
    
    # Build and train model
    gat_model = GATModel()
    model = gat_model.build_model(max_nodes=X_features.shape[1])
    
    print("Model architecture:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=20, restore_best_weights=True, monitor='val_loss'),
        keras.callbacks.ReduceLROnPlateau(patience=10, factor=0.5, monitor='val_loss'),
        keras.callbacks.ModelCheckpoint('best_ohana_model.h5', save_best_only=True, monitor='val_loss')
    ]
    
    # Train model
    print("Training model...")
    history = model.fit(
        [X_feat_train, X_adj_train], y_train,
        validation_data=([X_feat_test, X_adj_test], y_test),
        epochs=100,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("\\nEvaluating model...")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(
        [X_feat_test, X_adj_test], y_test, verbose=0
    )
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    
    # Generate predictions for analysis
    y_pred = model.predict([X_feat_test, X_adj_test])
    y_pred_binary = (y_pred > 0.5).astype(int)
    
    # Detailed metrics
    for i, label_name in enumerate(['Has Missing Parent', 'Missing Father', 'Missing Mother']):
        if np.sum(y_test[:, i]) > 0:  # Only if we have positive examples
            precision, recall, f1, _ = precision_recall_fscore_support(
                y_test[:, i], y_pred_binary[:, i], average='binary'
            )
            auc = roc_auc_score(y_test[:, i], y_pred[:, i])
            
            print(f"\\n{label_name}:")
            print(f"  Precision: {precision:.4f}")
            print(f"  Recall: {recall:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  AUC: {auc:.4f}")
    
    # Save model for TensorFlow.js
    print("\\nSaving model...")
    os.makedirs('models/parent_predictor', exist_ok=True)
    
    # Save in native Keras format
    model.save('models/parent_predictor/ohana_model.h5')
    
    # Also save model architecture and weights separately for TensorFlow.js conversion
    model_json = model.to_json()
    with open('models/parent_predictor/model_architecture.json', 'w') as f:
        json.dump(json.loads(model_json), f, indent=2)
    
    model.save_weights('models/parent_predictor/model_weights.h5')
    
    print("\\nTraining completed successfully!")
    print("Model saved to models/parent_predictor/")
    print("\\nTo convert to TensorFlow.js format, run:")
    print("tensorflowjs_converter --input_format=keras models/parent_predictor/ohana_model.h5 models/parent_predictor/")
    
    # Plot training history
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    
    plt.subplot(1, 2, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig('models/parent_predictor/training_history.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    print(f"\\nCompleted at: {datetime.now()}")

if __name__ == "__main__":
    main()