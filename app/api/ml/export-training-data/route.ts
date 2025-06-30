import { NextRequest, NextResponse } from 'next/server'
import { db } from '@/lib/db'
import { mlTrainingData, gedcomFiles } from '@/lib/db/schema'
import { eq } from 'drizzle-orm'
import { writeFile, mkdir } from 'fs/promises'
import { join } from 'path'

// This endpoint should be secured or called via a cron job
export async function POST(request: NextRequest) {
  try {
    // Basic security check - in production, use proper authentication
    const { authorization } = await request.json()
    
    if (authorization !== process.env.EXPORT_SECRET) {
      return NextResponse.json(
        { error: 'Unauthorized' },
        { status: 401 }
      )
    }

    // Get all training data that hasn't been exported yet
    const trainingData = await db
      .select({
        id: mlTrainingData.id,
        gedcomFileId: mlTrainingData.gedcomFileId,
        graphData: mlTrainingData.graphData,
        labels: mlTrainingData.labels,
        exportedAt: mlTrainingData.exportedAt,
        includedInTraining: mlTrainingData.includedInTraining
      })
      .from(mlTrainingData)
      .where(eq(mlTrainingData.includedInTraining, false))

    if (trainingData.length === 0) {
      return NextResponse.json({
        message: 'No new training data to export',
        count: 0
      })
    }

    // Create training data directory
    const trainingDir = join(process.cwd(), 'training_data')
    try {
      await mkdir(trainingDir, { recursive: true })
    } catch (error) {
      // Directory might already exist
    }

    // Export data in batches
    const batchSize = 100
    const timestamp = new Date().toISOString().replace(/[:.]/g, '-')
    
    for (let i = 0; i < trainingData.length; i += batchSize) {
      const batch = trainingData.slice(i, i + batchSize)
      const batchNumber = Math.floor(i / batchSize) + 1
      
      const exportData = {
        metadata: {
          exportedAt: new Date().toISOString(),
          batchNumber,
          totalBatches: Math.ceil(trainingData.length / batchSize),
          count: batch.length
        },
        data: batch.map(item => ({
          id: item.id,
          gedcomFileId: item.gedcomFileId,
          graphData: item.graphData,
          labels: item.labels
        }))
      }

      const filename = `training_batch_${batchNumber}_${timestamp}.json`
      const filepath = join(trainingDir, filename)
      
      await writeFile(filepath, JSON.stringify(exportData, null, 2))
    }

    // Mark data as exported (but not yet included in training)
    const exportedIds = trainingData.map(item => item.id)
    // Note: This would need proper SQL IN clause handling
    // For now, we'll mark them one by one (inefficient but works)
    for (const id of exportedIds) {
      await db.update(mlTrainingData)
        .set({ exportedAt: new Date() })
        .where(eq(mlTrainingData.id, id))
    }

    // Generate training script
    const trainingScript = generateTrainingScript(trainingData.length)
    await writeFile(join(trainingDir, 'run_training.py'), trainingScript)
    
    // Generate requirements file
    const requirements = generateRequirements()
    await writeFile(join(trainingDir, 'requirements.txt'), requirements)

    return NextResponse.json({
      message: 'Training data exported successfully',
      count: trainingData.length,
      batches: Math.ceil(trainingData.length / batchSize),
      directory: trainingDir
    })

  } catch (error) {
    console.error('Export error:', error)
    return NextResponse.json(
      { error: 'Failed to export training data' },
      { status: 500 }
    )
  }
}

function generateTrainingScript(dataCount: number): string {
  return `#!/usr/bin/env python3
"""
Ohana AI - Parent Prediction Model Training Script
This script trains a Graph Neural Network (GNN) or Graph Attention Network (GAT) 
to predict missing parents in family trees.

Generated automatically on ${new Date().toISOString()}
Data samples: ${dataCount}
"""

import os
import json
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import networkx as nx
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
import glob

class GATLayer(layers.Layer):
    """Graph Attention Network Layer"""
    
    def __init__(self, units, num_heads=8, dropout_rate=0.1, **kwargs):
        super(GATLayer, self).__init__(**kwargs)
        self.units = units
        self.num_heads = num_heads
        self.dropout_rate = dropout_rate
        
        self.linear_proj = layers.Dense(units * num_heads, use_bias=False)
        self.attn_proj = layers.Dense(num_heads, use_bias=False)
        self.dropout = layers.Dropout(dropout_rate)
        
    def call(self, inputs, training=None):
        # inputs: [node_features, adjacency_matrix]
        node_features, adj_matrix = inputs
        batch_size = tf.shape(node_features)[0]
        num_nodes = tf.shape(node_features)[1]
        
        # Linear transformation
        projected = self.linear_proj(node_features)
        projected = tf.reshape(projected, [batch_size, num_nodes, self.num_heads, self.units])
        
        # Attention mechanism
        attention_scores = self.attn_proj(node_features)
        attention_weights = tf.nn.softmax(attention_scores, axis=1)
        
        # Apply attention and aggregate
        attended = tf.reduce_sum(
            tf.expand_dims(attention_weights, -1) * projected, 
            axis=2
        )
        
        return self.dropout(attended, training=training)

def load_training_data(data_dir='./'):
    """Load all training data batches"""
    data_files = glob.glob(os.path.join(data_dir, 'training_batch_*.json'))
    
    all_graphs = []
    all_labels = []
    
    for file_path in data_files:
        with open(file_path, 'r') as f:
            batch_data = json.load(f)
            
        for item in batch_data['data']:
            graph_data = item['graphData']
            labels = item['labels']
            
            # Convert to training format
            graph_features, adjacency = process_graph_data(graph_data)
            label_vector = process_labels(labels, len(graph_data['nodes']))
            
            all_graphs.append((graph_features, adjacency))
            all_labels.append(label_vector)
    
    return all_graphs, all_labels

def process_graph_data(graph_data):
    """Convert graph data to tensor format"""
    nodes = graph_data['nodes']
    edges = graph_data['edges']
    
    num_nodes = len(nodes)
    feature_dim = len(nodes[0]['features']) if nodes else 12
    
    # Node features matrix
    features = np.zeros((num_nodes, feature_dim))
    node_id_map = {}
    
    for i, node in enumerate(nodes):
        node_id_map[node['id']] = i
        features[i] = node['features'][:feature_dim]
    
    # Adjacency matrix
    adjacency = np.zeros((num_nodes, num_nodes))
    
    for edge in edges:
        if edge['source'] in node_id_map and edge['target'] in node_id_map:
            src_idx = node_id_map[edge['source']]
            tgt_idx = node_id_map[edge['target']]
            
            # Weight based on relationship type
            weight = edge.get('weight', 1.0)
            if edge['type'] == 'parent':
                weight *= 1.0
            elif edge['type'] == 'spouse':
                weight *= 0.8
            elif edge['type'] == 'sibling':
                weight *= 0.6
                
            adjacency[src_idx][tgt_idx] = weight
            adjacency[tgt_idx][src_idx] = weight  # Undirected
    
    return features.astype(np.float32), adjacency.astype(np.float32)

def process_labels(labels, num_nodes):
    """Convert labels to training format"""
    # Create label vector: [has_missing_parent, missing_father, missing_mother]
    label_vector = np.zeros((num_nodes, 3))
    
    node_id_map = {label['personId']: i for i, label in enumerate(labels)}
    
    for i, label in enumerate(labels):
        if i < num_nodes:
            label_vector[i][0] = 1.0 if label['hasMissingParent'] else 0.0
            
            if label.get('missingParentType') == 'father':
                label_vector[i][1] = 1.0
            elif label.get('missingParentType') == 'mother':
                label_vector[i][2] = 1.0
            elif label.get('missingParentType') == 'both':
                label_vector[i][1] = 1.0
                label_vector[i][2] = 1.0
    
    return label_vector.astype(np.float32)

def create_gat_model(num_features, max_nodes):
    """Create Graph Attention Network model"""
    
    # Input layers
    node_features = keras.Input(shape=(max_nodes, num_features), name='node_features')
    adjacency = keras.Input(shape=(max_nodes, max_nodes), name='adjacency')
    
    # GAT layers
    x = GATLayer(64, num_heads=8, dropout_rate=0.1)([node_features, adjacency])
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    
    x = GATLayer(32, num_heads=4, dropout_rate=0.1)([x, adjacency])
    x = layers.LayerNormalization()(x)
    x = layers.Activation('relu')(x)
    
    # Global pooling and prediction layers
    x = layers.GlobalAveragePooling1D()(x)
    x = layers.Dense(64, activation='relu')(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Dense(32, activation='relu')(x)
    
    # Output layer: [has_missing_parent, missing_father, missing_mother]
    outputs = layers.Dense(3, activation='sigmoid', name='predictions')(x)
    
    model = keras.Model(inputs=[node_features, adjacency], outputs=outputs)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy', 'precision', 'recall']
    )
    
    return model

def main():
    print("Loading training data...")
    graphs, labels = load_training_data()
    
    if len(graphs) == 0:
        print("No training data found!")
        return
    
    print(f"Loaded {len(graphs)} training examples")
    
    # Prepare data for training
    max_nodes = max(graph[0].shape[0] for graph, _ in graphs)
    num_features = graphs[0][0].shape[1] if graphs else 12
    
    print(f"Max nodes: {max_nodes}, Features: {num_features}")
    
    # Pad graphs to same size
    X_features = []
    X_adjacency = []
    y = []
    
    for (features, adjacency), label in zip(graphs, labels):
        # Pad features and adjacency to max_nodes
        padded_features = np.zeros((max_nodes, num_features))
        padded_adjacency = np.zeros((max_nodes, max_nodes))
        
        nodes = features.shape[0]
        padded_features[:nodes] = features
        padded_adjacency[:nodes, :nodes] = adjacency
        
        X_features.append(padded_features)
        X_adjacency.append(padded_adjacency)
        
        # Global label (any missing parent in the graph)
        global_label = np.max(label, axis=0)
        y.append(global_label)
    
    X_features = np.array(X_features)
    X_adjacency = np.array(X_adjacency)
    y = np.array(y)
    
    print(f"Training data shape: {X_features.shape}, {X_adjacency.shape}, {y.shape}")
    
    # Split data
    X_feat_train, X_feat_test, X_adj_train, X_adj_test, y_train, y_test = train_test_split(
        X_features, X_adjacency, y, test_size=0.2, random_state=42
    )
    
    # Create and train model
    print("Creating GAT model...")
    model = create_gat_model(num_features, max_nodes)
    
    print("Model summary:")
    model.summary()
    
    # Callbacks
    callbacks = [
        keras.callbacks.EarlyStopping(patience=10, restore_best_weights=True),
        keras.callbacks.ReduceLROnPlateau(patience=5, factor=0.5),
        keras.callbacks.ModelCheckpoint(
            'best_model.h5', 
            save_best_only=True, 
            monitor='val_accuracy'
        )
    ]
    
    print("Training model...")
    history = model.fit(
        [X_feat_train, X_adj_train], y_train,
        validation_data=([X_feat_test, X_adj_test], y_test),
        epochs=100,
        batch_size=32,
        callbacks=callbacks,
        verbose=1
    )
    
    # Evaluate model
    print("Evaluating model...")
    test_loss, test_acc, test_prec, test_rec = model.evaluate(
        [X_feat_test, X_adj_test], y_test, verbose=0
    )
    
    print(f"Test Accuracy: {test_acc:.4f}")
    print(f"Test Precision: {test_prec:.4f}")
    print(f"Test Recall: {test_rec:.4f}")
    
    # Save model for TensorFlow.js
    print("Converting model for TensorFlow.js...")
    os.makedirs('../models/parent_predictor', exist_ok=True)
    
    # Save in TensorFlow.js format
    model.save('../models/parent_predictor/model.h5')
    
    # Convert to TensorFlow.js format (requires tensorflowjs package)
    os.system('tensorflowjs_converter --input_format=keras ../models/parent_predictor/model.h5 ../models/parent_predictor/')
    
    print("Training completed successfully!")
    print("Model saved to ../models/parent_predictor/")

if __name__ == "__main__":
    main()
`
}

function generateRequirements(): string {
  return `tensorflow>=2.13.0
tensorflow-js>=4.0.0
scikit-learn>=1.3.0
numpy>=1.24.0
networkx>=3.1
tensorflowjs>=4.0.0
`
}