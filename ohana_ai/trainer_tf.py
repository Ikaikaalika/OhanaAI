"""
TensorFlow training pipeline for OhanaAI model.
"""

import logging
import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import tensorflow as tf
import yaml

from .gedcom_parser import Family, Individual
from .gnn_model_tf import OhanaAIModelTF
from .graph_builder import GraphBuilder, GraphData

logger = logging.getLogger(__name__)


@dataclass
class TrainingMetrics:
    epoch: int
    train_loss: float
    val_loss: float
    train_accuracy: float
    val_accuracy: float
    learning_rate: float
    epoch_time: float


class EarlyStopping:
    def __init__(self, patience: int = 5, min_delta: float = 0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_loss = float("inf")
        self.early_stop = False

    def __call__(self, val_loss: float):
        if val_loss < self.best_loss - self.min_delta:
            self.best_loss = val_loss
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


class OhanaAITrainerTF:
    def __init__(self, config_path: str = "config.yaml"):
        with open(config_path, "r") as f:
            self.config = yaml.safe_load(f)

        self.model = None
        self.optimizer = None
        self.loss_fn = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        self.early_stopping = EarlyStopping(patience=self.config["training"]["patience"])

        self.graph_data = None
        self.all_pairs = None
        self.all_labels = None
        self.train_dataset = None
        self.val_dataset = None

        self.training_history: List[TrainingMetrics] = []
        self._setup_logging()

    def _setup_logging(self):
        log_dir = self.config["paths"]["logs"]
        os.makedirs(log_dir, exist_ok=True)
        log_file = os.path.join(log_dir, "training_tf.log")
        logging.basicConfig(
            level=getattr(logging, self.config["logging"]["level"]),
            format=self.config["logging"]["format"],
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

    def prepare_data(self, individuals: Dict[str, Individual], families: Dict[str, Family]):
        logger.info("Preparing training data...")
        graph_builder = GraphBuilder()
        self.graph_data = graph_builder.build_graph(individuals, families)

        positive_pairs, negative_pairs = graph_builder.create_training_pairs(self.graph_data, individuals, families)

        self.all_pairs = np.vstack([positive_pairs, negative_pairs])
        self.all_labels = np.hstack([np.ones(len(positive_pairs)), np.zeros(len(negative_pairs))])

        self._create_train_val_split()
        logger.info(f"Training data prepared: {len(self.train_dataset)} train, {len(self.val_dataset)} val samples")

    def _create_train_val_split(self):
        val_split = self.config["training"]["validation_split"]
        indices = np.arange(len(self.all_pairs))
        np.random.shuffle(indices)

        val_size = int(len(indices) * val_split)
        val_indices = indices[:val_size]
        train_indices = indices[val_size:]

        train_pairs = self.all_pairs[train_indices]
        train_labels = self.all_labels[train_indices]
        val_pairs = self.all_pairs[val_indices]
        val_labels = self.all_labels[val_indices]

        batch_size = self.config["training"]["batch_size"]
        self.train_dataset = tf.data.Dataset.from_tensor_slices((train_pairs, train_labels)).batch(batch_size)
        self.val_dataset = tf.data.Dataset.from_tensor_slices((val_pairs, val_labels)).batch(batch_size)

    def initialize_model(self):
        logger.info("Initializing model and optimizer...")
        model_config = self.config["model"].copy()
        model_config.pop("edge_types", None)
        self.model = OhanaAIModelTF(**model_config)
        learning_rate = self.config["training"]["learning_rate"]
        self.optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

    @tf.function
    def train_step(self, batch_pairs, batch_labels):
        with tf.GradientTape() as tape:
            predictions = self.model([self.graph_data.node_features, self.graph_data.edge_index, self.graph_data.edge_types, batch_pairs], training=True)
            loss = self.loss_fn(batch_labels, predictions)
        
        gradients = tape.gradient(loss, self.model.trainable_variables)
        self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables))
        
        accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(batch_labels, predictions))
        return loss, accuracy

    def train_epoch(self):
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_pairs, batch_labels in self.train_dataset:
            loss, accuracy = self.train_step(batch_pairs, batch_labels)
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

        return total_loss / num_batches, total_accuracy / num_batches

    @tf.function
    def validation_step(self, batch_pairs, batch_labels):
        predictions = self.model([self.graph_data.node_features, self.graph_data.edge_index, self.graph_data.edge_types, batch_pairs], training=False)
        loss = self.loss_fn(batch_labels, predictions)
        accuracy = tf.reduce_mean(tf.keras.metrics.binary_accuracy(batch_labels, predictions))
        return loss, accuracy

    def validate(self):
        total_loss = 0.0
        total_accuracy = 0.0
        num_batches = 0

        for batch_pairs, batch_labels in self.val_dataset:
            loss, accuracy = self.validation_step(batch_pairs, batch_labels)
            total_loss += loss
            total_accuracy += accuracy
            num_batches += 1

        return total_loss / num_batches, total_accuracy / num_batches

    def train(self, num_epochs: Optional[int] = None):
        if num_epochs is None:
            num_epochs = self.config["training"]["epochs"]

        if self.model is None:
            self.initialize_model()

        logger.info(f"Starting training for {num_epochs} epochs...")

        for epoch in range(num_epochs):
            start_time = time.time()

            train_loss, train_acc = self.train_epoch()
            val_loss, val_acc = self.validate()

            epoch_time = time.time() - start_time

            metrics = TrainingMetrics(
                epoch=epoch + 1,
                train_loss=train_loss.numpy(),
                val_loss=val_loss.numpy(),
                train_accuracy=train_acc.numpy(),
                val_accuracy=val_acc.numpy(),
                learning_rate=self.optimizer.learning_rate.numpy(),
                epoch_time=epoch_time,
            )
            self.training_history.append(metrics)

            logger.info(
                f"Epoch {epoch+1}/{num_epochs} - "
                f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}, "
                f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}, "
                f"Time: {epoch_time:.2f}s"
            )

            self.early_stopping(val_loss)
            if self.early_stopping.early_stop:
                logger.info(f"Early stopping triggered at epoch {epoch+1}")
                break

            if (epoch + 1) % 10 == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch+1}")

        self.save_model()
        logger.info("Training completed!")
        return self.training_history

    def save_checkpoint(self, filename: str):
        checkpoint_dir = self.config["paths"]["checkpoints"]
        os.makedirs(checkpoint_dir, exist_ok=True)
        checkpoint_path = os.path.join(checkpoint_dir, filename)
        self.model.save_weights(checkpoint_path)
        logger.info(f"Checkpoint saved: {checkpoint_path}")

    def save_model(self):
        model_path = self.config["paths"]["models"]
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        self.model.save(model_path)
        logger.info(f"Model saved: {model_path}")

    def load_model(self, model_path: Optional[str] = None):
        if model_path is None:
            model_path = self.config["paths"]["models"]

        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model file not found: {model_path}")

        if self.model is None:
            self.initialize_model()

        self.model = tf.keras.models.load_model(model_path)
        logger.info(f"Model loaded: {model_path}")
