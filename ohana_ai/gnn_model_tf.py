"""
TensorFlow implementation of the Graph Neural Network model.
"""

import tensorflow as tf
from tensorflow.keras import layers, Model

class GraphAttentionLayer(layers.Layer):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 num_heads: int = 1,
                 num_edge_types: int = 4,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.num_edge_types = num_edge_types
        self.dropout_rate = dropout

        assert out_features % num_heads == 0
        self.head_dim = out_features // num_heads

        self.W_query = layers.Dense(out_features, use_bias=False, name="w_query")
        self.W_key = layers.Dense(out_features, use_bias=False, name="w_key")
        self.W_value = layers.Dense(out_features, use_bias=False, name="w_value")

        self.edge_type_embedding = layers.Embedding(num_edge_types, self.head_dim, name="edge_embedding")

        self.out_proj = layers.Dense(out_features, name="out_proj")
        self.layer_norm = layers.LayerNormalization()
        self.dropout = layers.Dropout(dropout)

    def call(self, inputs, training=False):
        x, edge_index, edge_types = inputs

        def with_edges_fn():
            num_nodes = tf.shape(x)[0]
            queries = self.W_query(x)
            keys = self.W_key(x)
            values = self.W_value(x)

            queries = tf.reshape(queries, (num_nodes, self.num_heads, self.head_dim))
            keys = tf.reshape(keys, (num_nodes, self.num_heads, self.head_dim))
            values = tf.reshape(values, (num_nodes, self.num_heads, self.head_dim))

            edge_embeds = self.edge_type_embedding(edge_types)
            edge_embeds = tf.expand_dims(edge_embeds, axis=1)
            edge_embeds = tf.tile(edge_embeds, [1, self.num_heads, 1])

            src_nodes, dst_nodes = edge_index[0], edge_index[1]

            src_queries = tf.gather(queries, dst_nodes)
            src_keys = tf.gather(keys, src_nodes)
            src_values = tf.gather(values, src_nodes)

            modified_keys = src_keys + edge_embeds

            attention_scores = tf.reduce_sum(src_queries * modified_keys, axis=-1) / tf.math.sqrt(float(self.head_dim))

            attention_weights = tf.nn.softmax(attention_scores, axis=0)
            attention_weights = self.dropout(attention_weights, training=training)

            attention_weights = tf.expand_dims(attention_weights, axis=-1)
            messages = attention_weights * src_values

            aggregated = tf.zeros((num_nodes, self.num_heads, self.head_dim), dtype=x.dtype)
            aggregated = tf.tensor_scatter_nd_add(aggregated, tf.expand_dims(dst_nodes, axis=-1), messages)

            aggregated = tf.reshape(aggregated, (num_nodes, self.out_features))
            output = self.out_proj(aggregated)

            if x.get_shape().as_list()[-1] == self.out_features:
                 output = self.layer_norm(output + x)
            else:
                 output = self.layer_norm(output)
            return output

        def without_edges_fn():
            transformed = self.W_value(x)
            if x.get_shape().as_list()[-1] == self.out_features:
                return self.layer_norm(transformed + x)
            return self.layer_norm(transformed)

        return tf.cond(tf.equal(tf.shape(edge_index)[1], 0),
                       true_fn=without_edges_fn,
                       false_fn=with_edges_fn)

class OhanaAIModelTF(Model):
    def __init__(self,
                 input_features: int,
                 hidden_dim: int = 256,
                 num_heads: int = 4,
                 num_layers: int = 3,
                 num_edge_types: int = 4,
                 dropout: float = 0.1,
                 **kwargs):
        super().__init__(**kwargs)
        self.input_proj = layers.Dense(hidden_dim, activation='relu')
        self.gat_layers = [GraphAttentionLayer(hidden_dim, hidden_dim, num_heads, num_edge_types, dropout) for _ in range(num_layers)]
        self.node_embedding = layers.Dense(hidden_dim)
        self.parent_predictor = tf.keras.Sequential([
            layers.Dense(hidden_dim * 2, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(hidden_dim // 2, activation='relu'),
            layers.Dropout(dropout),
            layers.Dense(1)
        ], name="parent_predictor")

    def call(self, inputs, training=False):
        x, edge_index, edge_types, candidate_pairs = inputs
        h = self.input_proj(x)

        for layer in self.gat_layers:
            h = layer([h, edge_index, edge_types], training=training)

        node_embeddings = self.node_embedding(h)

        if candidate_pairs is not None:
            return self.predict_parents(node_embeddings, candidate_pairs)

        return node_embeddings

    def predict_parents(self, node_embeddings, candidate_pairs):
        parent_embeddings = tf.gather(node_embeddings, candidate_pairs[:, 0])
        child_embeddings = tf.gather(node_embeddings, candidate_pairs[:, 1])

        pair_embeddings = tf.concat([parent_embeddings, child_embeddings], axis=1)

        scores = self.parent_predictor(pair_embeddings)
        return tf.squeeze(scores, axis=-1)
