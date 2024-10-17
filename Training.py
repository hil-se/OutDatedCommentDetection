import tensorflow as tf
import numpy as np
import pandas as pd
from DataReader import Reader
import os

# Load embeddings using DataReader
embedding_reader = Reader("cbow")
embeddings = embedding_reader.load("embeddings/java_train")  # Replace with the actual file name (without .csv)

# Unpack loaded embeddings
source_series_old = np.array(embeddings["source_old"])
target_series_old = np.array(embeddings["target_old"])
source_series_new = np.array(embeddings["source_new"])
target_series_new = np.array(embeddings["target_new"])

# Ensure all embeddings have the same shape
if source_series_old.shape[1] != target_series_old.shape[1] or \
   source_series_old.shape[1] != target_series_new.shape[1]:
    raise ValueError("The dimensions of the embeddings do not match!")

# Define the input shape (embedding size)
embedding_dim = source_series_old.shape[1]

# Define the encoder model
def create_encoder(input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    dense_layer = tf.keras.layers.Dense(embedding_dim, activation='relu')(input_layer)
    norm_layer = tf.keras.layers.LayerNormalization(axis=1)(dense_layer)
    return tf.keras.Model(inputs=input_layer, outputs=norm_layer)

# Create two encoders
encoder_source = create_encoder((embedding_dim,))
encoder_target = create_encoder((embedding_dim,))

# Inputs to the encoders
input_source = tf.keras.Input(shape=(embedding_dim,))
input_target = tf.keras.Input(shape=(embedding_dim,))

# Get the outputs of the encoders
encoded_source = encoder_source(input_source)
encoded_target = encoder_target(input_target)

# Calculate cosine similarity
cosine_similarity = tf.keras.layers.Dot(axes=1, normalize=False)([encoded_source, encoded_target])

# Build and compile the model
model = tf.keras.Model(inputs=[input_source, input_target], outputs=cosine_similarity)
model.compile(optimizer='adam', loss='mean_squared_error')

# ------------------------
# Prepare Data for One-Step Training
# ------------------------

# Repeat source_series_old twice
input_source_combined = np.vstack([source_series_old, source_series_old])

# Concatenate target_series_old and target_series_new
input_target_combined = np.vstack([target_series_old, target_series_new])

# Create labels: ones for (source_series_old, target_series_old)
labels_old = np.ones(source_series_old.shape[0])

# Calculate cosine similarities for (source_series_old, source_series_new)
cosine_labels_new = np.array([
    np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))
    for a, b in zip(source_series_old, source_series_new)
])

# Concatenate all labels
combined_labels = np.concatenate([labels_old, cosine_labels_new])

# ------------------------
# Train the Model in One Step
# ------------------------
print("Starting one-step training...")
model.fit([input_source_combined, input_target_combined], combined_labels, epochs=10, batch_size=32)

# ------------------------
# Save the Model
# ------------------------
model_save_path = "model_cosine_similarity.h5"
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")


