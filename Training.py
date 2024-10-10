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
target_series_new = np.array(embeddings["target_new"])

# Ensure that all the embeddings have the correct shape
if source_series_old.shape[1] != target_series_old.shape[1] or source_series_old.shape[1] != target_series_new.shape[1]:
    raise ValueError("The dimensions of the embeddings do not match!")

# Define the input shape (embedding size)
embedding_dim = source_series_old.shape[1]

# Define the encoder model
def create_encoder(input_shape):
    input_layer = tf.keras.Input(shape=input_shape)
    dense_layer = tf.keras.layers.Dense(embedding_dim, activation='relu')(input_layer)
    norm_layer = tf.keras.layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(dense_layer)
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

# Labels: Cosine similarity between source_series_old and target_series_old is labeled as 1
labels_old = np.ones(source_series_old.shape[0])

# Train the model with source_series_old and target_series_old embeddings
model.fit([source_series_old, target_series_old], labels_old, epochs=10, batch_size=32)

# Saving the model
model_save_path = "model_cosine_similarity.h5"  # Path to save the model
model.save(model_save_path)
print(f"Model saved to: {model_save_path}")

# Testing: Use the same encoder for source_series_old and the new target embeddings
target_encoder_new = create_encoder((embedding_dim,))

# Get the encoded representations for the new target series
encoded_target_new = target_encoder_new(target_series_new)

# Calculate cosine similarity between source_series_old and encoded_target_new
predicted_similarity = model.predict([source_series_old, encoded_target_new])

# Print the predicted similarities
print("Cosine similarity between source_series_old and target_series_new:")
print(predicted_similarity)

# Save the predicted similarities to a CSV file
results_df = pd.DataFrame(predicted_similarity, columns=["Cosine Similarity"])
results_csv_path = "predicted_cosine_similarity.csv"  # Path to save the results
results_df.to_csv(results_csv_path, index=False)
print(f"Predicted cosine similarity saved to: {results_csv_path}")
