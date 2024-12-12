import tensorflow as tf
import numpy as np
import pandas as pd
from DataReader import Reader
import os



# Custom L2 normalization layer to avoid Lambda issues
class L2Normalization(tf.keras.layers.Layer):
    def call(self, inputs):
        return tf.math.l2_normalize(inputs, axis=1)

    def compute_output_shape(self, input_shape):
        return input_shape



# class EuclideanDistanceLayer(tf.keras.layers.Layer):
#     def call(self, inputs):
#         source, target = inputs
#         return tf.sqrt(tf.reduce_sum(tf.square(source - target), axis=1))

# Load embeddings using DataReader
embedding_reader = Reader("cbow")
embeddings_true = embedding_reader.load("embeddings/java_train")  # Replace with the actual file name (without .csv)
embeddings_false = embedding_reader.load("embeddings/java_train_false_subset")
# Unpack loaded embeddings
source_series_old_true = np.array(embeddings_true["source_old"])
target_series_old_true = np.array(embeddings_true["target_old"])
source_series_new_true = np.array(embeddings_true["source_new"])
target_series_new_true = np.array(embeddings_true["target_new"])


# Unpack loaded embeddings
source_series_old_false = np.array(embeddings_false["source_old"])
target_series_old_false = np.array(embeddings_false["target_old"])
source_series_new_false = np.array(embeddings_false["source_new"])
target_series_new_false = np.array(embeddings_false["target_new"])



# Ensure all embeddings have the same shape
if source_series_old_true.shape[1] != target_series_old_true.shape[1] or \
   source_series_old_true.shape[1] != target_series_new_true.shape[1]:
    raise ValueError("The dimensions of the embeddings do not match!")


# Ensure all embeddings have the same shape
if source_series_old_false.shape[1] != target_series_old_false.shape[1] or \
   source_series_old_false.shape[1] != target_series_new_false.shape[1]:
    raise ValueError("The dimensions of the embeddings do not match!")

# Define the input shape (embedding size)
embedding_dim_true = source_series_old_true.shape[1]
embedding_dim_false = source_series_old_false.shape[1]


print(embedding_dim_true)
print(embedding_dim_false)



# Create an encoder model using the custom normalization layer
def create_encoder(input_shape, embedding_dim_true):
    input_layer = tf.keras.Input(shape=input_shape)

    # Double the size of the dense layer
    dense_layer1 = tf.keras.layers.Dense(4 * embedding_dim_true, activation='relu')(input_layer)
    dense_layer2 = tf.keras.layers.Dense(2 * embedding_dim_true, activation='relu')(dense_layer1)
    reduced_layer = tf.keras.layers.Dense(embedding_dim_true, activation='relu')(dense_layer2)
    norm_layer = L2Normalization()(reduced_layer)
    return tf.keras.Model(inputs=input_layer, outputs=norm_layer)




# Create two encoders
encoder_source = create_encoder((embedding_dim_true,), embedding_dim_true)
encoder_target = create_encoder((embedding_dim_true,), embedding_dim_true)

# Inputs to the encoders
input_source = tf.keras.Input(shape=(embedding_dim_true,))
input_target = tf.keras.Input(shape=(embedding_dim_true,))

# Get the outputs of the encoders
encoded_source = encoder_source(input_source)
encoded_target = encoder_target(input_target)




# Then you can just use this pre-calculated distance in the model
# distance_calculation = EuclideanDistanceLayer()([encoded_source, encoded_target])



# # Define the contrastive loss function
# def contrastive_loss(y_true, y_pred, margin=1.0):
#     # y_pred represents the Euclidean distance between encoded pairs
#     square_pred = tf.square(y_pred)
#     margin_square = tf.square(tf.maximum(margin - y_pred, 0))
#     return tf.reduce_mean(y_true * square_pred + (1 - y_true) * margin_square)



# Calculate cosine similarity
cosine_similarity = tf.keras.layers.Dot(axes=1, normalize=False)([encoded_source, encoded_target])


# Build and compile the model
model = tf.keras.Model(inputs=[input_source, input_target], outputs=cosine_similarity)

model.compile(optimizer='adam', loss='mean_squared_error')

# ------------------------
# Prepare Data for One-Step Training
# ------------------------

# Repeat source_series_old twice
input_source_combined = np.vstack([source_series_old_true, source_series_old_true, source_series_old_false, source_series_old_false])

# Concatenate target_series_old and target_series_new
input_target_combined = np.vstack([target_series_old_true, target_series_new_true, target_series_old_false, target_series_new_false])

# Create labels: ones for (source_series_old, target_series_old)
labels_old_true = np.ones(source_series_old_true.shape[0])
labels_old_false = np.ones(source_series_old_false.shape[0])
labels_new_false = np.ones(source_series_old_false.shape[0])

# Calculate cosine similarities for (source_series_old, source_series_new)
# epsilon = 1e-10
# cosine_labels_new = np.array([
#     np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + epsilon)
#     for a, b in zip(source_series_old_true, source_series_new_true)
# ])

cosine_labels_new = np.zeros(source_series_old_true.shape[0])


# Concatenate all labels
combined_labels = np.concatenate([labels_old_true, cosine_labels_new, labels_old_false, labels_new_false])
assert combined_labels.shape[0] == input_source_combined.shape[0]


# ------------------------
# Train the Model in One Step
# ------------------------
print("Starting one-step training...")
model.fit([input_source_combined, input_target_combined], combined_labels, epochs=20, batch_size=32)

# ------------------------
# Save the Model
# ------------------------
model_save_path = "model_cosine_similarity.h5"
model.save(model_save_path, save_format='h5')
print(f"Model saved to: {model_save_path}")


