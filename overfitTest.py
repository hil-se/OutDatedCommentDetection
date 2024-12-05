import tensorflow as tf
import numpy as np
from DataReader import Reader
from Training import L2Normalization
from keras.utils import custom_object_scope

# Load training and validation datasets
embedding_reader_train = Reader("cbow")
embedding_reader_valid = Reader("cbow")

# Load training embeddings
train_embeddings = embedding_reader_train.load("embeddings/java_train_all_subset")
train_source = np.array(train_embeddings["source_old"])
train_target = np.array(train_embeddings["target_new"])
train_labels = np.array(train_embeddings["label"]).astype(bool)

# Load validation embeddings
valid_embeddings = embedding_reader_valid.load("embeddings/java_valid_all_subset")
valid_source = np.array(valid_embeddings["source_old"])
valid_target = np.array(valid_embeddings["target_new"])
valid_labels = np.array(valid_embeddings["label"]).astype(bool)

# Load the trained model
model_path = "model_cosine_similarity.h5"
with custom_object_scope({'L2Normalization': L2Normalization}):
    model = tf.keras.models.load_model(model_path)

# Instantiate the MeanSquaredError class
mse = tf.keras.losses.MeanSquaredError()

# Evaluate model on training data
train_predictions = model.predict([train_source, train_target])
train_loss = mse(train_labels, train_predictions).numpy()
train_accuracy = np.mean(
    (train_predictions.flatten() < 0.8) == train_labels
)

# Evaluate model on validation data
valid_predictions = model.predict([valid_source, valid_target])
valid_loss = mse(valid_labels, valid_predictions).numpy()
valid_accuracy = np.mean(
    (valid_predictions.flatten() < 0.8) == valid_labels
)

# Output results
print(f"Training Loss: {np.mean(train_loss)}")
print(f"Training Accuracy: {train_accuracy}")
print(f"Validation Loss: {np.mean(valid_loss)}")
print(f"Validation Accuracy: {valid_accuracy}")

# Assess model performance
if train_accuracy > valid_accuracy and valid_loss > train_loss:
    print("The model is likely overfitting.")
elif train_loss > valid_loss and train_accuracy < valid_accuracy:
    print("The model is likely underfitting.")
else:
    print("The model is likely balanced.")
