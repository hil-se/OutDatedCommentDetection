import tensorflow as tf
import numpy as np
import pandas as pd
from DataReader import Reader
from Training import L2Normalization, contrastive_loss, EuclideanDistanceLayer

from keras.utils import custom_object_scope

# Load test embeddings and labels using DataReader
embedding_reader = Reader("cbow")  # Set testset=True if reading test data
test_embeddings = embedding_reader.load("embeddings/java_test")  # Ensure correct file path

# Unpack embeddings and labels
source_series_old = np.array(test_embeddings["source_old"])
target_series_new = np.array(test_embeddings["target_new"])
true_labels = np.array(test_embeddings["label"]).astype(bool)  # Extract labels

# Ensure embeddings are correctly loaded
print(f"Loaded {len(source_series_old)} samples for testing.")

# Load the pre-trained model
model_path = "model_cosine_similarity.h5"
with custom_object_scope({'L2Normalization': L2Normalization, 'contrastive_loss': contrastive_loss, 'EuclideanDistanceLayer': EuclideanDistanceLayer}):
    model = tf.keras.models.load_model(model_path)


# Predict cosine similarities
# predicted_similarity = model.predict([source_series_old, target_series_new])
# predicted_similarity = predicted_similarity.flatten()  # Ensure it's a 1D array
# print(predicted_similarity)


# Predict Euclidean (L2) distance
predicted_distance = model.predict([source_series_old, target_series_new])
predicted_distance = predicted_distance.flatten()  # Ensure it's a 1D array
print(predicted_distance)


# Create 'predicted_label' based on cosine similarity threshold
predicted_labels = predicted_distance < 0.5  # True if similarity < 0.5

# Create a DataFrame to store results
# results_df = pd.DataFrame({
#     "Cosine Similarity": predicted_similarity,
#     "True Label": true_labels,
#     "Predicted Label": predicted_labels
# })

results_df = pd.DataFrame({
    "Euclidean Distance": predicted_distance,
    "True Label": true_labels,
    "Predicted Label": predicted_labels
})

# Save the results to a CSV file
results_csv_path = "predicted_cosine_similarity_results.csv"
results_df.to_csv(results_csv_path, index=False)
print(f"Results saved to: {results_csv_path}")



######
def rank_error(df):
    ranks = df["True Label"][np.argsort(df["Cosine Similarity"])]
    sum = 0
    tmp = 0
    num = 0
    for label in ranks:
        if label == "True":
            sum += tmp/len(ranks)
            num += 1
        else:
            tmp += 1
    return sum / num
