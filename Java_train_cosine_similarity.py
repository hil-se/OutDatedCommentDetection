import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# Path to the input CSV file
input_file = './Data/embedding/CodeSearch300/java_train.csv'

# Load the CSV file
df = pd.read_csv(input_file)

# Function to compute cosine similarity between two embeddings
def cosine_sim(embedding1, embedding2):
    return cosine_similarity([embedding1], [embedding2])[0][0]

# Convert the string embeddings to numpy arrays
df['Source_Old'] = df['Source_Old'].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))
df['Source_New'] = df['Source_New'].apply(lambda x: np.array([float(i) for i in x.strip('[]').split()]))

# Compute cosine similarity for each row
df['Cosine_Similarity'] = df.apply(lambda row: cosine_sim(row['Source_Old'], row['Source_New']), axis=1)

# Select the relevant columns and save to a new CSV
output_df = df[['Source_Old', 'Source_New', 'Cosine_Similarity']]
output_file = './Data/embedding/CodeSearch300/Java_train_cosine_similarity.csv'
output_df.to_csv(output_file, index=False)

print(f"New CSV with cosine similarities saved to {output_file}")
