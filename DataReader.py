import pandas as pd
import numpy as np

class Reader:
    def __init__(self, folder="embeddings", embedding="", testset=False):
        if testset:
            self.path = "Results/Tensors/"
        else:
            self.path = "Data/" + embedding + "/" + folder + "/"

    def load(self, name):
        # Load embeddings and label from the CSV file
        self.source_old, self.target_old, self.source_new, self.target_new, self.labels = \
            self.load_embeddings(self.path + name + ".csv")

        # Store features in a dictionary for easy access
        self.feature = {
            "source_old": self.source_old,
            "target_old": self.target_old,
            "source_new": self.source_new,
            "target_new": self.target_new,
            "label": self.labels  # Include labels in the features
        }
        self.size = len(self.source_old)
        return self.feature

    def load_embeddings(self, file):
        def parse(emb_str):
            """Helper function to parse embedding strings into float arrays."""
            emb_str = str(emb_str).replace(",", "")
            emb_list = emb_str[1:-1].split()
            return [float(emb.strip()) for emb in emb_list]

        # Load the CSV file into a DataFrame
        df = pd.read_csv(file).dropna()

        # Extract and parse embeddings for each column
        source_old = np.array([parse(emb) for emb in df["Source_Old"]])
        target_old = np.array([parse(emb) for emb in df["Target_Old"]])
        source_new = np.array([parse(emb) for emb in df["Source_New"]])
        target_new = np.array([parse(emb) for emb in df["Target_New"]])

        # Extract the label column (converted to boolean)
        labels = df["Label"].astype(bool).to_numpy()

        return source_old, target_old, source_new, target_new, labels
