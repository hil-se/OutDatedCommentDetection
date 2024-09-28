import pandas as pd
import numpy as np


class Reader:
    def __init__(self, folder="embeddings", embedding="", testset=False):
        if testset:
            self.path = "Results/Tensors/"
        else:
            self.path = "Data/"+embedding+"/"+folder+"/"

    def load(self, name):
        self.source_old, self.target_old, self.source_new, self.target_new= self.load_embeddings(self.path + name + ".csv")
        self.feature = {"source_old": self.source_old, "target_old": self.target_old, "source_new": self.source_new, "target_new": self.target_new}
        self.size = len(self.source_old)
        return self.feature


    def load_embeddings(self, file):
        def parse(emb_str):
            emb_str = str(emb_str)
            emb_str = emb_str.replace(",", "")
            emb_list = emb_str[1:-1].split()
            embeddings = [float(emb.strip()) for emb in emb_list]
            return embeddings
        df = pd.read_csv(file)
        df = df.dropna()

        embeddings = df["Source_Old"].apply(parse)
        self.source_series_old = embeddings
        source_old = np.array([emb for emb in embeddings])

        embeddings = df["Target_Old"].apply(parse)
        self.target_series_old = embeddings
        target_old = np.array([emb for emb in embeddings])

        embeddings = df["Source_New"].apply(parse)
        self.source_series_new = embeddings
        source_new = np.array([emb for emb in embeddings])

        embeddings = df["Target_New"].apply(parse)
        self.target_series_new = embeddings
        target_new = np.array([emb for emb in embeddings])

        return source_old, target_old, source_new, target_new
