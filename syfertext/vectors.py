import pickle
import os
from pathlib import Path
import numpy as np

from .utils import hash_string


class Vectors:
    def __init__(self, model_name):

        # Create the path to where the folder named 'model_name' is stored.
        dirname = str(Path.home())

        self.model_path = os.path.join(dirname, "SyferText", model_name)

        # Load the array holding the word vectors
        self.data, self.default_vector = self.load_vectors()

        # Load the mappings between word hashes and row indices in 'self.data'
        self.key2row = self.load_key2row()

    def load_vectors(self):

        vectors_path = os.path.join(self.model_path, "vectors")

        with open(vectors_path, "rb") as vectors_file:
            vectors = pickle.load(vectors_file)

        # default vector
        default_vector = np.zeros(vectors.shape[1], dtype=vectors.dtype)

        return vectors, default_vector

    def load_key2row(self):

        key2row_path = os.path.join(self.model_path, "key2row")

        with open(key2row_path, "rb") as key2row_file:
            key2row = pickle.load(key2row_file)

        return key2row

    def __getitem__(self, word):
        """takes a word as a string and returns the corresponding vector
        """

        # Create the word hash key
        key = hash_string(word)

        # if the key does not exists return default vector
        if key not in self.key2row:
            return self.default_vector

        # Get the vector row correponding to 'key
        row = self.key2row[key]

        # Get the vector
        vector = self.data[row]

        return vector
