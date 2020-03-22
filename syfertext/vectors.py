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
        self.data, self.default_vector = self._load_vectors()

        # Load the mappings between word hashes and row indices in 'self.data'
        self.key2row = self._load_key2row()

    def _load_vectors(self):
        """Loads the embedding vectors of the vocabulary string from disk.
           
        Returns:
            vectors (array): a numpy array which as much rows as words in the vocabulary.
                                the number of columns is equal to the vector's dimensions.
            default_vector (array): a numpy array of size (number of vector's dimensions,)
                                    this vector is used for out-of-vocabulary tokens.
        """

        # Get the path to the file where vectors are stored
        vectors_path = os.path.join(self.model_path, "vectors")

        # Unpickle the vectors
        with open(vectors_path, "rb") as vectors_file:
            vectors = pickle.load(vectors_file)

        # Create a default vector that is returned
        # when an out-of-vocabulary token is encountered
        default_vector = np.zeros(vectors.shape[1], dtype=vectors.dtype)

        return vectors, default_vector

    def _load_key2row(self):
        """Loads the key2row dictionary from disk.

        Returns:
            key2row (dict): a dictionary that maps a hash to a word.
        """

        # Create the path to the file where hash keys to row indices
        # mappings are stored
        key2row_path = os.path.join(self.model_path, "key2row")

        # Unpickle the file
        with open(key2row_path, "rb") as key2row_file:
            key2row = pickle.load(key2row_file)

        return key2row

    def has_vector(self, word):
        """Checks whether 'word' has a vector or not in self.data

        Args:
            word (str): the word to which we wish to test whether a vector exists or not.

        Returns:
            True if a vector for 'word' already exists in self.data.
        """

        # Create the word hash key
        key = hash_string(word)

        # if the key exists return True
        if key in self.key2row:

            return True

        else:
            return False

    def __getitem__(self, word):
        """takes a word as a string and returns the corresponding vector

        Args:
            word (str): the word to which we wish to return a vector.


        Returns:
            The vector embedding of the word.
            if no vector is found, self.default_vector is returned.
        """

        # Create the word hash key
        key = hash_string(word)

        # if the key does not exists return default vector
        if not self.has_vector(word):
            return self.default_vector

        # Get the vector row correponding to the hash
        row = self.key2row[key]

        # Get the vector
        vector = self.data[row]

        return vector
