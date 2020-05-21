import pickle
import os
from pathlib import Path
import numpy as np
import importlib

from .utils import hash_string

from typing import Dict


class Vectors:
    def __init__(self, key2index: Dict[int, int], vectors: np.array):
        """Creates the Vectors object.

        Args:
            key2index: A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input to a an embedding layer.
            vectors: A 2D numpy array that contains the word embeddings of tokens.
        """

        self.key2index = key2index
        self.vectors = vectors


    def load_data(self, key2index: Dict[int, int], vectors: np.array):
        """Loads the vector data. This is needed when the Vocab object loads its
        state, which might contain vector data.

        Args:
            key2index: A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input to a an embedding layer.
            vectors: A 2D numpy array that contains the word embeddings of tokens.
        """

        self.key2index = key2index
        self.vectors = vectors

        
    def has_vector(self, word: str):
        """Checks whether 'word' has a vector or not in self.data

        Args:
            word (str): the word to which we wish to test whether a vector exists or not.

        Returns:
            True if a vector for 'word' already exists in self.data.
        """

        if self.vectors is None:
            return False

        # Create the word hash key
        key = hash_string(word)

        # if the key exists return True
        if key in self.key2index:
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



        # if the key does not exists return default vector
        if not self.has_vector(word):
            return self.default_vector

        # Create the word hash key
        key = hash_string(word)
        
        # Get the vector row corresponding to the hash
        row = self.key2index[key]

        # Get the vector
        vector = self.data[row]

        return vector
