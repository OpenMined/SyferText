import pickle
import os
from pathlib import Path
import numpy as np
import importlib

from .utils import hash_string


class Vectors:
    
    def __init__(self, model_name):

        # Import the language model
        model = importlib.import_module(model_name)

        # Import the dictionary of loaders:
        # `vectors` or `key2row`
        LOADER = getattr(model, 'LOADER')
    
        # Load the array holding the word vectors
        self.data, self.default_vector = LOADER['vectors']()

        # Load the mappings between word hashes and row indices in 'self.data'
        self.key2row = LOADER['key2row']()

        
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

        # Get the vector row corresponding to the hash
        row = self.key2row[key]

        # Get the vector
        vector = self.data[row]

        return vector
