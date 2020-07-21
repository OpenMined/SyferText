import pickle
import os
from pathlib import Path
import importlib
import torch
from typing import Union

from .utils import hash_string


class Vectors:
    def __init__(self, model_name):

        self.model_name = model_name

        # At initialization, no vectors are loaded
        # They will be loaded once a vector is
        # requested for the first time
        self.loaded = False

    def _load_data(self):
        """Loads the vectors from the language model package named
        `self.model_name` which should be installed.
        """

        # Import the language model
        model = importlib.import_module(f"syfertext_{self.model_name}")

        # Import the dictionary of loaders:
        # This dictionary will be used to laod
        # `vectors` array and `key2row` dictionary
        LOADERS = getattr(model, "LOADERS")

        # Load the array holding the word vectors
        self.data, self.default_vector = LOADERS["vectors"]()

        # Convert the default vector to torch Tensor
        self.default_vector = torch.Tensor(self.default_vector)

        # Load the mappings between word hashes and row indices in 'self.data'
        self.key2row = LOADERS["key2row"]()

        # Set the `loaded` property to True since data is now loaded
        self.loaded = True

    def has_vector(self, key: Union[str, int]) -> bool:
        """Checks whether 'word' has a vector or not in self.data

        Args:
            key: the word or its hash to which we wish to test whether a vector exists or not.

        Returns:
            True if a vector for 'word' already exists.
        """

        # If data is not yet loaded, then load it
        if not self.loaded:
            self._load_data()

        if isinstance(key, str):
            # Create the word hash key
            orth = hash_string(key)

        else:
            orth = key

        # if the key exists return True
        if orth in self.key2row:
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

        # If data is not yet loaded, then load it
        if not self.loaded:
            self._load_data()

        # Create the word hash key
        key = hash_string(word)

        # if the key does not exists return default vector
        if not self.has_vector(word):
            return self.default_vector

        # Get the vector row corresponding to the hash
        row = self.key2row[key]

        # Get the vector
        vector = self.data[row]

        # Convert the vectors to torch Tensors
        vector = torch.tensor(vector, dtype=torch.float32)

        return vector
