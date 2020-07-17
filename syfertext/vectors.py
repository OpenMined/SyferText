import pickle
import os
from pathlib import Path
import numpy as np
import importlib

from .utils import hash_string
from .typecheck.typecheck import type_hints

from typing import Dict


class Vectors:
    def __init__(self, hash2row: Dict[int, int], vectors: np.ndarray):
        """Creates the Vectors object.

        Args:
            hash2row: A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input to a an embedding layer.
            vectors: A 2D numpy array that contains the word embeddings of tokens.
        """

        self.hash2row = hash2row
        self.vectors = vectors

        # Create a default vector that is returned
        # when an out-of-vocabulary token is encountered
        # This will create the self.default_vector property
        self._create_default_vector()

    @type_hints
    def _create_default_vector(self) -> None:
        """Create a default vector that is returned
        when an out-of-vocabulary token is encountered

        Modifies:
            self.default_vectors: This property is created or modified by
        this method

        """

        # Create the default vector as a numpy array if the `vectors` property is
        # set.

        if isinstance(self.vectors, np.ndarray):
            self.default_vector = np.zeros(self.vectors.shape[1], dtype=self.vectors.dtype)

    # Check parameter
    @type_hints
    def load_data(self, hash2row: Dict[int, int], vectors: np.ndarray) -> None:
        """Loads the vector data. This is needed when the Vocab object loads its
        state, which might contain vector data.

        Args:
            hash2row: A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input to a an embedding layer.
            vectors: A 2D numpy array that contains the word embeddings of tokens.
        """

        self.hash2row = hash2row
        self.vectors = vectors

        # Create the default vector return in case of
        # out-of-vocabulary
        self._create_default_vector()

    @type_hints
    def has_vector(self, word: str) -> bool:
        """Checks whether 'word' has a vector or not in self.vectors

        Args:
            word (str): the word to which we wish to test whether a vector exists or not.

        Returns:
            True if a vector for 'word' already exists in self.vectors.
        """

        if self.vectors is None:
            return False

        # Create the word hash
        key = hash_string(word)

        # if the key exists return True
        if key in self.hash2row:
            return True

        else:
            return False

    # Find return type: Vector
    def __getitem__(self, word):
        """takes a word as a string and returns the corresponding vector

        Args:
            word (str): the word to which we wish to return a vector.


        Returns:
            The vector embedding of the word.
            if no vector is found, self.default_vector is returned.
        """

        # Make sure the Vocab has a `vector` property set.
        assert self.vectors is not None, "No vector array is loaded in 'Vocab'."

        # if the key does not exists return default vector
        if not self.has_vector(word):
            return self.default_vector

        # Create the word hash key
        key = hash_string(word)

        # Get the vector row corresponding to the hash
        row = self.hash2row[key]

        # Get the vector
        vector = self.vectors[row]

        return vector
