import torch

import numpy as np

from typing import Union
from typing import Dict

from .utils import hash_string


class Vectors:
    def __init__(self, hash2row: Dict[int, int], vectors: Union[np.ndarray, torch.tensor]):
        """Creates the Vectors object.

        Args:
            hash2row: A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input to a an embedding layer.
            vectors: A 2D numpy array or a torch tensor that contains the
                word embeddings of tokens.
        """

        self.load_data(hash2row=hash2row, vectors=vectors)

    def _create_default_vector(self) -> None:
        """Create a default vector that is returned
        when an out-of-vocabulary token is encountered

        Modifies:
            self.default_vectors: This property is created or modified by
        this method

        """

        # Create the default vector as a numpy array if the `vectors` property is
        # set.
        if self.vectors is not None:
            self.default_vector = torch.zeros((1, self.vectors.shape[1]), dtype=torch.float)

    def load_data(self, hash2row: Dict[int, int], vectors: Union[np.ndarray, torch.tensor]):
        """Loads the vector data. This is needed when the Vocab object loads its
        state, which might contain vector data.

        Args:
            hash2row: A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input to a an embedding layer.
            vectors: A 2D numpy array or a torch tensor that contains the
                word embeddings of tokens.

        """

        self.hash2row = hash2row

        # Set the vector property
        if isinstance(vectors, np.ndarray):
            self.vectors = torch.tensor(vectors, dtype=torch.float)
        else:
            self.vectors = vectors

        # Create a default vector that is returned
        # when an out-of-vocabulary token is encountered
        # This will create the self.default_vector property
        self._create_default_vector()

    def has_vector(self, key: Union[str, int]) -> bool:
        """Checks whether 'word' has a vector or not in self.data

        Args:
            key: the word or its hash to which we wish to test whether a vector exists or not.

        Returns:
            True if a vector for 'word' already exists in self.vectors.
        """

        if self.vectors is None:
            return False

        if isinstance(key, str):
            # Create the word hash key
            orth = hash_string(key)

        else:
            orth = key

        # if the key exists return True
        if orth in self.hash2row:
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

        # Make it a 2D vector of size (1, nb coefficients)
        vector = vector.unsqueeze(0)

        return vector
