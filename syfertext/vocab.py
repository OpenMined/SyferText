import pickle
import os
from pathlib import Path

from .vectors import Vectors
from .string_store import StringStore


class Vocab:
    def __init__(self, model_name: str):

        self.model_name = model_name

        # Create a `StringStore` object which acts like a lookup table
        # mapping between all strings known to the vocabulary and
        # their hashes. It can be used to retrieve a string given its hash
        # key, or vice versa.
        # Only strings that are encountered during tokenization will be stored here
        self.store = StringStore()

        # Create the Vectors object
        self.vectors = Vectors(self.model_name)
