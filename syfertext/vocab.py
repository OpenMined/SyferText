import pickle
import os
from pathlib import Path

from .vectors import Vectors


class Vocab:
    def __init__(self, model_name: str):
        # TODO: this method of using 'model_name' to load vectors, strings and key2row is temporary, made specifically for the DevFest POC.
        #       it should be changed to something similar to what spaCy does for ease.

        self.model_name = model_name

        # Create the path to where the folder named 'model_name' is stored.
        dirname = str(Path.home())

        self.model_path = os.path.join(dirname, "SyferText", model_name)

        # Create the 'strings' list that holds all of the words that the Vocab object knows and
        # have vectors for
        self.strings = self.load_strings()

        # Create the Vectors object
        self.vectors = Vectors(model_name)

    def load_strings(self):
        """
           load the pickled list of words that the Vocab object knows and has vectors fot.

           # TODO: this should be later replaced by an object of StringStore as in spaCy
        """

        words_path = os.path.join(self.model_path, "words")

        with open(words_path, "rb") as word_file:
            strings = pickle.load(word_file)

        return strings
