import pickle
import os
from pathlib import Path
from typing import Union
import functools

from .vectors import Vectors
from .string_store import StringStore
from .lexeme import Lexeme
from .lexeme import LexemeMeta
from .attrs import Attributes
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS


class Vocab:
    def __init__(self, model_name: str):

        self.model_name = model_name

        # Create a `StringStore` object which acts like a lookup table
        # mapping between all strings known to the vocabulary and
        # their hashes. It can be used to retrieve a string given its hash
        # key, or vice versa.
        # Only strings that are encountered during tokenization will be stored here
        self.store = StringStore()

        # Lookup table of Lexeme objects, the key is equal to orth value of lex(hash of string)
        self.lex_store = {}

        # Function to get the lexical attributes stored in a dict
        # with key as correseponding attribute ID.
        self.lex_attr_getters = LEX_ATTRS

        # List of stop words
        self.stop_words = STOP_WORDS

        # get the stop word attribute getter.
        is_stop = self.lex_attr_getters[Attributes.IS_STOP]

        # Update the function for is_stop with the stop word list
        self.lex_attr_getters[Attributes.IS_STOP] = functools.partial(
            is_stop, stops=self.stop_words
        )

        # Create the Vectors object
        self.vectors = Vectors(model_name)

    def load_strings(self):
        """load the pickled list of words that the Vocab object knows and has vectors for"""

        words_path = os.path.join(self.model_path, "words")

        with open(words_path, "rb") as word_file:
            strings = pickle.load(word_file)

        return strings

    def get_vector(self, string_or_id: Union[str, int]):
        """Retrieve a vector for a word in the vocabulary. Words can be looked
        up by string or int ID. 
        """
        if isinstance(string_or_id, int):
            string_or_id = self.store[string_or_id]

        return self.vectors[string_or_id]

    def __iter__(self):
        """Iterate over the lexemes in the vocabulary.

        Yields:
            Lexeme: An entry in the vocabulary.
        """

        for orth, lex in self.lex_store.items():
            # Create the Lexeme object for the given orth
            lexeme = Lexeme(self, orth)

            yield lexeme

    def __getitem__(self, id_or_string: Union[str, int]) -> Lexeme:
        """Retrieve a string, given an int ID or a string. If a
        previously unseen string is given, a new lexeme is created and
        stored.

        Args:
            id_or_string (int or unicode): The integer ID of a word, or its string. 

        Returns:
            Lexeme: The lexeme indicated by the given ID.
        """

        if isinstance(id_or_string, str):
            orth = self.store.add(id_or_string)

        else:
            orth = id_or_string

        return Lexeme(self, orth)

    def __contains__(self, key: Union[str, int]) -> bool:
        """Check whether the lexeme for the string or int key has an entry in the vocabulary."""

        if isinstance(key, str):
            orth = self.store[key]

        else:
            orth = key

        # Get the LexemeMeta object
        # Note: if it is not present in the lex_store than new one is created
        lex = self.lex_store.get(orth)

        return lex is not None

    def has_vector(self, string: str) -> bool:
        """If for the given string there is an entry in word vectors table"""
        return self.vectors.has_vector(string)

    def get_by_orth(self, orth: int) -> LexemeMeta:
        """Get a LexemeMeta from the lexstore, creating a new
        Lexeme if necessary.
        """

        if orth == 0:
            return None

        # get the LexemeMeta object from lex store if exist.
        lex = self.lex_store.get(orth)

        # If the LexemeMeta object already exist it returned.
        if lex != None:
            return lex

        # LexemeMeta instance doesn't exist for the given orth
        else:
            # Create the new LexemeMeta object.
            return self._create_lex(self.store[orth])

    def _create_lex(self, string: str) -> LexemeMeta:

        # Initialize vectors for the provided model_name
        # If data is not yet loaded, then load it
        if not self.vectors.loaded:
            self.vectors._load_data()

        # create the new LexemeMeta object
        lex = LexemeMeta()

        # Assign the lex attributes
        lex.orth = self.store.add(string)
        lex.length = len(string)

        # The language model name of parent vocabulary
        lex.lang = self.store.add(self.model_name)

        # id is the index of row the corresponding vector
        # is stored in word vectors table
        lex.id = self.vectors.key2row.get(lex.orth)

        # Traverse all the lexical attributes getters in the dict.
        for attr, func in self.lex_attr_getters.items():
            value = func(string)

            # check if string id out of vocabulary
            if attr == Attributes.IS_OOV:
                value = not self.has_vector(string)

            # for attributes with string values add them to string store
            # and assign the orth id of that to LexemeMeta object
            if isinstance(value, str):
                value = self.store.add(value)

            # Assign rest of the attributes to the LexemeMeta object
            if value:
                Lexeme.set_lex_attr(lex, attr, value)

        # Store the LexemeMeta object in the lex store.
        self.lex_store[lex.orth] = lex

        return lex
