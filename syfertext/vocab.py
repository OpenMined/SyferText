import pickle
import os
from pathlib import Path
from typing import Union
from typing import List
from typing import Callable
import functools
import warnings

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

    def load_strings(self) -> List[str]:
        """load the pickled list of words that the Vocab object knows and has vectors for"""

        words_path = os.path.join(self.model_path, "words")

        with open(words_path, "rb") as word_file:
            strings = pickle.load(word_file)

        return strings

    def get_vector(self, key: Union[str, int]):
        """Retrieve a vector for a word in the vocabulary. Words can be looked
        up by string or int ID. 

        Args:
            key: The word hash, or its plaintext.
        """
        if isinstance(key, int):
            key = self.store[key]

        return self.vectors[key]

    def __iter__(self):
        """Iterate over the lexemes in the vocabulary.

        Yields:
            lexeme (Lexeme): An entry in the vocabulary.
        """

        for orth in self.lex_store:

            # Create the Lexeme object for the given orth
            lexeme = Lexeme(self, orth)

            yield lexeme

    def __getitem__(self, key: Union[str, int]) -> Lexeme:
        """Retrieve a Lexeme object corresponding to a string, given its hash or its plaintext. If a
        previously unseen string is given, a new lexeme is created and
        stored.

        Args:
            key: The word hash, or its plaintext. 

        Returns:
            Lexeme: The lexeme specified by the given key.
        """

        if isinstance(key, str):
            orth = self.store.add(key)

        else:
            orth = key

        return Lexeme(self, orth)

    def __contains__(self, key: Union[str, int]) -> bool:
        """Check whether the lexeme for the string or int key has an entry in the vocabulary."""

        if isinstance(key, str):
            orth = self.store[key]

        else:
            orth = key

        return orth in self.lex_store

    def has_vector(self, key: Union[str, int]) -> bool:
        """Check whether the given string has an entry in word vectors table
        
        Args:
            key: The word or its hash for which the existence of a vector is to be checked out.
        
        Returns :
            bool: True if a vector for 'word' already exists.
        """

        return self.vectors.has_vector(key)

    def add_flag(self, flag_getter: Callable, flag_id=-1) -> int:
        """Sets a boolean flag to all the entries in the vocabulary. 
        Also adds for the future entries. This Method is inspired from Spacy.
        You'll then be able to access the flag value using token.check_flag(flag_id) or 
        you can also call on a Lexeme object also like lexemeobj.check_flag(flag_id).
        The maximum value of flag_id is capped at 68 and the flag_id below 28 are reserved 
        for pre-defined SyferText attributes.

        Args:
            flag_getter: The function which takes a string as an argument and 
                returns the boolean flag for the given string.
            flag_id: The flag_id on which the attribute value will be assigned, 
                if -1 the next available flag id will be assigned, it should be in range [28,68].
          
        Returns:
            flag_id(int): returns the flag_id through which user can access the flag value.
        """

        if flag_id == -1:
            # set the next availabel flag_id
            flag_id = len(self.lex_attr_getters) + 1

            if flag_id > 68:
                raise Exception(
                    "The maximum number of custom flags is reached, you can replace a current flag by passing its id b/w 28 and 68."
                )

        if flag_id in range(28):
            raise Exception(
                "Custom flag_id should be greater than 27 as flags for these ids are reserved"
            )

        if flag_id > 68:
            raise Exception(
                "Custom flag_id should be less than 68, maximum numbers of custom flags are capped"
            )

        # Iterate over all the current entries in vocabulary and set the flag attribute.
        for lex in self:
            lex.set_flag(flag_id, flag_getter(lex.orth_))

        # store the flag attribute getter with the flag_id as key
        self.lex_attr_getters[flag_id] = flag_getter

        return flag_id

    def get_lex_meta(self, orth: int) -> Union[LexemeMeta]:
        """Get a LexemeMeta from the lexstore, creating a new
        Lexeme if necessary.
        
        Args:
            orth: The word hash for which the LexemeMeta object is requested.

        Returns:
            lex_meta: The `LexemeMeta` object for given orth.
        """

        if orth == 0:
            return None

        # get the LexemeMeta object from lex store if exist.
        lex_meta = self.lex_store.get(orth)

        # If the LexemeMeta object already exist it returned.
        if lex_meta:
            return lex_meta

        # LexemeMeta instance doesn't exist for the given orth
        else:
            # Create the new LexemeMeta object.
            return self._create_lex_meta(self.store[orth])

    def _create_lex_meta(self, string: str) -> LexemeMeta:
        """Creates a LexemeMeta object corresponding to `string`.
        
        Args:
            string: The plaintext string for which a LexemeMeta object is to be created.
            
        Returns:
            A LexemeMeta object corresponding to `string`.
        """

        # create the new LexemeMeta object
        lex_meta = LexemeMeta()

        # Assign the lex attributes
        lex_meta.orth = self.store.add(string)
        lex_meta.length = len(string)

        # The language model name of parent vocabulary
        lex_meta.lang = self.store.add(self.model_name)

        # id is the index of the corresponding vector
        # in self.vectors if we vectors are loaded.
        if not self.vectors.loaded:
            lex_meta.id = None
        else:
            lex_meta.id = self.vectors.key2row.get(lex_meta.orth)

        # Traverse all the lexical attributes getters in the dict.
        for attr, func in self.lex_attr_getters.items():
            value = func(string)

            # check if string id out of vocabulary
            if attr == Attributes.IS_OOV:
                value = not self.has_vector(string)

            # For attributes with string values add them to string store
            # and assign the orth id of that to LexemeMeta object
            if isinstance(value, str):
                value = self.store.add(value)

            # Assign rest of the attributes to the LexemeMeta object
            if value:
                lex_meta.set_lexmeta_attr(attr_id=attr, value=value)

        # Store the LexemeMeta object in the lex store.
        self.lex_store[lex_meta.orth] = lex_meta

        return lex_meta
