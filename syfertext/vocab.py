import pickle
import os
from pathlib import Path

from .vectors import Vectors
from .string_store import StringStore
from .lexeme import Lexeme
from .lexeme import LexemeMeta


class Vocab:
    def __init__(self, model_name: str, lex_attr_getters=None):
        # TODO: this method of using 'model_name' to load vectors, strings and key2row is temporary, made specifically for the DevFest POC.
        #       it should be changed to something similar to what spaCy does for ease.

        self.model_name = model_name

        # Create the path to where the folder named 'model_name' is stored.
        dirname = str(Path.home())

        self.model_path = os.path.join(dirname, "SyferText", model_name)

        # Create the 'strings' list that holds words that the Vocab object knows and
        # have vectors for
        strings = self.load_strings()

        # Create a `StringStore` object which acts like a lookup table
        # mapping between all strings known to the vocabulary and
        # their hashes. It can be used to retrieve a string given its hash
        # key, or vice versa.
        self.store = StringStore(strings=strings)

        #Lookup table of Lexeme objects, the key is equal to orth value of lex(hash of string)
        self.lex_store = {}
        # Create the Vectors object
        self.vectors = Vectors(model_name)

    def load_strings(self):
        """load the pickled list of words that the Vocab object knows and has vectors for"""

        words_path = os.path.join(self.model_path, "words")

        with open(words_path, "rb") as word_file:
            strings = pickle.load(word_file)

        return strings

    def __iter__(self):
        """Iterate over the lexemes in the vocabulary.
        YIELDS (Lexeme): An entry in the vocabulary.
        """
        for key, addr in self.lex_store.items():
            lex = Lexeme(self, key)
            yield lex

    def __getitem__(self, id_or_string):
        """Retrieve a string, given an int ID or a unicode string. If a
        previously unseen unicode string is given, a new lexeme is created and
        stored.
        id_or_string (int or unicode): The integer ID of a word, or its unicode
        string. 
        RETURNS (Lexeme): The lexeme indicated by the given ID.
        """
        
        if isinstance(id_or_string, str):
            orth = self.store.add(id_or_string)
        else:
            orth = id_or_string 
        return Lexeme(self, orth)

     def __contains__(self, key):
        """Check whether the string or int key has an entry in the vocabulary.
        string (unicode): The ID string.
        RETURNS (bool) Whether the string has an entry in the vocabulary.
        DOCS: https://spacy.io/api/vocab#contains
        """
       
        
        if isinstance(key, str):
            orth = self.strings[key]
        else:
            orth = key
        lex = self.lex_store.get(orth)
        return lex is not NULL


    def get_by_orth(self, orth):
        """Get a LexemeMeta from the lexstore, creating a new
        Lexeme if necessary.
        """
        if orth == 0:
            return None
        lex = self.lex_store.get(orth)
        if lex != NULL:
            return lex
        else:
            return self._create_lex(self.store[orth])

    def _create_lex(self, string) :
        lex = LexemeMeta()
        lex.orth = self.store.add(string)
        lex.length = len(string)
        
        lex.id = self.vectors.key2row.get(lex.orth)
        
        if self.lex_attr_getters is not None:
            for attr, func in self.lex_attr_getters.items():
                value = func(string)
                if isinstance(value, str):
                    value = self.strings.add(value)
                
                if value:
                    Lexeme.set_struct_attr(lex, attr, value)
        
        self.lex_store[lex.orth] =  lex
        
        return lex

        