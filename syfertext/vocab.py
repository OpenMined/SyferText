import os
import torch
from pathlib import Path
from typing import Union
from typing import List
from typing import Callable
from typing import Dict
import functools

from .vectors import Vectors
from .string_store import StringStore
from .utils import search_resource
from .utils import create_state_query
from .state import State
from .pointers import StatePointer
from . import LOCAL_WORKER

import syft.serde.msgpack.serde as serde
from syft.workers.base import BaseWorker

from .lexeme import Lexeme
from .lexeme import LexemeMeta
from .attrs import Attributes
from .lex_attrs import LEX_ATTRS
from .stop_words import STOP_WORDS


class Vocab:
    def __init__(
        self,
        hash2row: Dict[int, int] = None,
        vectors: torch.tensor = None,
        model_name: str = None,
        owner: BaseWorker = None,
    ):
        """Initializes the Vocab object.

        Args:
            hash2row (optional): A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input a an embedding layer.
            vectors (optional): A 2D numpy array that contains the word embeddings of tokens.
            model_name (optional): The name of the language model the owns this vocab.
            owner (optional): The worker on which the vocab object is located.
        """

        # Create the Vectors object
        self.vectors = Vectors(hash2row, vectors)

        self.model_name = model_name
        self.owner = owner

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

    def set_model_name(self, model_name: str) -> None:
        """Set the language model name to which this object belongs.

        Args:
            name: The name of the language model.
        """

        self.model_name = model_name

    def load_state(self) -> None:
        """Search for the state of this Vocab object on PyGrid.

        Modifies:
            self.vectors: The `vectors` property is initialized with the loaded
                state which included the `hash2row` mapping and optionally the
                `vectors` array.
        """

        # Create the query. This is the ID according to which the
        # State object is searched on PyGrid
        state_id = create_state_query(model_name=self.model_name, state_name="vocab")

        # Search for the state
        result = search_resource(query=state_id, local_worker=self.owner)

        # If no state is found, return
        if not result:
            return

        # If a state is found get either its pointer if it is remote
        # or the state itself if it is local
        elif isinstance(result, StatePointer):
            # Get a copy of the state using its pointer
            state = result.get_copy()

        elif isinstance(result, State):
            state = result

        # Detail the simple object contained in the state
        hash2row_simple, vectors_simple = state.simple_obj

        hash2row = serde._detail(LOCAL_WORKER, hash2row_simple)
        vectors = serde._detail(LOCAL_WORKER, vectors_simple)

        # Load the state
        self.vectors.load_data(vectors=vectors, hash2row=hash2row)

    def dump_state(self) -> State:
        """Returns a State object that holds the current state of this object.
        The state is characterized by the `hash2row` mapping and optionally the
        `vectors` array.

        Returns:
            A State object that holds a simplified version of this object's state.
        """

        # Simply the state variables
        hash2row_simple = serde._simplify(LOCAL_WORKER, self.vectors.hash2row)
        vectors_simple = serde._simplify(LOCAL_WORKER, self.vectors.vectors)

        # Create the query. This is the ID according to which the
        # State object is searched for on PyGrid1
        state_id = f"{self.model_name}:vocab"

        # Create the State object
        state = State(simple_obj=(hash2row_simple, vectors_simple), id=state_id, access={"*"})

        return state

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
        if self.vectors.vectors is None:
            lex_meta.id = None
        else:
            lex_meta.id = self.vectors.hash2row.get(lex_meta.orth)

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
