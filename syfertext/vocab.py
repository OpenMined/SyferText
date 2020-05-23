import pickle
import os
from pathlib import Path

from .vectors import Vectors
from .string_store import StringStore
from .utils import search_state
from .state import State
from . import LOCAL_WORKER

import syft.serde.msgpack.serde as serde

import numpy

from typing import Dict

class Vocab:
    def __init__(
        self, key2index: Dict[int, int] = None, vectors: numpy.array = None, model_name: str = None
    ):
        """Initializes the Vocab object.

        Args:
            key2index (optional): A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input a an embedding layer.
            vectors (optional): A 2D numpy array that contains the word embeddings of tokens.
            model_name (optional): The name of the language model the owns this vocab.
        """

        # Create the Vectors object
        self.vectors = Vectors(key2index, vectors)

        self.model_name = model_name

        # Create a `StringStore` object which acts like a lookup table
        # mapping between all strings known to the vocabulary and
        # their hashes. It can be used to retrieve a string given its hash
        # key, or vice versa.
        # Only strings that are encountered during tokenization will be stored here
        self.store = StringStore()

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
                state which included the `key2index` mapping and optionally the
                `vectors` array. 
        """

        # Create the query. This is the ID according to which the
        # State object is searched on PyGrid
        state_id = f"{self.model_name}:vocab"

        # Search for the state
        state = search_state(query=state_id)

        # If no state is found, just return
        if not state:
            return

        # Detail the simple object contained in the state
        key2index_simple, vectors_simple = state.simple_obj

        key2index = serde._detail(LOCAL_WORKER, key2index_simple)
        vectors = serde._detail(LOCAL_WORKER, vectors_simple)

        # Load the state
        self.vectors.load_data(vectors=vectors, key2index=key2index)

    def dump_state(self) -> State:
        """Returns a State object that holds the current state of this object.
        The state is characterized by the `key2index` mapping and optionally the
        `vectors` array.

        Returns:
            A State object that holds a simplified version of this object's state.
        """

        # Simply the state variables
        key2index_simple = serde._simplify(LOCAL_WORKER, self.vectors.key2index)
        vectors_simple = serde._simplify(LOCAL_WORKER, self.vectors.vectors)

        # Create the query. This is the ID according to which the
        # State object is searched for on PyGrid
        state_id = f"{self.model_name}:vocab"

        # Create the State object
        state = State(simple_obj=(key2index_simple, vectors_simple), id=state_id, access={"*"})

        return state
