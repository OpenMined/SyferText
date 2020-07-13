import pickle
import os
from pathlib import Path

from .vectors import Vectors
from .string_store import StringStore
from .utils import search_resource
from .utils import create_state_query
from .state import State
from .pointers import StatePointer
from . import LOCAL_WORKER

import syft.serde.msgpack.serde as serde
from syft.workers.base import BaseWorker

import numpy as np

from typing import Dict


class Vocab:
    def __init__(
        self,
        hash2row: Dict[int, int] = None,
        vectors: np.ndarray = None,
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
