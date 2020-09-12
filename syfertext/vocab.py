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
from typing import Set


class Vocab(object):
    def __init__(self, hash2row: Dict[int, int] = None, vectors: np.ndarray = None):
        """Initializes the Vocab object.

        Args:
            hash2row (optional): A dictionary that maps each token hash to an index that
                points to the embedding vector of that token in `vectors`.
                This index can also be used as an input a an embedding layer.
            vectors (optional): A 2D numpy array that contains the word embeddings of tokens.
        """

        super(Vocab, self).__init__()

        # Create the Vectors object
        self.vectors = Vectors(hash2row, vectors)

        # Create a `StringStore` object which acts like a lookup table
        # mapping between all strings known to the vocabulary and
        # their hashes. It can be used to retrieve a string given its hash
        # key, or vice versa.
        # Only strings that are encountered during tokenization will be stored here
        self.store = StringStore()

    @property
    def pipeline_name(self) -> str:
        """A getter for the `_pipeline_name` property.

        Returns:
           The lower cased `_pipeline_name` property.
        """

        return self._pipeline_name.lower()

    @pipeline_name.setter
    def pipeline_name(self, name: str) -> None:
        """Set the pipeline name to which this object belongs.

        Args:
            name: The name of the pipeline.
        """

        # Convert the name of lower case
        if isinstance(name, str):
            name = name.lower()

        self._pipeline_name = name

    @property
    def name(self) -> str:
        """A getter for the `_name` property.

        Returns:
           The lower cased `_name` property.
        """

        return self._name.lower()

    @name.setter
    def name(self, name: str) -> None:
        """Set the component name.

        Args:
            name: The name of the component
        """

        # Convert the name of lower case
        if isinstance(name, str):
            name = name.lower()

        self._name = name

    @property
    def access(self) -> Set[str]:
        """Get the access rules for this component.

        Returns:
            The set of worker ids where this component's state
            could be sent.
            If the string '*' is included in the set,  then all workers are
            allowed to receive a copy of the state. If set to None, then
            only the worker where this component is saved will be allowed
            to get a copy of the state.
        """

        return self._access_rules

    @access.setter
    def access(self, rules: Set[str]) -> None:
        """Set the access rules of this object.

        Args:
            rules: The set of worker ids where this component's state
                could be sent.
                If the string '*' is included in the set,  then all workers are
                allowed to receive a copy of the state. If set to None, then
                only the worker where this component is saved will be allowed
                to get a copy of the state.
        """

        self._access_rules = rules

    def load_state(self) -> None:
        """Search for the state of this Vocab object on PyGrid.

        Modifies:
            self.vectors: The `vectors` property is initialized with the loaded
                state which included the `hash2row` mapping and optionally the
                `vectors` array.
        """

        # Create the query. This is the ID according to which the
        # State object is searched on PyGrid
        state_id = create_state_query(pipeline_name=self.pipeline_name, state_name=self.name)

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

        hash2row = serde._detail(self.owner, hash2row_simple)
        vectors = serde._detail(self.owner, vectors_simple)

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
        state_id = f"{self.pipeline_name}:vocab"

        # Create the State object
        state = State(simple_obj=(hash2row_simple, vectors_simple), id=state_id, access=self.access)

        return state
