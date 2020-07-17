import syft
import torch

hook = syft.TorchHook(torch)

from syft.generic.abstract.object import AbstractObject
from syft.workers.base import BaseWorker
from syfertext.token import Token

from typing import List
from typing import Dict
from typing import Set
from typing import Union
from typing import Iterator

from .typecheck.typecheck import type_hints
from .underscore import Underscore
from .utils import normalize_slice


class Span(AbstractObject):
    """A slice from a Doc object.
    """

    def __init__(
        self, doc: "Doc", start: int, end: int, id: int = None, owner: BaseWorker = None,
    ):
        """Create a `Span` object from the slice `doc[start : end]`.

        Args:
            doc (Doc): The parent document.
            start (int): The index of the first token of the span.
            end (int): The index of the first token after the span.
        
        Returns (Span): 
            The newly constructed object.

        """
        super(Span, self).__init__(id=id, owner=owner)

        self.doc = doc
        self.start = start

        # We don't need to handle `None` here
        # it will be handled by normalize slice
        # Invalid ranges handled by normalize function
        self.end = end

        # This is used to keep track of the client worker that this span
        # caters to.
        # Usually, it would be the worker operating the pipeline.
        # we set this equal to `doc.client_id` as span's client will be same as doc's client
        self.client_id = doc.client_id

        # The owner of the span object will be same worker where doc resides
        self.owner = doc.owner

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()

    @type_hints
    def set_attribute(self, name: str, value: object) -> None:
        """Creates a custom attribute with the name `name` and
       value `value` in the Underscore object `self._`

        Args:
            name (str): name of the custom attribute.
            value (object): value of the custom named attribute.
        """

        # make sure there is no space in name as well prevent empty name
        assert (
            isinstance(name, str) and len(name) > 0 and (not (" " in name))
        ), "Argument `name` should be a non-empty `str` type containing no spaces"

        setattr(self._, name, value)

    # Find return type: Union[token, span]
    @type_hints
    def __getitem__(self, key: Union[int, slice]) -> Union[Token, "Span"]:
        """Returns a Token object at position `key` or returns Span using slice `key` or the 
        id of the Token object or id of the Span object at remote location.

        Args:
            key (int or slice): The index of the token within the span, or slice of
            the span to get.

        Returns:
            Token or Span or id of the Token or id of the Span
        """

        if isinstance(key, int):

            if key < 0:
                token_meta = self.doc.container[self.end + key]
            else:
                token_meta = self.doc.container[self.start + key]

            # Create a Token object with owner same as the span object
            token = Token(doc=self.doc, token_meta=token_meta, position=key, owner=self.owner)

            return token

        if isinstance(key, slice):

            # normalize to handle negative slicing
            start, end = normalize_slice(len(self), key.start, key.stop, key.step)

            # shift the origin
            start += self.start
            end += self.start

            # Assign the new span to the same owner as this object
            owner = self.owner

            # Create a new span object
            span = Span(self.doc, start, end, owner=owner)

            # If the following condition is satisfied, this means that this
            # Span is on a different worker (the Span's owner) than the one where
            # the Language object that operates the pipeline is located (the Span's client).
            # In this case we will create the new Span at the same worker as
            # this Span, and return its ID to the client worker where a SpanPointer
            # will be made out of this id.
            if span.owner.id != span.client_id:

                # Register the Span on it's owners object store
                self.owner.register_obj(obj=span)

                # Return span_id using which we can create the SpanPointer
                return span.id

            return span

    @type_hints
    def __len__(self) -> int:
        """Return the number of tokens in the Span."""
        return self.end - self.start

    @type_hints
    def __iter__(self) -> Iterator[Token]:
        """Allows to loop over tokens in `Span.doc`"""

        for i in range(len(self)):

            # Yield a Token object
            yield self[i]

    @property
    # Find return type: torch.Tensor
    def vector(self):
        """Get span vector as an average of in-vocabulary token's vectors

        Returns:
            span_vector: span vector
        """

        # Accumulate the vectors here
        vectors = None

        # Count the tokens that have vectors
        vector_count = 0

        for token in self:

            # Get the vector of the token if one exists
            if token.has_vector:

                # Increment the vector counter
                vector_count += 1

                # Cumulate token's vector by summing them
                vectors = token.vector if vectors is None else vectors + token.vector

        # If no tokens with vectors were found, just get the default vector(zeros)
        if vector_count == 0:
            span_vector = self.doc.vocab.vectors.default_vector
        else:
            # Create the final span vector
            span_vector = vectors / vector_count

        return span_vector

    # Find return type: torch.Tensor
    def get_vector(self, excluded_tokens: Dict[str, Set[object]] = None):
        """Get Span vector as an average of in-vocabulary token's vectors,
        excluding token according to the excluded_tokens dictionary.

        Args:
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes, the keys are the attributes names and they index, for efficiency, sets of values.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            span_vector: Span vector ignoring excluded tokens
        """

        # If the excluded_token dict in None then all token are included
        if excluded_tokens is None:
            return self.vector

        # Enforcing that the values of the excluded_tokens dict are sets, not lists.
        excluded_tokens = {
            attribute: set(excluded_tokens[attribute]) for attribute in excluded_tokens
        }

        vectors = None

        # Count the tokens that have vectors
        vector_count = 0

        for token in self:

            # Get the vector of the token if one exists and if token is not excluded

            include_token = True

            include_token = all(
                [
                    getattr(token._, key) not in excluded_tokens[key]
                    for key in excluded_tokens.keys()
                    if hasattr(token._, key)
                ]
            )

            if token.has_vector and include_token:
                # Increment the vector counter
                vector_count += 1

                # Cumulate token's vector by summing them
                vectors = token.vector if vectors is None else vectors + token.vector

        # If no tokens with vectors were found, just get the default vector(zeros)
        if vector_count == 0:
            span_vector = self.doc.vocab.vectors.default_vector
        else:
            # Create the final span vector
            span_vector = vectors / vector_count

        return span_vector

    # Import inside: Doc
    def as_doc(self):
        """Create a `Doc` object with a copy of the `Span`'s tokens.

        Returns :
            The new `Doc` copy (or id to `Doc` object) of the span.
        """

        # Handle circular imports
        from .doc import Doc

        # Create a new doc object on the required location
        # Assign the same owner on which this object resides
        # Client of the doc created will be same as the span's client
        doc = Doc(self.doc.vocab, owner=self.owner, client_id=self.client_id)

        # Iterate over the token_meta present in span
        for idx in range(self.start, self.end):

            # Add token meta object to the new doc
            doc.container.append(self.doc.container[idx])

        # Same reason as explained in __getitem__ above
        if doc.owner.id != doc.client_id:

            # Register the Doc on its owner's object store
            doc.owner.register_obj(obj=doc)

            # Return doc_id which can be used to create DocPointer
            return doc.id

        return doc

    @staticmethod
    # Import inside: SpanPointer
    def create_pointer(
        span,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
    ):
        """Creates a SpanPointer object that points to a Span object living in the the worker 'location'.

        Returns:
            SpanPointer: pointer object to a Span
        """

        # I put the import here in order to avoid circular imports
        from .pointers.span_pointer import SpanPointer

        if id_at_location is None:
            id_at_location = span.id

        if owner is None:
            owner = span.owner

        """ Gives unexpected keyword argument """
        span_pointer = SpanPointer(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=ptr_id,
            garbage_collect_data=garbage_collect_data,
        )

        return span_pointer
