import syft
import torch

hook = syft.TorchHook(torch)

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker

from typing import List, Dict, Set, Union

from .underscore import Underscore
from .utils import normalize_slice


# TODO: Extend span as child of AbstaractObject ?
# TODO: Extend span as child of Doc as most of the functions are same ?


class Span(AbstractObject):
    """A slice from a Doc object.
    """

    def __init__(
        self,
        doc: "Doc",
        start: int,
        end: int,
        id: int = None,
        owner: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
    ):
        """Create a `Span` object from the slice `doc[start : end]`.

        Args:
            doc (Doc): The parent document.
            start (int): The index of the first token of the span.
            end (int): The index of the first token after the span.
        
        Returns (Span): 
            The newly constructed object.

        """
        super(Span, self).__init__(id=id, owner=owner, tags=tags, description=description)

        self.doc = doc
        self.start = start

        # we dont need to handle `None` here
        # it will be handled by normalize slice
        # Invalid ranges handled by normalize function
        self.end = end

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()

    def set_attribute(self, name: str, value: object):
        """Creates a custom attribute with the name `name` and
           value `value` in the Underscore object `self._`

        Args:
            name (str): name of the custom attribute.
            value (object): value of the custom named attribute.
        """

        # make sure there is no space in name as well prevent empty name
        assert (
            isinstance(name, str) and len(name) > 0 and (not (" " in name))
        ), "Argument name should be a non-empty str type containing no spaces"

        setattr(self._, name, value)

    def __getitem__(self, key):
        """Returns a Token object at position `key`.

        Args:
            key (int or slice): The index of the token within the span, or slice of
            the span to get.

        Returns:
            Token or Span at index key
        """

        if isinstance(key, int):
            if key < 0:
                return self.doc[self.end + key]
            else:
                return self.doc[self.start + key]

        if isinstance(key, slice):

            # normalize to handle negative slicing
            start, end = normalize_slice(len(self), key.start, key.stop, key.step)

            # shift the origin
            start += self.start
            end += self.start

            return Span(self.doc, start, end)

    @staticmethod
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

        span_pointer = SpanPointer(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=ptr_id,
            garbage_collect_data=garbage_collect_data,
        )

        return span_pointer

    def __len__(self):
        """Return the number of tokens in the Span."""
        return self.end - self.start

    def __iter__(self):
        """Allows to loop over tokens in `Span.doc`"""

        for i in range(len(self)):

            # Yield a Token object
            yield self[i]

    def __repr__(self):
        """Returns the text of the span with whitespaces"""
        return "".join(token.text_with_ws for token in self)

    @property
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

    def get_vector(self, excluded_tokens: Dict[str, Set[object]] = None):
        """Get Span vector as an average of in-vocabulary token's vectors,
        excluding token according to the excluded_tokens dictionary.

        Args
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes, the keys are the attributes names and they index, for efficiency, sets of values.
                Example: {'attribute1_name' : {value1, value2},'attribute2_name': {v1, v2}, ....}

        Returns:
            span_vector: Span vector ignoring excluded tokens
        """
        # if the excluded_token dict in None all token are included
        if excluded_tokens is None:
            return self.vector

        # enforcing that the values of the excluded_tokens dict are sets, not lists.
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

    def as_doc(self, owner: BaseWorker = None):  # tags: List[str] = None, description: str = None):
        """Create a `Doc` object with a copy of the `Span`'s tokens.

            Args:
                owner (BaseWorker) :  An optional BaseWorker object to specify the worker
                                    on which the new doc object is located. By default, it is
                                    located on the same worker as the span

            Returns (Doc): 
                The new `Doc` copy of the span.
            """

        # Owner on which new doc object will be located
        owner = owner if owner else self.doc.owner

        # handle circular imports
        from .doc import Doc

        # Create a new doc object on the required location
        doc = Doc(self.doc.vocab, text=None, owner=owner)

        # Iterate over the token_meta present in span
        for idx in range(self.start, self.end):
            # Add token meta object to the new doc
            doc.container.append(self.doc.container[idx])

        return doc
