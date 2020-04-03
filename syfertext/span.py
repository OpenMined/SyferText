import syft
import torch

hook = syft.TorchHook(torch)

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker

from typing import List, Dict, Set, Union

from .doc import Doc
from .underscore import Underscore


# TODO: Extend span as child of AbstaractObject ?
# TODO: Extend span as child of Doc as most of the functions are same ?

class Span:
    """A slice from a Doc object.
    """

    def __init__(self, doc , start : int, end : int):
        """Create a `Span` object from the slice `doc[start : end]`.

        Args:
            doc (Doc): The parent document.
            start (int): The index of the first token of the span.
            end (int): The index of the first token after the span.
        
        Returns (Span): 
            The newly constructed object.

        """

        self.doc = doc
        self.start = start
        self.end = end  # TODO: What if the end is None object ? like doc[1:]

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

        if isinstance(key,int):
            if key < 0 :
                return self.doc[self.end + key]
            else:
                return self.doc[self.start + key]

        if isinstance(key,slice):
            # TODO create a normalize slice function here to handle negative slices
            start, end = key.start, key.stop

            # shift the origin
            start += self.start
            end += self.start

            # how to handle empty ranges ?
            assert self.start <= start < self.end and start < end <= self.end, "Not a valid slice"

            return Span(self.doc, start, end)

    def __len__(self):
        """Return the number of tokens in the Span."""
        return self.end - self.start

    def __iter__(self):
        """Allows to loop over tokens in `Span.doc`"""
        for i in range(self.start,self.end):

            # Yield a Token object
            yield self[i] 

    def _normalize_slice(inp : slice):
        # TODO : Complete this function
        pass

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


def as_doc(self, owner: BaseWorker = None): # tags: List[str] = None, description: str = None):
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

        # Create a new doc object on the required location
        doc = Doc(self.doc.vocab, owner = owner)

        # Iterate over the token_meta present in span
        for token in self:
            # Add token meta object to the new doc
            doc.container.append(token)

        return doc