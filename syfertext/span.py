from .token import Token
import syft
import torch

hook = syft.TorchHook(torch)


from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker

from typing import List
from typing import Dict
from typing import Set
from typing import Union
from .underscore import Underscore

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

        if isinstance(key,int):
            if key < 0 :
                return self.doc[self.end + key]
            else:
                return self.doc[self.start + key]

        if isinstance(key,slice):
            start, end = key.start,key.end  # TODO create a normalize slice function here to handle negative slices

            # shift the origin
            start += self.start
            end += self.start

            # how to handle empty ranges ?
            assert self.start <= start < self.end and start < end <= self.end, "Not a valid slice"

            return Span(self.doc, start, end)

    def __len__(self):
        """Return the number of tokens in the Span."""
        return end - start

    def __iter__(self):
        """Allows to loop over tokens in `Span.doc`"""
        for i in range(self.start,self.end):

            # Yield a Token object
            yield self[i] 

    def _normalize_slice(inp : slice):
        # TODO

    @property
    def vector:
        # TODO
    
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
        # TODO

    def as_doc(self):
        """Create a `Doc` object with a copy of the `Span`'s data.
        
        Returns (Doc): 
            The `Doc` copy of the span.
        """ 
        # TODO
