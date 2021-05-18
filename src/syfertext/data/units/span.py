# stdlib
from typing import List
from typing import Dict
from typing import Set
from typing import Union
from typing import Generator

# SyferText relative
from .token import Token
from .utils import normalize_slice


class Span:
    """A slice from a Doc object."""

    def __init__(self, doc: "TextDoc", start: int, end: int):
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

        # We don't need to handle `None` here
        # it will be handled by normalize slice
        # Invalid ranges handled by normalize function
        self.end = end

        # A dictionary to hold custom attributes
        self.attributes = token_meta.attributes

    def __getitem__(self, key: Union[int, slice]):
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
                token_meta = self.doc.token_metas[self.end + key]
            else:
                token_meta = self.doc.token_metas[self.start + key]

            # Create a Token object
            token = Token(doc=self.doc, token_meta=token_meta, position=key)

            return token

        if isinstance(key, slice):

            # normalize to handle negative slicing
            start, end = normalize_slice(len(self), key.start, key.stop, key.step)

            # shift the origin
            start += self.start
            end += self.start

            # Create a new span object
            span = Span(self.doc, start, end)

            return span

    def __len__(self):
        """Return the number of tokens in the Span."""
        return self.end - self.start

    def __iter__(self):
        """Allows to loop over tokens in `Span.doc`"""

        for i in range(len(self)):

            # Yield a Token object
            yield self[i]

    def as_doc(self):
        """Create a `Doc` object with a copy of the `Span`'s tokens.

        Returns :
            A Doc object representing this span.
        """

        # Create a new doc object
        doc = self.doc.__class__()

        # Iterate over the token_meta present in span
        for idx in range(self.start, self.end):

            # Add token meta object to the new doc
            doc.token_metas.append(self.doc.token_metas[idx])

        return doc
