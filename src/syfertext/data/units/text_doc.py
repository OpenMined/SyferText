from .token import Token

from typing import List
from typing import Dict
from typing import Set
from typing import Union
from typing import Generator
from .span import Span
from .utils import normalize_slice


class TextDoc:
    def __init__(self):

        # This list is populated in the __call__ method of the Tokenizer object.
        # Its members are objects of the TokenMeta class
        self.token_metas = list()

        # A dictionary to hold custom attributes
        self.attributes: Dict[str, List[str]] = dict()

    def __getitem__(self, key: Union[int, slice]) -> Union[Token, Span, int]:
        """Returns a Token object at position `key` or Span object using slice.

        Args:
            key (int or slice): The index of the token within the Doc,
                or the slice of the Doc to return as a Span object.

        Returns:
            Token or Span or id of the Span object.
        """

        if isinstance(key, int):

            idx = 0
            if key < 0:
                idx = len(self) + key
            else:
                idx = key

            # Get the corresponding TokenMeta object
            token_meta = self.token_metas[idx]

            # Create a Token object
            token = Token(doc=self, token_meta=token_meta, position=key)

            return token

        if isinstance(key, slice):

            # Normalize slice to handle negative slicing
            start, end = normalize_slice(len(self), key.start, key.stop, key.step)

            # Create a new span object
            span = Span(self, start, end)

            return span

    def __len__(self):
        """Return the number of tokens in the Doc."""
        return len(self.token_metas)

    def __iter__(self):
        """Allows to loop over tokens in `self.token_metas`"""
        for i in range(len(self.token_metas)):

            # Yield a Token object
            yield self[i]

    @property
    def text(self):
        """Returns the text present in the doc with whitespaces"""
        return "".join(token.text_with_ws for token in self)
