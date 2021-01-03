from .token import Token

from typing import List
from typing import Dict
from typing import Set
from typing import Union
from typing import Generator
from .underscore import Underscore
from .span import Span
from .utils import normalize_slice


class TextDoc:
    def __init__(self):

        # This list is populated in the __call__ method of the Tokenizer object.
        # Its members are objects of the TokenMeta class
        self.container = list()

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

        # make sure that the name is not empty and does not contain any spaces
        assert (
            isinstance(name, str) and len(name) > 0 and (" " not in name)
        ), "Argument name should be a non-empty str type containing no spaces"

        setattr(self._, name, value)

    def has_attribute(self, name: str) -> bool:
        """Returns `True` if the Underscore object `self._` has an attribute `name`. otherwise returns `False`
        Args:
            name (str): name of the custom attribute.
        Returns:
            attr_exists (bool): `True` if `self._.name` exists, otherwise `False`
        """

        # `True` if `self._` has attribute `name`, `False` otherwise
        attr_exists = hasattr(self._, name)

        return attr_exists

    def remove_attribute(self, name: str):
        """Removes the attribute `name` from the Underscore object `self._`
        Args:
            name (str): name of the custom attribute.
        """

        # Before removing the attribute, check if it exists
        assert self.has_attribute(name), f"Document does not have the attribute {name}"

        delattr(self._, name)

    def get_attribute(self, name: str):
        """Returns value of custom attribute with the name `name` if it is present, else raises `AttributeError`.

        Args:
            name (str): name of the custom attribute.
        Returns:
            value (obj): value of the custom attribute with name `name`.
        """

        return getattr(self._, name)

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
            token_meta = self.container[idx]

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
        return len(self.container)

    def __iter__(self):
        """Allows to loop over tokens in `self.container`"""
        for i in range(len(self.container)):

            # Yield a Token object
            yield self[i]

    @property
    def text(self):
        """Returns the text present in the doc with whitespaces"""
        return "".join(token.text_with_ws for token in self)

    def _get_valid_tokens(
        self, excluded_tokens: Dict[str, Set[object]] = None
    ) -> Generator[Token, None, None]:
        """Handy function to handle the logic of excluding tokens while performing operations on Doc.

        Args:
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}
        Yields:
            A generator with valid tokens, i.e. tokens which are `not` to be excluded.
        """

        if excluded_tokens:
            # Enforcing that the values of the excluded_tokens dict are sets, not lists.
            excluded_tokens = {
                attribute: set(excluded_tokens[attribute]) for attribute in excluded_tokens
            }

            # Iterate over all tokens in doc
            for token in self:

                # Check if token can be included by comparing its attribute values
                # to those in excluded_tokens dictionary.
                include_token = all(
                    [
                        token.get_attribute(key) not in excluded_tokens[key]
                        for key in excluded_tokens.keys()
                        if token.has_attribute(key)
                    ]
                )

                if include_token:
                    yield token
        else:
            # All tokens are included
            for token in self:
                yield token
