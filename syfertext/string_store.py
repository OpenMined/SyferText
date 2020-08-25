from __future__ import annotations

from .typecheck.typecheck import type_hints
from .utils import hash_string
from typing import Union


class StringStore:
    """ StringStore object acts as a lookup table.
        It looks up strings by 64-bit hashes and vice-versa, looks up hashes by their corresponding strings.
    """

    def __init__(self, strings=None):
        """Create the StringStore object

        Args:
            strings (list): List of Strings to add to store
        """
        # key_to_str maps hashes to strings; i.e it stores (key == hash : value == string)
        self.key_to_str = {}

        # str_to_key maps strings to hashes; i.e it stores (key == string : value == hash)
        self.str_to_key = {}

        if strings is not None:  # load strings
            for word in strings:
                self.add(string=word)

    # refers to the 'in' operation, have not added typechecking here
    def __contains__(self, string:str) -> bool:
        """Check whether string is in the store

        Args:
            string (str): string to check

        Returns:
            Boolean: True if string in store else False
        """

        return string in self.str_to_key.keys()

    @type_hints
    def add(self, string:str) -> int:
        """Add a sting to the StringStore

        Args:
            string (str): The string to add to store

        Returns:
            key (int): Hash key for corresponding string
        """

        if not isinstance(string, str):
            raise TypeError(
                f"Argument `string` is of type `{type(string)}`. Expected type is `str`"
            )

        if string in self:  # store contains string

            key = self.str_to_key[string]

        else:
            # get corresponding hash value
            key = hash_string(string=string)

            # add string to dictionaries
            self.str_to_key[string] = key
            self.key_to_str[key] = string

        return key

    @type_hints
    def __getitem__(self, string_or_id: Union[str, int]) -> Union[str, int]:
        """Retrieve a string from a given hash or vice-versa.
        If passed argument is a string which is not found in the store,
        then it is added to the store and the corresponding key is returned.

        Args:
            string_or_id (str, int): The hash/string value

        Returns:
            key or string (int, str): Hash key for argument string or string for corresponding hash key
        """

        if not (isinstance(string_or_id, str) or isinstance(string_or_id, int)):
            # TODO: Add custom SyferText error messgage
            raise TypeError(
                f"Argument `key` is of type `{type(string_or_id)}`. Expected type is `str` or `int`"
            )

        # If string_or_id is hash value return corresponding string
        if isinstance(string_or_id, int):
            return self.key_to_str[string_or_id]

        # If string_or_id is of type string then return the corresponding key (hash value).
        # And if string is not found in store, add it to the store and then return the corresponding key
        elif isinstance(string_or_id, str):

            if string_or_id not in self:
                key = self.add(string_or_id)  # add string to store
            else:
                key = self.str_to_key[string_or_id]

            return key

    @type_hints
    def __len__(self) -> int:
        """Get the number of strings in the store."""

        return len(self.str_to_key)
