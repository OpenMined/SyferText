from .utils import hash_string


class StringStore:
    """ StringStore object acts as a lookup table.
        It looks up strings by 64-bit hashes
    """

    def __init__(self, strings=None):
        """Create the StringStore object

        Args:
            strings (list): List of Strings to add to store
        """
        self.key_to_str = {}
        self.str_to_key = {}

        if strings is not None:  # load strings
            for word in strings:
                self.add(word)

    def __contains__(self, string):
        """Check whether string is in the store

        Args:
            string (str): string to check

        Returns:
            Boolean: True if string in store else False
        """

        return string in self.str_to_key.keys()

    def add(self, string):
        """Add a sting to the StringStore

        Args:
            string (str): The string to add

        Returns:
            key (int): hash key for corresponding string
        """

        if isinstance(string, str):

            if string in self:  # store contains string

                key = self.str_to_key[string]

            else:
                # get corresponding hash value
                key = hash_string(string)

                # add string to dictionaries
                self.str_to_key[string] = key
                self.key_to_str[key] = string

        else:
            raise TypeError(f"string is of type {type(string)}; Expected type str")

        return key

    def __getitem__(self, string_or_id):
        """Retrieve a string from a given hash,  or vice-versa

        Args:
            string_or_id (str, int): The hash/string value
        """

        if isinstance(string_or_id, int):
            # if string_or_id is hash value
            # return corresponding string
            return self.key_to_str[string_or_id]

        elif isinstance(string_or_id, str):
            # if string_or_id is of type string
            # return corresponding key (hash value)

            if string_or_id not in self:  # string not in store
                key = self.add(string_or_id)  # add string to store
            else:
                key = self.str_to_key[string_or_id]

            return key

        else:  # [TODO] Add custom SyferText error messgage
            raise TypeError(f"key is of type {type(string_or_id)}; Expected type str or int")
