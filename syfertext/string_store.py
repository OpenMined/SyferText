from .utils import hash_string

#############################
# StringStore object acts as a lookup table
# It looks up strings by hashes 
#############################

class StringStore:

    def __init__(self, strings=None):
        """Create the StringStore"""

        self.key_to_string = {}
        self.string_to_key = {}

    def add(self, string):
        """Add a sting to the StringStore

        string (str): The string to add
        """

        if isinstance(string, str):
            if string in self.key_to_string.values():
                key = self.string_to_key[string]
            else:
                # get corresponding hash value
                key = hash_string(string)

                # add string to dictionaries
                self.string_to_key[string] = key
                self.key_to_string[key] = string                
        else:
            raise TypeError(f"string is of type {type(string)}; Expected type str")
        
        return key
    
    def __getitem__(self, string_or_id):
        """Retrieve a string from a given hash,  or vice-versa

        string_or_id (str, int): The hash/string value
        """

        if isinstance(string_or_id, int):
            # if string_or_id is hash value
            # return correpoding string
            return self.key_to_string[string_or_id]
        
        elif isinstance(string_or_id, str):
            # if string_or_id is of type string
            # return correpoding hash value
        
            return self.string_to_key[string_or_id]
        else:
            raise TypeError(f"key is of type {type(string_or_id)}; Expected type str or int")