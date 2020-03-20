from .utils import hash_string

#############################
# StringStore object acts as a lookup table
# It looks up strings by hashes 
#############################

class StringStore:

    def __init__(self, strings=None):
        """Create the StringStore object

        strings (list): List of Strings to add to StringStore 
        """
        self.key_to_str = {}
        self.str_to_key = {}

        if strings is not None:  # load strings 
            for word in strings:
                key = hash_string(word)
                self.key_to_str[key] = word
                self.str_to_key[word] = key


     def __contains__(self, string):
        """Check whether a string in in the store"""

        return string in self.str_to_key.keys()


    def add(self, string):
        """Add a sting to the StringStore

        string (str): The string to add
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

        string_or_id (str, int): The hash/string value
        """

        if isinstance(string_or_id, int):
            # if string_or_id is hash value
            # return correpoding string
            return self.key_to_str[string_or_id]
        
        elif isinstance(string_or_id, str):
            # if string_or_id is of type string

            if string_or_id not in self:    # string not in store
                key = self.add(string_or_id)
            else: 
                key = self.str_to_key[string_or_id]

            return key

        else:   # Wrong type
            raise TypeError(f"key is of type {type(string_or_id)}; Expected type str or int")

   