from .token import Token

class Doc(object):

    def __init__(self,
                 vocab,
                 text
    ):
        self.vocab = vocab
        self.text = text
        
        # This list is populated in the __call__ method of the Tokenizer object.
        # Its members are objects of the TokenMeta class defined in the tokenizer.py
        # file
        self.container = list()


    def __getitem__(self,
                    key: int
    ):
        """
           Returns a Token object at position `key`.

           Parameters
           ----------
           key: int
                the index of the token to return. 
                Example: 0 -> first token
                         1 -> second token
                         :
                         :
        """

        # Get the corresponding TokenMeta object
        token_meta =  self.container[key]

        # The start and stop positions of the token in self.text
        # notice that stop_position refers to one position after `token_meta.end_pos`.
        # this is practical for indexing
        start_pos = token_meta.start_pos
        stop_pos = token_meta.end_pos + 1 if token_meta.end_pos is not None else None
        
        # Create a Token object
        token = Token(doc = self,
                      #string = self.text[start_pos:end_pos],
                      start_pos = start_pos,
                      stop_pos = stop_pos,
                      is_space = token_meta.is_space,
                      space_after = token_meta.space_after)

        return token
                      
