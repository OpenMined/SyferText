from .utils import hash_string

class Token:

    def __init__(self,
                 doc,
                 start_pos: int,
                 stop_pos: int,
                 is_space: bool,
                 space_after: bool
    ):

        self.doc = doc
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.is_space = is_space
        self.space_after = space_after


    def __str__(self):

        # The call to `str()` in the following is to account for the case
        # when text is of type String or StringPointer (which are Syft string
        # types)
        return str(self.doc.text[self.start_pos: self.stop_pos])


    @property
    def orth(self):
        """
           Get the corresponding hash value of this token
        """

        return hash_string(str(self))

    @property
    def vector(self):
        """
           Get the token vector
        """

        return self.doc.vocab.vectors[self.__str__()]
