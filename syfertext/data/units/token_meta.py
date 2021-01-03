from .underscore import Underscore


class TokenMeta:
    """This class holds some meta data about a token from the text held by a Doc object.
    This allows to create a Token object when needed.
    """

    def __init__(self, text: str, space_after: bool):
        """Initializes a TokenMeta object

        Args:
            text (str): The token's text.
            space_after (bool): Whether the token is followed by a single white
                space (True) or not (False).
        """

        self.text = text

        self.space_after = space_after

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()
