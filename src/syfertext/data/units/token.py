class Token:
    def __init__(self, doc: "TextDoc", token_meta: "TokenMeta", position: int):

        self.doc = doc

        # The token's meta object
        self.token_meta = token_meta

        self.position = position

        # A dictionary to hold custom attributes
        self.attributes = token_meta.attributes

    def __str__(self):
        # The call to `str()` in the following is to account for the case
        # when text is of type String or StringPointer (which are Syft string types)
        return self.text

    @property
    def text(self):
        """Get the token text in str type"""
        return self.token_meta.text

    @property
    def space_after(self):
        """Whether the current token's text is followed by a white space or not
        """
        return self.token_meta.space_after

    @property
    def text_with_ws(self):
        """The text content of the token with the trailing whitespace(if any)."""

        if self.space_after:
            return self.text + " "
        else:
            return self.text

    def __len__(self):
        """The number of unicode characters in the token, i.e. `token.text`.
        The number of unicode characters in the token.
        """
        return len(self.text)

    def __repr__(self):
        return f"Token[{self.text}]"
