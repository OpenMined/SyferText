class Token:
    def __init__(self, doc: "TextDoc", token_meta: "TokenMeta", position: int):

        self.doc = doc

        # The token's meta object
        self.token_meta = token_meta

        self.position = position

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = token_meta._

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

        # Before removing the attribute, check if it exist
        assert self.has_attribute(name), f"token does not have the attribute {name}"

        delattr(self._, name)

    def get_attribute(self, name: str):
        """Returns value of custom attribute with the name `name` if it is present, else raises `AttributeError`.

        Args:
            name (str): name of the custom attribute.

        Returns:
            value (obj): value of the custom attribute with name `name`.
        """

        return getattr(self._, name)

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
