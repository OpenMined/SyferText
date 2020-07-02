from .attrs import Attributes

from typing import Union


class LexemeMeta(object):
    """This class holds some meta data about a Lexeme from the text held by a Doc object.
    This allows to create a Lexeme object when needed.
    """

    def __init__(self):
        """Initializes a LexemeMeta object
        """

        self.flags = 0
        self.lang = 0
        self.id = 0
        self.length = 0
        self.orth = 0
        self.lower = 0
        self.shape = 0
        self.prefix = 0
        self.suffix = 0

    def set_lexmeta_attr(self, attr_id: int, value: Union[int, bool]) -> None:
        """ Sets all the attributes for given attribute id for LexemeMeta object 
        according to the provided `value`.

        Args:
            attr_id: The integer id for the corresponding attribute.
            value: Value of attribute to set.
        """

        # Assign the flag attribute of `LexemeMeta` object.
        # All flags have id >9. check `Attributes` for reference ids.
        # id>9 is only because ids less than 10 are reserved for other attributes.
        if attr_id > 9:
            self.set_flag(attr_id, value)

        # Assign the rest of the `LexemeMeta` object attributes.
        # length and orth attributes are assigned in Vocab class.
        elif attr_id == Attributes.ID:
            self.id = value

        elif attr_id == Attributes.LOWER:
            self.lower = value

        elif attr_id == Attributes.SHAPE:
            self.shape = value

        elif attr_id == Attributes.PREFIX:
            self.prefix = value

        elif attr_id == Attributes.SUFFIX:
            self.suffix = value

        elif attr_id == Attributes.LANG:
            self.lang = value

    # These 2 methods for checking and setting flags for
    # boolean attributes for Lexeme class are taken from Spacy.
    def check_flag(self, flag_id: int) -> bool:
        """This method checks the value of flag corresponding to given flag_id. 
        It check if bit at index corresponding to flag_id is 1 or 0. This method 
        is inspired from Spacy.
        
        Args:
            flag_id(int): The flag_id for corresponding attribute to check.

        Returns:
            bool: Returns True if the value of `lex_meta.flags` bit at index flag_id is 1 else 0.

        """

        one = 1

        # Check if bit at index corresponding to flag_id is 1 or 0
        # if one return True otherwise if 0  return False
        if self.flags & (one << flag_id):
            return True

        else:
            return False

    def set_flag(self, flag_id: int, value: bool) -> None:
        """Set the sets the value of flag to 1 or zero according to provided attribute value.
        It's inspired from Spacy.

        Args:
            flag_id(int): The flag_id for corresponding attribute to set.
        """

        one = 1

        # lex flag is an integer whose bits are manipulated
        # using bitwise OR with 2^(flag_id){flags = flags|1<<flag_id}
        # if corresponding bool `value` is True. Otherwise flags bit are
        # manipulated using bitwise AND with the complement of 2^flag_id
        # { flags = flags& ~1<<flag_id}
        # The result of above operation is that the bit at the index
        # corresponding to flag_id is changed to 1 if `value` is True
        # or else it's changed to 0. (by default all bits of flag are 0 as
        # it's initialzed with flags = 0)
        if value:
            self.flags |= one << flag_id

        else:
            self.flags &= ~(one << flag_id)


class Lexeme:
    """Inspired by Spacy's Lexeme class. It is an entry in the vocabulary.
    It holds various non-contextual attributes related to the corresponding string.  
    """

    def __init__(self, vocab: "Vocab", orth: int) -> None:
        """Initializes a Lexeme object.

        Args:
            vocab (Vocab): The parent vocabulary.
            orth (int): The orth id of the lexeme, i.e, the token's hash.
        """

        self.vocab = vocab
        self.orth = orth

        # Get the LexMeta stored in Vocab's lex_store
        # Note: This creates no entry in lex_store if the LexemeMeta is not already present
        self.lex_meta = vocab.get_lex_meta(orth)

    def check_flag(self, flag_id: int) -> bool:
        """Checks the value of a boolean flag. This method is inspired from Spacy.
        Args:
            flag_id(int): The attribute ID of the flag to check.

        Returns:
            bool: Returns True if the value of flag corresponding to flag_id is 1 else False.
        """

        # Check the flag value for given flag_id
        # the flags are contained in LexemeMeta object.
        return self.lex_meta.check_flag(flag_id)

    def set_flag(self, flag_id: int, value: bool) -> None:
        """Set the sets the value of flag corresponding to flag_id to 1 or zero 
        according to provided attribute value. This method is inspired from Spacy.

        Args:
            flag_id(int): The flag_id for corresponding attribute to set.
            value(bool): boolean value used to set flag.
        """

        # Sets the value of flag which is inside lexememeta object
        self.lex_meta.set_flag(flag_id=flag_id, value=value)

    @property
    def has_vector(self):
        """Whether the word has a vector in vocabulary or not."""
        return self.vocab.has_vector(self.orth_)

    @property
    def vector_norm(self) -> float:
        """The L2 norm of the vector."""

        return torch.sqrt((self.vector ** 2).sum())

    @property
    def vector(self):
        """ Returns the  vector of a given word in the vocabulary."""

        return self.vocab.get_vector(self.lex_meta.orth)

    @property
    def rank(self):
        """ The key to index in the vectors array."""

        return self.lex_meta.id

    @property
    def orth_(self):
        """The original text of the lexeme (identical to `Lexeme.text`). 
        This method is defined for consistency with the other attributes.
        """
        return self.vocab.store[self.lex_meta.orth]

    @property
    def text(self):
        """ The original text of the lexeme."""

        return self.orth_

    @property
    def lower(self):
        """Orth id of lowercase form of the lexeme."""

        return self.lex_meta.lower

    @property
    def flags(self):
        """Returns the flags Integer value.
        One can get the value assigned to a specific flag_id by 
        looking at the bit corresponding to flag_id index of flags.
        """

        return self.lex_meta.flags

    @property
    def shape(self):
        """Orth id of the transform of the word's text, to show orthographic features."""
        return self.lex_meta.shape

    @property
    def prefix(self):
        """Orth id of length-1 substring from the start of the word. """

        return self.lex_meta.prefix

    @property
    def suffix(self):
        """Orth id of length-3 substring from the end of the word."""

        return self.lex_meta.suffix

    @property
    def lang(self):
        """Orth id of language of the parent vocabulary."""

        return self.lex_meta.lang

    @property
    def lower_(self):
        """Lowercase form of the word."""

        return self.vocab.store[self.lower]

    @property
    def shape_(self):
        """Transform of the word's string, to show orthographic features."""

        return self.vocab.store[self.lex_meta.shape]

    @property
    def prefix_(self):
        """Length-1 substring from the start of the word."""

        return self.vocab.store[self.lex_meta.prefix]

    @property
    def suffix_(self):
        """Length-3 substring from the end of the word."""

        return self.vocab.store[self.lex_meta.suffix]

    @property
    def lang_(self):
        """Language of the parent vocabulary."""

        return self.vocab.store[self.lex_meta.lang]

    @property
    def is_oov(self):
        """Whether the lexeme is out-of-vocabulary."""

        return self.lex_meta.check_flag(Attributes.IS_OOV)

    @property
    def is_stop(self):
        """Whether the lexeme is a stop word.
        """
        return self.lex_meta.check_flag(Attributes.IS_STOP)

    @property
    def is_alpha(self):
        """Whether the lexeme consists of alphabetical characters only."""

        return self.lex_meta.check_flag(Attributes.IS_ALPHA)

    @property
    def is_ascii(self):
        """Whether the lexeme consists of ASCII characters."""

        return self.lex_meta.check_flag(Attributes.IS_ASCII)

    @property
    def is_digit(self) -> bool:
        """Whether the lexeme consists of digits."""

        return self.lex_meta.check_flag(Attributes.IS_DIGIT)

    @property
    def is_lower(self) -> bool:
        """Whether the lexeme is in lowercase."""

        return self.lex_meta.check_flag(Attributes.IS_LOWER)

    @property
    def is_upper(self) -> bool:
        """Whether the lexeme is in uppercase."""

        return self.lex_meta.check_flag(Attributes.IS_UPPER)

    @property
    def is_title(self) -> bool:
        """Whether the lexeme is in titlecase."""

        return self.lex_meta.check_flag(Attributes.IS_TITLE)

    @property
    def is_punct(self) -> bool:
        """Whether the lexeme is punctuation."""

        return self.lex_meta.check_flag(Attributes.IS_PUNCT)

    @property
    def is_space(self) -> bool:
        """Whether the lexeme consists of only whitespace characters."""

        return self.lex_meta.check_flag(Attributes.IS_SPACE)

    @property
    def is_bracket(self) -> bool:
        """Whether the lexeme is a bracket."""

        return self.lex_meta.check_flag(Attributes.IS_BRACKET)

    @property
    def is_quote(self) -> bool:
        """Whether the lexeme is a quotation mark."""

        return self.lex_meta.check_flag(Attributes.IS_QUOTE)

    @property
    def is_left_punct(self) -> bool:
        """Whether the lexeme is a left punctuation mark."""

        return self.lex_meta.check_flag(Attributes.IS_LEFT_PUNCT)

    @property
    def is_right_punct(self) -> bool:
        """Whether the lexeme is a right punctuation mark."""

        return self.lex_meta.check_flag(Attributes.IS_RIGHT_PUNCT)

    @property
    def is_currency(self) -> bool:
        """Whether the lexeme is a currency symbol."""

        return self.lex_meta.check_flag(Attributes.IS_CURRENCY)

    @property
    def like_url(self) -> bool:
        """Whether the lexeme resembles a URL."""

        return self.lex_meta.check_flag(Attributes.LIKE_URL)

    @property
    def like_num(self) -> bool:
        """Whether the lexeme resembles a number, e.g. "10.9", "10", etc.
        """

        return self.lex_meta.check_flag(Attributes.LIKE_NUM)

    @property
    def like_email(self) -> bool:
        """Whether the lexeme resembles an email address."""

        return self.lex_meta.check_flag(Attributes.LIKE_EMAIL)
