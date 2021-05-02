# Inspired from Spacy attrs structs.


class Attributes:
    """Class holding Attributes Id's for Lexical attributes."""

    (
        NULL_ATTR,  # NULL_ATTR with value 0, is not used.
        ID,
        ORTH,
        LOWER,
        NORM,
        SHAPE,
        PREFIX,
        SUFFIX,
        LENGTH,
        LANG,
        IS_ALPHA,
        IS_ASCII,
        IS_DIGIT,
        IS_LOWER,
        IS_PUNCT,
        IS_SPACE,
        IS_TITLE,
        IS_UPPER,
        LIKE_URL,
        LIKE_NUM,
        LIKE_EMAIL,
        IS_STOP,
        IS_OOV,
        IS_BRACKET,
        IS_QUOTE,
        IS_LEFT_PUNCT,
        IS_RIGHT_PUNCT,
        IS_CURRENCY,
    ) = range(28)
