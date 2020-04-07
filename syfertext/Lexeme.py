from attrs import IS_ALPHA, IS_ASCII, IS_DIGIT, IS_LOWER, IS_PUNCT, IS_SPACE
from attrs import IS_TITLE, IS_UPPER, LIKE_URL, LIKE_NUM, LIKE_EMAIL, IS_STOP
from attrs import IS_BRACKET, IS_QUOTE, IS_LEFT_PUNCT, IS_RIGHT_PUNCT
from attrs import IS_CURRENCY, IS_OOV, PROB


class LexemeMeta(object):
    """This class holds some meta data about a Lexeme from the text held by a Doc object.
       This allows to create aLexeme object when needed.
    """
    def __init__(self):
        """Initializes a LexemeMeta object
        """
        self.flags
        self.lang
        self.id
        self.length
        self.orth
        self.lower
        self.norm
        self.shape
        self.prefix
        self.suffix



class Lexeme:
    """An entry in the vocabulary. A `Lexeme` has no string context – it's a
    word-type, as opposed to a word token.  
    """
    def __init__(self, Vocab vocab, orth):
        """Create a Lexeme object.
        vocab (Vocab): The parent vocabulary
        orth (uint64): The orth id of the lexeme.
        Returns (Lexeme): The newly constructd object.
        """
        self.vocab = vocab
        self.orth = orth
        #To do implement get_by_orth in Vocab class
        self.c = vocab.get_by_orth(orth)
        
    @staticmethod
    def set_lex_attr(LexemeMeta lex, name, value):
        if name < 65:
            Lexeme.set_flag(lex, name, value)
        elif name == ID:
            lex.id = value
        elif name == LOWER:
            lex.lower = value
        elif name == SHAPE:
            lex.shape = value
        elif name == PREFIX:
            lex.prefix = value
        elif name == SUFFIX:
            lex.suffix = value
        elif name == LANG:
            lex.lang = value

    @staticmethod
    def check_flag(LexemeMeta lex, flag_id) :
        one = 1
        if lex.flags & (one << flag_id):
            return True
        else:
            return False

    @staticmethod
    def set_flag(LexemeMeta lex, flag_id, value) :
        one = 1
        if value:
            lex.flags |= one << flag_id
        else:
            lex.flags &= ~(one << flag_id)
    

    @property
    def has_vector(self):
        """RETURNS (bool): Whether a word vector is associated with the object.
        """
        return self.vocab.has_vector(self.c.orth)

    @property
    def vector_norm(self):
        """RETURNS (float): The L2 norm of the vector representation."""
        vector = self.vector
        return numpy.sqrt((vector**2).sum())


    @property
    def vector(self):
        #To do add custom error message, and functionality of setting a vector.
        return self.vocab.get_vector(self.c.orth)

    @property 
    def rank(self):
    """RETURNS (unicode): Sequential ID of the lexemes's lexical type, used
        to index into tables, e.g. for word vectors."""
        return self.c.id
    
    @property
    def orth_(self):
        """RETURNS (unicode): The original verbatim text of the lexeme
            (identical to `Lexeme.text`). Exists mostly for consistency with
            the other attributes."""
        return self.vocab.strings[self.c.orth]

    @property
    def text(self):
        """RETURNS (unicode): The original verbatim text of the lexeme."""
        return self.orth_

    @property 
    def lower(self):
        """RETURNS (unicode): Lowercase form of the lexeme."""
        return self.c.lower

        
    @property 
    def norm(self):
        """RETURNS (uint64): The lexemes's norm, i.e. a normalised form of the
            lexeme text.
        """
        return self.c.norm

    @property 
    def shape(self):
        """RETURNS (uint64): Transform of the word's string, to show
            orthographic features.
        """
        return self.c.shape

    @property 
    def prefix(self):
        """RETURNS (uint64): Length-N substring from the start of the word.
            Defaults to `N=1`.
        """
        return self.c.prefix

    @property 
    def suffix(self):
        """RETURNS (uint64): Length-N substring from the end of the word.
            Defaults to `N=3`.
        """
        return self.c.suffix

    @property 
    def lang:
        """RETURNS (uint64): Language of the parent vocabulary."""
        return self.c.lang

    @property 
    def lower_(self):
        """RETURNS (unicode): Lowercase form of the word."""
        return self.vocab.strings[self.lower]

    @property 
    def norm_(self):
        """RETURNS (unicode): The lexemes's norm, i.e. a normalised form of the
            lexeme text.
        """
        return self.vocab.strings[self.c.norm]

    @property 
    def shape_(self):
        """RETURNS (unicode): Transform of the word's string, to show
            orthographic features.
        """
        return self.vocab.strings[self.c.shape]

    @property 
    def prefix_(self):
        """RETURNS (unicode): Length-N substring from the start of the word.
            Defaults to `N=1`.
        """
        return self.vocab.strings[self.c.prefix]

    @property 
    def suffix_(self):
        """RETURNS (unicode): Length-N substring from the end of the word.
            Defaults to `N=3`.
        """
        return self.vocab.strings[self.c.suffix]
        
    @property 
    def lang_(self):
        """RETURNS (unicode): Language of the parent vocabulary."""
        return self.vocab.strings[self.c.lang]


    @property
    def is_oov(self):
        """RETURNS (bool): Whether the token is out-of-vocabulary."""
        return Lexeme.c_check_flag(self.c, IS_OOV)

    @property
    def is_stop(self):
        """RETURNS (bool): Whether the token is a stop word, i.e. part of a
            "stop list" defined by the language data.
        """
        return Lexeme.c_check_flag(self.c, IS_STOP)

    @property
    def is_alpha(self):
        """RETURNS (bool): Whether the token consists of alpha characters.
            Equivalent to `token.text.isalpha()`.
        """
        return Lexeme.c_check_flag(self.c, IS_ALPHA)

    @property
    def is_ascii(self):
        """RETURNS (bool): Whether the token consists of ASCII characters.
            Equivalent to `[any(ord(c) >= 128 for c in token.text)]`.
        """
        return Lexeme.c_check_flag(self.c, IS_ASCII)

    @property
    def is_digit(self):
        """RETURNS (bool): Whether the token consists of digits. Equivalent to
            `token.text.isdigit()`.
        """
        return Lexeme.c_check_flag(self.c, IS_DIGIT)

    @property
    def is_lower(self):
        """RETURNS (bool): Whether the token is in lowercase. Equivalent to
            `token.text.islower()`.
        """
        return Lexeme.c_check_flag(self.x, IS_LOWER)

    @property
    def is_upper(self):
        """RETURNS (bool): Whether the token is in uppercase. Equivalent to
            `token.text.isupper()`
        """
        return Lexeme.c_check_flag(self.c, IS_UPPER)

    @property
    def is_title(self):
        """RETURNS (bool): Whether the token is in titlecase. Equivalent to
            `token.text.istitle()`.
        """
        return Lexeme.c_check_flag(self.c, IS_TITLE)

    @property
    def is_punct(self):
        """RETURNS (bool): Whether the token is punctuation."""
        return Lexeme.c_check_flag(self.c, IS_PUNCT)

    @property
    def is_space(self):
        """RETURNS (bool): Whether the token consists of whitespace characters.
            Equivalent to `token.text.isspace()`.
        """
        return Lexeme.c_check_flag(self.c, IS_SPACE)

    @property
    def is_bracket(self):
        """RETURNS (bool): Whether the token is a bracket."""
        return Lexeme.c_check_flag(self.c, IS_BRACKET)

    @property
    def is_quote(self):
        """RETURNS (bool): Whether the token is a quotation mark."""
        return Lexeme.c_check_flag(self.c, IS_QUOTE)

    @property
    def is_left_punct(self):
        """RETURNS (bool): Whether the token is a left punctuation mark."""
        return Lexeme.c_check_flag(self.c, IS_LEFT_PUNCT)

    @property
    def is_right_punct(self):
        """RETURNS (bool): Whether the token is a right punctuation mark."""
        return Lexeme.c_check_flag(self.c, IS_RIGHT_PUNCT)

    @property
    def is_currency(self):
        """RETURNS (bool): Whether the token is a currency symbol."""
        return Lexeme.c_check_flag(self.c, IS_CURRENCY)

    @property
    def like_url(self):
        """RETURNS (bool): Whether the token resembles a URL."""
        return Lexeme.c_check_flag(self.c, LIKE_URL)

    @property
    def like_num(self):
        """RETURNS (bool): Whether the token resembles a number, e.g. "10.9",
            "10", "ten", etc.
        """
        return Lexeme.c_check_flag(self.c, LIKE_NUM)

    @property
    def like_email(self):
        """RETURNS (bool): Whether the token resembles an email address."""
        return Lexeme.c_check_flag(self.c, LIKE_EMAIL)
    
