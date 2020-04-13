from .attrs import Attributes
import numpy as np

class LexemeMeta(object):
    """This class holds some meta data about a Lexeme from the text held by a Doc object.
       This allows to create aLexeme object when needed.
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



class Lexeme:
    """ Inspired from Spacy Lexeme class. It is an entry in the vocabulary.
    A `Lexeme` has no string context – it's a word-type, as opposed to a 
    word token. It holds various attributes related to the corresponding word.  
    """
    def __init__(self, vocab: "Vocab", orth: int):
        """Initiate a Lexeme object.

        Args:
            vocab (Vocab): The parent vocabulary
            orth (int): The orth id of the lexeme.
        """

        self.vocab = vocab
        self.orth = orth
        
        # Get the LexMeta stored in Vocab lex_store
        # Note: It created ne entry in lex_store if the LexemeMeta is not already present
        self.lex = vocab.get_by_orth(orth)
        
    @staticmethod
    def set_lex_attr( lex: LexemeMeta, attr_id, value):

        # Assign the flag attribute of `LexemeMeta` object.
        # All flags have id < 65. check `Attributes` for refrence ids.
        if attr_id < 65:
            Lexeme.set_flag(lex, attr_id, value)

        # Assign the rest of the `LexemeMeta` object attributes.
        # length and orth attributes are assigned in Vocab class.
        elif attr_id == Attributes.ID:
            lex.id = value

        elif attr_id == Attributes.LOWER:
            lex.lower = value

        elif attr_id == Attributes.SHAPE:
            lex.shape = value

        elif attr_id == Attributes.PREFIX:
            lex.prefix = value

        elif attr_id == Attributes.SUFFIX:
            lex.suffix = value
            
        elif attr_id == Attributes.LANG:
            lex.lang = value

    @staticmethod
    def check_flag(lex: LexemeMeta , flag_id) :
        one = 1
        if lex.flags & (one << flag_id):
            return True
        else:
            return False

    @staticmethod
    def set_flag(lex: LexemeMeta , flag_id, value) :
        one = 1
        if value:
            lex.flags |= one << flag_id
        else:
            lex.flags &= ~(one << flag_id)
    

    @property
    def has_vector(self):
        """Whether the word have a vector in vocabulary."""
        return self.vocab.has_vector(self.orth_)

    @property
    def vector_norm(self):
        """The L2 norm of the vector."""
        vector = self.vector

        return np.sqrt((vector**2).sum())


    @property
    def vector(self):
        """ A vector for given theword in Voacnulary"""
        return self.vocab.get_vector(self.lex.orth)

    @property 
    def rank(self):
        """ The key to index in the vectors table"""
        return self.lex.id
    
    @property
    def orth_(self):
        """The original text of the lexeme(identical to `Lexeme.text`). 
            Exists mostly for consistency with the other attributes.
        """
        return self.vocab.strings[self.lex.orth]

    @property
    def text(self):
        """ The original text of the lexeme."""
        return self.orth_

    @property 
    def lower(self):
        """Orth id of lowercase form of the lexeme."""
        return self.lex.lower


    @property 
    def shape(self):
        """Orth id of transform of the word's string, to show orthographic features."""
        return self.lex.shape

    @property 
    def prefix(self):
        """Orth id of length-1 substring from the start of the word. """
        return self.lex.prefix

    @property 
    def suffix(self):
        """Orth id of length-3 substring from the end of the word."""
        return self.lex.suffix

    @property 
    def lang(self):
        """Orth id of language of the parent vocabulary."""
        return self.lex.lang

    @property 
    def lower_(self):
        """Lowercase form of the word."""
        return self.vocab.strings[self.lower]


    @property 
    def shape_(self):
        """Transform of the word's string, to show orthographic features."""
        return self.vocab.strings[self.lex.shape]

    @property 
    def prefix_(self):
        """Length-1 substring from the start of the word."""
        return self.vocab.strings[self.lex.prefix]

    @property 
    def suffix_(self):
        """Length-3 substring from the end of the word."""
        return self.vocab.strings[self.lex.suffix]
        
    @property 
    def lang_(self):
        """Language of the parent vocabulary."""
        return self.vocab.strings[self.lex.lang]


    @property
    def is_oov(self):
        """Whether the lexeme is out-of-vocabulary."""
        return Lexeme.c_check_flag(self.lex, IS_OOV)

    @property
    def is_stop(self):
        """Whether the lexeme is a stop word, i.e. part of a
            stop list defined by the language data.
        """
        return Lexeme.c_check_flag(self.lex, IS_STOP)

    @property
    def is_alpha(self):
        """Whether the lexeme consists of alpha characters."""
        return Lexeme.c_check_flag(self.lex, IS_ALPHA)

    @property
    def is_ascii(self):
        """Whether the lexeme consists of ASCII characters."""
        return Lexeme.c_check_flag(self.lex, IS_ASCII)

    @property
    def is_digit(self):
        """Whether the lexeme consists of digits."""
        return Lexeme.c_check_flag(self.lex, IS_DIGIT)

    @property
    def is_lower(self):
        """Whether the lexeme is in lowercase."""
        return Lexeme.c_check_flag(self.lex, IS_LOWER)

    @property
    def is_upper(self):
        """Whether the lexeme is in uppercase."""
        return Lexeme.c_check_flag(self.lex, IS_UPPER)

    @property
    def is_title(self):
        """Whether the lexeme is in titlecase."""
        return Lexeme.c_check_flag(self.lex, IS_TITLE)

    @property
    def is_punct(self):
        """Whether the lexeme is punctuation."""
        return Lexeme.c_check_flag(self.lex, IS_PUNCT)

    @property
    def is_space(self):
        """Whether the lexeme consists of only whitespace characters."""
        return Lexeme.c_check_flag(self.lex, IS_SPACE)

    @property
    def is_bracket(self):
        """Whether the lexeme is a bracket."""
        return Lexeme.c_check_flag(self.lex, IS_BRACKET)

    @property
    def is_quote(self):
        """Whether the lexeme is a quotation mark."""
        return Lexeme.c_check_flag(self.lex, IS_QUOTE)

    @property
    def is_left_punct(self):
        """Whether the lexeme is a left punctuation mark."""
        return Lexeme.c_check_flag(self.lex, IS_LEFT_PUNCT)

    @property
    def is_right_punct(self):
        """Whether the lexeme is a right punctuation mark."""
        return Lexeme.c_check_flag(self.lex, IS_RIGHT_PUNCT)

    @property
    def is_currency(self):
        """Whether the lexeme is a currency symbol."""
        return Lexeme.c_check_flag(self.lex, IS_CURRENCY)

    @property
    def like_url(self):
        """Whether the lexeme resembles a URL."""
        return Lexeme.c_check_flag(self.lex, LIKE_URL)

    @property
    def like_num(self):
        """Whether the lexeme resembles a number, e.g. "10.9",
            "10", "ten", etc.
        """
        return Lexeme.c_check_flag(self.lex, LIKE_NUM)

    @property
    def like_email(self):
        """Whether the lexeme resembles an email address."""
        return Lexeme.c_check_flag(self.lex, LIKE_EMAIL)
    
