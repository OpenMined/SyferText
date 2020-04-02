
def class Lexeme:
    """An entry in the vocabulary. A `Lexeme` has no string context – it's a
    word-type, as opposed to a word token.  It therefore has no part-of-speech
    tag, dependency parse, or lemma (lemmatization depends on the
    part-of-speech tag).
    DOCS: https://spacy.io/api/lexeme
    """
    def __init__(self, Vocab vocab, orth):
        """Create a Lexeme object.
        vocab (Vocab): The parent vocabulary
        orth (uint64): The orth id of the lexeme.
        Returns (Lexeme): The newly constructd object.
        """
        self.vocab = vocab
        self.orth = orth
        
    @staticmethod
    def set_lex_attr(Lexeme lex,  name,  value):
        setattr(lex,name,value)

    @staticmethod
    def check_lex_attr(Lexeme lex, name):
        getattr(lex,name, default= False)
    

    @property
    def has_vector(self):
        """RETURNS (bool): Whether a word vector is associated with the object.
        """
        return self.vocab.has_vector(self.orth)

    @property
    def vector_norm(self):
        """RETURNS (float): The L2 norm of the vector representation."""
        vector = self.vector
        return numpy.sqrt((vector**2).sum())

    property vector:
        """A real-valued meaning representation.
        RETURNS (numpy.ndarray[ndim=1, dtype='float32']): A 1D numpy array
            representing the lexeme's semantics.
        """
        def __get__(self):
            length = self.vocab.vectors_length
            if length == 0:
                raise ValueError(Errors.E010)
            return self.vocab.get_vector(self.c.orth)

        def __set__(self, vector):
            if len(vector) != self.vocab.vectors_length:
                raise ValueError(Errors.E073.format(new_length=len(vector),
                                                    length=self.vocab.vectors_length))
            self.vocab.set_vector(self.orth, vector)
    #To do
    # add Rank

    @property
    def orth_(self):
        """RETURNS (unicode): The original verbatim text of the lexeme
            (identical to `Lexeme.text`). Exists mostly for consistency with
            the other attributes."""
        return self.vocab.strings[self.orth]

    @property
    def text(self):
        """RETURNS (unicode): The original verbatim text of the lexeme."""
        return self.orth_

    property lower:
        """RETURNS (unicode): Lowercase form of the lexeme."""
        def __get__(self):
            return self.lower

        def __set__(self, x):
            self.lower = x

    property norm:
        """RETURNS (uint64): The lexemes's norm, i.e. a normalised form of the
            lexeme text.
        """
        def __get__(self):
                return self.norm

        def __set__(self, x):
            self.norm = x

    property shape:
        """RETURNS (uint64): Transform of the word's string, to show
            orthographic features.
        """
        def __get__(self):
            return self.shape

        def __set__(self, x):
            self.shape = x

    property prefix:
        """RETURNS (uint64): Length-N substring from the start of the word.
            Defaults to `N=1`.
        """
        def __get__(self):
            return self.prefix

        def __set__(self, x):
            self.prefix = x

    property suffix:
        """RETURNS (uint64): Length-N substring from the end of the word.
            Defaults to `N=3`.
        """
        def __get__(self):
            return self.suffix

        def __set__(self, x):
            self.suffix = x

    property lang:
        """RETURNS (uint64): Language of the parent vocabulary."""
        def __get__(self):
            return self.lang

        def __set__(self, x):
            self.lang = x


    property lower_:
        """RETURNS (unicode): Lowercase form of the word."""
        def __get__(self):
            return self.vocab.strings[self.lower]

        def __set__(self, x):
            self.lower = self.vocab.strings.add(x)

    property norm_:
        """RETURNS (unicode): The lexemes's norm, i.e. a normalised form of the
            lexeme text.
        """
        def __get__(self):
            return self.vocab.strings[self.norm]

        def __set__(self, x):
            self.norm = self.vocab.strings.add(x)

    property shape_:
        """RETURNS (unicode): Transform of the word's string, to show
            orthographic features.
        """
        def __get__(self):
            return self.vocab.strings[self.shape]

        def __set__(self, x):
            self.shape = self.vocab.strings.add(x)

    property prefix_:
        """RETURNS (unicode): Length-N substring from the start of the word.
            Defaults to `N=1`.
        """
        def __get__(self):
            return self.vocab.strings[self.prefix]

        def __set__(self, x):
            self.prefix = self.vocab.strings.add(x)

    property suffix_:
        """RETURNS (unicode): Length-N substring from the end of the word.
            Defaults to `N=3`.
        """
        def __get__(self):
            return self.vocab.strings[self.suffix]

        def __set__(self, x):
            self.suffix = self.vocab.strings.add(x)

    property lang_:
        """RETURNS (unicode): Language of the parent vocabulary."""
        def __get__(self):
            return self.vocab.strings[self.lang]

        def __set__(self, x):
            self.lang = self.vocab.strings.add(x)

    property is_oov:
        """RETURNS (bool): Whether the lexeme is out-of-vocabulary."""
        def __get__(self):
            return Lexeme.c(self.c, IS_OOV)

        def __set__(self, attr_t x):
            Lexeme.c_set_flag(self.c, IS_OOV, x)

    property is_stop:
        """RETURNS (bool): Whether the lexeme is a stop word."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_STOP)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_STOP, x)

    property is_alpha:
        """RETURNS (bool): Whether the lexeme consists of alphabetic
            characters. Equivalent to `lexeme.text.isalpha()`.
        """
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_ALPHA)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_ALPHA, x)

    property is_ascii:
        """RETURNS (bool): Whether the lexeme consists of ASCII characters.
            Equivalent to `[any(ord(c) >= 128 for c in lexeme.text)]`.
        """
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_ASCII)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_ASCII, x)

    property is_digit:
        """RETURNS (bool): Whether the lexeme consists of digits. Equivalent
            to `lexeme.text.isdigit()`.
        """
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_DIGIT)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_DIGIT, x)

    property is_lower:
        """RETURNS (bool): Whether the lexeme is in lowercase. Equivalent to
            `lexeme.text.islower()`.
        """
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_LOWER)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_LOWER, x)

    property is_upper:
        """RETURNS (bool): Whether the lexeme is in uppercase. Equivalent to
            `lexeme.text.isupper()`.
        """
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_UPPER)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_UPPER, x)

    property is_title:
        """RETURNS (bool): Whether the lexeme is in titlecase. Equivalent to
            `lexeme.text.istitle()`.
        """
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_TITLE)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_TITLE, x)

    property is_punct:
        """RETURNS (bool): Whether the lexeme is punctuation."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_PUNCT)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_PUNCT, x)

    property is_space:
        """RETURNS (bool): Whether the lexeme consist of whitespace characters.
            Equivalent to `lexeme.text.isspace()`.
        """
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_SPACE)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_SPACE, x)

    property is_bracket:
        """RETURNS (bool): Whether the lexeme is a bracket."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_BRACKET)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_BRACKET, x)

    property is_quote:
        """RETURNS (bool): Whether the lexeme is a quotation mark."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_QUOTE)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_QUOTE, x)

    property is_left_punct:
        """RETURNS (bool): Whether the lexeme is left punctuation, e.g. )."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_LEFT_PUNCT)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_LEFT_PUNCT, x)

    property is_right_punct:
        """RETURNS (bool): Whether the lexeme is right punctuation, e.g. )."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_RIGHT_PUNCT)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_RIGHT_PUNCT, x)

    property is_currency:
        """RETURNS (bool): Whether the lexeme is a currency symbol, e.g. $, €."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, IS_CURRENCY)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, IS_CURRENCY, x)

    property like_url:
        """RETURNS (bool): Whether the lexeme resembles a URL."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, LIKE_URL)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, LIKE_URL, x)

    property like_num:
        """RETURNS (bool): Whether the lexeme represents a number, e.g. "10.9",
            "10", "ten", etc.
        """
        def __get__(self):
            return Lexeme.c_check_flag(self.c, LIKE_NUM)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, LIKE_NUM, x)

    property like_email:
        """RETURNS (bool): Whether the lexeme resembles an email address."""
        def __get__(self):
            return Lexeme.c_check_flag(self.c, LIKE_EMAIL)

        def __set__(self, bint x):
            Lexeme.c_set_flag(self.c, LIKE_EMAIL, x)