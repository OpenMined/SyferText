
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
        
    def set_attrs(self, **attrs):
        cdef attr_id_t attr
        attrs = intify_attrs(attrs)
        for attr, value in attrs.items():
            
            if isinstance(value, int) or isinstance(value, long):
                Lexeme.set_lex_attr(self, attr, value)
            else:
                Lexeme.set_lex_attr(self, attr, self.vocab.strings.add(value))

    @staticmethod
    def set_lex_attr(Lexeme lex, attr_id_t name, attr_t value):
        if name < (sizeof(flags_t) * 8):
            Lexeme.c_set_flag(lex, name, value)
        elif name == ID:
            lex.id = value
        elif name == LOWER:
            lex.lower = value
        elif name == NORM:
            lex.norm = value
        elif name == SHAPE:
            lex.shape = value
        elif name == PREFIX:
            lex.prefix = value
        elif name == SUFFIX:
            lex.suffix = value
        elif name == CLUSTER:
            lex.cluster = value
        elif name == LANG:
            lex.lang = value

    @staticmethod
    def c_check_flag(Lexeme lexeme, attr_id_t flag_id):
        flags_t one = 1
        if lexeme.flags & (one << flag_id):
            return True
        else:
            return False

    @staticmethod
    def c_set_flag(Lexeme lex, attr_id_t flag_id, bint value):
        flags_t one = 1
        if value:
            lex.flags |= one << flag_id
        else:
            lex.flags &= ~(one << flag_id)
    
    def set_flag(self, attr_id_t flag_id, bint value):
        """Change the value of a boolean flag.
        flag_id (int): The attribute ID of the flag to set.
        value (bool): The new value of the flag.
        """
        Lexeme.c_set_flag(self, flag_id, value)

    def check_flag(self, attr_id_t flag_id):
        """Check the value of a boolean flag.
        flag_id (int): The attribute ID of the flag to query.
        RETURNS (bool): The value of the flag.
        """
        return True if Lexeme.c_check_flag(self, flag_id) else False