from .utils import hash_string
from .attrs import Attributes
from .lexeme import Lexeme
from .lexeme import LexemeMeta

import syft as sy
import torch
import numpy as np


hook = sy.TorchHook(torch)


class Token:
    def __init__(self, vocab, doc, token_meta: "TokenMeta"):

        self.vocab = vocab
        self.doc = doc

        # corresponding hash value of this token
        self.orth = token_meta.orth

        # LexMeta object for the corresponding token string
        self.lex = vocab.get_by_orth(self.orth)

        # Whether the token is followed by a single white
        self.space_after = token_meta.space_after

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = token_meta._

        # Whether this token has a vector or not
        self.has_vector = self.doc.vocab.vectors.has_vector(self.text)

    def __str__(self):

        # The call to `str()` in the following is to account for the case
        # when text is of type String or StringPointer (which are Syft string types)
        return self.text

    def set_attribute(self, name: str, value: object):
        """Creates a custom attribute with the name `name` and
           value `value` in the Underscore object `self._`
        """

        setattr(self._, name, value)

    @property
    def text(self):
        """Get the token text"""
        return self.orth_

    def __len__(self):
        """The number of unicode characters in the token, i.e. `token.text`.
        The number of unicode characters in the token.
        """
        return self.lex.length

    @property
    def vector(self):
        """Get the token vector"""
        return self.doc.vocab.vectors[self.text]

    def get_encrypted_vector(self, *workers, crypto_provider=None, requires_grad=True):
        """Get the mean of the vectors of each Token in this documents.

        Args:
            self (Token): current token.
            workers (sequence of BaseWorker): A sequence of remote workers from .
            crypto_provider (BaseWorker): A remote worker responsible for providing cryptography (SMPC encryption) functionalities.
            requires_grad (bool): A boolean flag indicating whether gradients are required or not.

        Returns:
            Tensor: A tensor representing the SMPC-encrypted vector of this token.
        """
        assert (
            len(workers) > 1
        ), "You need at least two workers in order to encrypt the vector with SMPC"

        # Get the vector
        vector = self.doc.vocab.vectors[self.text]

        # Create a Syft/Torch tensor
        vector = torch.Tensor(vector)

        # Encrypt the vector using SMPC
        vector = vector.fix_precision().share(
            *workers, crypto_provider=crypto_provider, requires_grad=requires_grad
        )

        return vector
    
    # Following attributes are inspired from Spacy, they have similar behaviour as in spacy. 
    # Some of the attributes are redundant but there to maintain consistency with other attributes

    @property
    def vector_norm(self):
        """The L2 norm of the token's vector"""
        vector = self.vector
        return np.sqrt((vector**2).sum())
    
    @property
    def text_with_ws(self):
        """The text content of the token with the trailing whitespace(if any)."""
        text = self.orth_

        if self.is_space:
            return text + " "
        else:
            return text
    
    @property
    def lex_id(self):
        """Sequential id of the token's lexical type. Used to index into words vector table"""
        return self.lex.id

    @property
    def rank(self):
        """The index to corresponding word vector in words vector table."""
        return self.lex.id

    @property
    def lower(self):
        """Orth id of the lowercase token text."""
        return self.lex.lower

    @property
    def shape(self):
        """Orth id of the token's shape, a transform of the
            tokens's string, to show orthographic features (e.g. "Xxxx", "dd").
        """
        return self.lex.shape

    @property
    def prefix(self):
        """Orth id of a length-1 substring from the start of the token."""
        return self.lex.prefix

    @property
    def suffix(self):
        """Orth id of a length-N substring from the end of the token."""
        return self.lex.suffix

    @property
    def lang(self):
        """Orth id of the language of the parent document's vocabulary."""
        return self.lex.lang

    @property
    def whitespace_(self):
        """The trailing whitespace character, if present."""
        return " " if self.space_after else ""

    @property
    def orth_(self):
        """Text content (identical to `Token.text`).
            Exists mostly for consistency with the other
            attributes.
        """
        return self.vocab.store[self.lex.orth]

    @property
    def lower_(self):
        """The lowercase token text."""
        return self.vocab.store[self.lex.lower]

    @property
    def shape_(self):
        """Transform of the tokens's string, to show
            orthographic features. For example, "Xxxx" or "dd".
        """
        return self.vocab.store[self.lex.shape]

    @property
    def prefix_(self):
        """A length-1 substring from the start of the token."""
        return self.vocab.store[self.lex.prefix]

    @property
    def suffix_(self):
        """A length-3 substring from the end of the token."""
        return self.vocab.store[self.lex.suffix]

    @property
    def lang_(self):
        """Language of the parent document's vocabulary,
            e.g. 'en_web_core_lm'.
        """
        return self.vocab.store[self.lex.lang]

    @property
    def is_oov(self):
        """Whether the token is out-of-vocabulary."""
        return Lexeme.check_flag(self.lex, Attributes.IS_OOV)

    @property
    def is_stop(self):
        """Whether the token is a stop word, i.e. part of a
            stop list defined by the language data.
        """
        return Lexeme.check_flag(self.lex, Attributes.IS_STOP)

    @property
    def is_alpha(self):
        """Whether the token consists of alpha characters."""
        return Lexeme.check_flag(self.lex, Attributes.IS_ALPHA)

    @property
    def is_ascii(self):
        """Whether the token consists of ASCII characters."""
        return Lexeme.check_flag(self.lex, Attributes.IS_ASCII)

    @property
    def is_digit(self):
        """Whether the token consists of digits."""
        return Lexeme.check_flag(self.lex, Attributes.IS_DIGIT)

    @property
    def is_lower(self):
        """Whether the token is in lowercase."""
        return Lexeme.check_flag(self.lex, Attributes.IS_LOWER)

    @property
    def is_upper(self):
        """Whether the token is in uppercase."""
        return Lexeme.check_flag(self.lex, Attributes.IS_UPPER)

    @property
    def is_title(self):
        """Whether the token is in titlecase."""
        return Lexeme.check_flag(self.lex, Attributes.IS_TITLE)

    @property
    def is_punct(self):
        """Whether the token is punctuation."""
        return Lexeme.check_flag(self.lex, Attributes.IS_PUNCT)

    @property
    def is_space(self):
        """Whether the token consists of whitespace characters."""
        return Lexeme.check_flag(self.lex, Attributes.IS_SPACE)

    @property
    def is_bracket(self):
        """Whether the token is a bracket."""
        return Lexeme.check_flag(self.lex, Attributes.IS_BRACKET)

    @property
    def is_quote(self):
        """Whether the token is a quotation mark."""
        return Lexeme.check_flag(self.lex, Attributes.IS_QUOTE)

    @property
    def is_left_punct(self):
        """Whether the token is a left punctuation mark."""
        return Lexeme.check_flag(self.lex, Attributes.IS_LEFT_PUNCT)

    @property
    def is_right_punct(self):
        """Whether the token is a right punctuation mark."""
        return Lexeme.check_flag(self.lex, Attributes.IS_RIGHT_PUNCT)

    @property
    def is_currency(self):
        """Whether the token is a currency symbol."""
        return Lexeme.check_flag(self.lex, Attributes.IS_CURRENCY)

    @property
    def like_url(self):
        """Whether the token resembles a URL."""
        return Lexeme.check_flag(self.lex, Attributes.LIKE_URL)

    @property
    def like_num(self):
        """Whether the token resembles a number, e.g. "10.9",
            "10", "ten", etc.
        """
        return Lexeme.check_flag(self.lex, Attributes.LIKE_NUM)

    @property
    def like_email(self):
        """Whether the token resembles an email address."""
        return Lexeme.check_flag(self.lex, Attributes.LIKE_EMAIL)