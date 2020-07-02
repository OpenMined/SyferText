from .utils import hash_string
from .attrs import Attributes
from .lexeme import Lexeme
from .lexeme import LexemeMeta

import syft as sy
import torch
from typing import Union
from syft.generic.string import String
from syft.generic.abstract.object import AbstractObject
from syft.workers.base import BaseWorker


hook = sy.TorchHook(torch)


class Token(AbstractObject):
    def __init__(
        self,
        doc: "Doc",
        token_meta: "TokenMeta",
        position: int,
        id: int = None,
        owner: BaseWorker = None,
    ):
        super(Token, self).__init__(id=id, owner=owner)

        self.doc = doc

        # corresponding hash value of this token
        self.orth = token_meta.orth

        # LexMeta object for the corresponding token string
        self.lex_meta = self.doc.vocab.get_lex_meta(self.orth)

        # Whether the token is followed by a single white space
        self.space_after = token_meta.space_after
        self.position = position

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = token_meta._

        # Whether this token has a vector or not
        self.has_vector = self.doc.vocab.vectors.has_vector(self.orth_)

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

    def nbor(self, offset=1):
        """Gets the neighbouring token at `self.position + offset` if it exists

        Args:
            offset (int): the relative position of the neighbour with respect to current token.

        Returns:
            neighbor (Token): the neighbor of the current token with a relative position `offset`.
        """

        # The neighbor's index should be within the document's range of indices
        assert (
            0 <= self.position + offset < len(self.doc)
        ), f"Token at position {self.position + offset} does not exist"

        neighbor = self.doc[self.position + offset]

        return neighbor

    def check_flag(self, flag_id: int) -> bool:
        """Checks the attribute corresponding to given `flag_id` flag value.
        
        Args:
            flag_id(int): The attribute ID of the flag to check.
    
        Returns:
            bool: Returns True if the value of flag corresponding to flag_id is 1 else False.
        """

        return self.lex_meta.check_flag(flag_id)

    def __str__(self):
        # The call to `str()` in the following is to account for the case
        # when text is of type String or StringPointer (which are Syft string types)
        return self.orth_

    @property
    def text(self):
        """Get the token text in str type"""
        return self.orth_

    def set_flag(self, flag_id: int, value: bool) -> None:
        """Set the sets the value of flag corresponding flag_id.

        Args:
            flag_id(int): The flag_id for corresponding attribute to set.
            value(bool): boolean value used to set flag.
        """

        # Sets the value of flag which is inside lexememeta object
        self.lex_meta.set_flag(flag_id=flag_id, value=value)

    @property
    def text(self):
        """Get the token text"""
        return self.orth_

    def __len__(self):
        """The number of unicode characters in the token, i.e. `token.text`.
        The number of unicode characters in the token.
        """
        return self.lex_meta.length

    def __repr__(self):
        return f"Token[{self.orth_}]"

    @property
    def vector(self):
        """Get the token vector"""
        return self.doc.vocab.vectors[self.orth_]

    @property
    def vector_norm(self) -> torch.Tensor:
        """The L2 norm of the token's vector representation.

        Returns: 
            Tensor: The L2 norm of the vector representation.
        """
        
        return torch.sqrt((self.vector ** 2).sum())

    def similarity(self, other):
        """Compute the cosine similarity between tokens vectors.
        
        Args:
            other (Token): The Token to compare with.
        
        Returns:
            Tensor: A cosine similarity score. Higher is more similar.
        """

        # Make sure both vectors have non-zero norms
        assert (
            self.vector_norm.item() != 0.0 and other.vector_norm.item() != 0.0
        ), "One of the provided tokens has a zero norm."

        # Compute similarity
        sim = torch.dot(torch.tensor(self.vector), torch.tensor(other.vector))
        sim /= self.vector_norm * other.vector_norm

        return sim

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
        vector = self.doc.vocab.vectors[self.orth_]

        # Encrypt the vector using SMPC
        vector = vector.fix_precision().share(
            *workers, crypto_provider=crypto_provider, requires_grad=requires_grad
        )

        return vector

    # Following attributes are inspired from Spacy, they have similar behaviour as in spacy.
    # Some of the attributes are redundant but they are to maintain consistency with other attributes


    @property
    def text_with_ws(self):
        """The text content of the token with the trailing whitespace(if any)."""
        text = self.orth_

        if self.space_after:
            return text + " "
        else:
            return text

    @property
    def lex_id(self):
        """Sequential id of the token's lexical type. Used to index into words vector table"""
        return self.lex_meta.id

    @property
    def rank(self):
        """The index to corresponding word vector in words vector table."""
        return self.lex_meta.id

    @property
    def lower(self):
        """Orth id of the lowercase token text."""
        return self.lex_meta.lower

    @property
    def shape(self):
        """Orth id of the token's shape, a transform of the
            tokens's string, to show orthographic features (e.g. "Xxxx", "dd").
        """
        return self.lex_meta.shape

    @property
    def prefix(self):
        """Orth id of a length-1 substring from the start of the token."""
        return self.lex_meta.prefix

    @property
    def suffix(self):
        """Orth id of a length-N substring from the end of the token."""
        return self.lex_meta.suffix

    @property
    def lang(self):
        """Orth id of the language of the parent document's vocabulary."""
        return self.lex_meta.lang

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
        return self.doc.vocab.store[self.lex_meta.orth]

    @property
    def lower_(self):
        """The lowercase token text."""
        return self.doc.vocab.store[self.lex_meta.lower]

    @property
    def shape_(self):
        """Transform of the tokens's string, to show
        orthographic features. For example, "Xxxx" or "dd".
        """
        return self.doc.vocab.store[self.lex_meta.shape]

    @property
    def prefix_(self):
        """A length-1 substring from the start of the token."""
        return self.doc.vocab.store[self.lex_meta.prefix]

    @property
    def suffix_(self):
        """A length-3 substring from the end of the token."""
        return self.doc.vocab.store[self.lex_meta.suffix]

    @property
    def lang_(self):
        """Language of the parent document's vocabulary,
            e.g. 'en_web_core_lm'.
        """
        return self.doc.vocab.store[self.lex_meta.lang]

    @property
    def is_oov(self):
        """Whether the token is out-of-vocabulary."""
        return self.lex_meta.check_flag(Attributes.IS_OOV)

    @property
    def is_stop(self):
        """Whether the token is a stop word, i.e. part of a
            stop list defined by the language data.
        """
        return self.lex_meta.check_flag(Attributes.IS_STOP)

    @property
    def is_alpha(self):
        """Whether the token consists of alphabets characters."""
        return self.lex_meta.check_flag(Attributes.IS_ALPHA)

    @property
    def is_ascii(self):
        """Whether the token consists of ASCII characters."""
        return self.lex_meta.check_flag(Attributes.IS_ASCII)

    @property
    def is_digit(self):
        """Whether the token consists of digits."""
        return self.lex_meta.check_flag(Attributes.IS_DIGIT)

    @property
    def is_lower(self):
        """Whether the token is in lowercase."""
        return self.lex_meta.check_flag(Attributes.IS_LOWER)

    @property
    def is_upper(self):
        """Whether the token is in uppercase."""
        return self.lex_meta.check_flag(Attributes.IS_UPPER)

    @property
    def is_title(self):
        """Whether the token is in titlecase."""
        return self.lex_meta.check_flag(Attributes.IS_TITLE)

    @property
    def is_punct(self):
        """Whether the token is punctuation."""
        return self.lex_meta.check_flag(Attributes.IS_PUNCT)

    @property
    def is_space(self):
        """Whether the token consists of whitespace characters."""
        return self.lex_meta.check_flag(Attributes.IS_SPACE)

    @property
    def is_bracket(self):
        """Whether the token is a bracket."""
        return self.lex_meta.check_flag(Attributes.IS_BRACKET)

    @property
    def is_quote(self):
        """Whether the token is a quotation mark."""
        return self.lex_meta.check_flag(Attributes.IS_QUOTE)

    @property
    def is_left_punct(self):
        """Whether the token is a left punctuation mark."""
        return self.lex_meta.check_flag(Attributes.IS_LEFT_PUNCT)

    @property
    def is_right_punct(self):
        """Whether the token is a right punctuation mark."""
        return self.lex_meta.check_flag(Attributes.IS_RIGHT_PUNCT)

    @property
    def is_currency(self):
        """Whether the token is a currency symbol."""
        return self.lex_meta.check_flag(Attributes.IS_CURRENCY)

    @property
    def like_url(self):
        """Whether the token resembles a URL."""
        return self.lex_meta.check_flag(Attributes.LIKE_URL)

    @property
    def like_num(self):
        """Whether the token resembles a number, e.g. "10.9",
        "10" etc.
        """
        return self.lex_meta.check_flag(Attributes.LIKE_NUM)

    @property
    def like_email(self):
        """Whether the token resembles an email address."""
        return self.lex_meta.check_flag(Attributes.LIKE_EMAIL)
