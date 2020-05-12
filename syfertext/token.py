from .utils import hash_string

import syft as sy
import torch
from syft.generic.string import String

from syft.generic.object import AbstractObject
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

        # The start and stop positions of the token in self.orth_
        # notice that stop_position refers to one position after `token_meta.end_pos`.
        # this is practical for indexing
        self.start_pos = token_meta.start_pos
        self.stop_pos = token_meta.end_pos + 1 if token_meta.end_pos is not None else None
        self.is_space = token_meta.is_space
        self.space_after = token_meta.space_after
        self.position = position

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = token_meta._

        # Whether this token has a vector or not
        self.has_vector = self.doc.vocab.vectors.has_vector(self.orth_)

    def __len__(self):
        """Returns the length of the token's text."""

        return len(self.text)

    def __str__(self):

        # The call to `str()` in the following is to account for the case
        # when text is of type String or StringPointer (which are Syft string types)
        return self.orth_

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
        assert self.has_attribute(name), "token does not have the attribute {}".format(name)

        delattr(self._, name)

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
        ), "Token at position {} does not exist".format(self.position + offset)

        neighbor = self.doc[self.position + offset]

        return neighbor

    @property
    def text(self):
        """Get the token text in str type"""
        return self.orth_

    @property
    def orth_(self):
        """Get the token text in str type"""
        return str(self.doc.vocab.store[self.orth])

    def __len__(self):
        """Get the length of the token"""
        return len(self.orth_)

    @property
    def text_with_ws(self) -> str:
        """Get the text with trailing whitespace if it exists"""

        if self.space_after:
            return self.orth_ + " "
        else:
            return self.orth_

    def __repr__(self):
        return "Token[{}]".format(self.orth_)

    @property
    def vector(self):
        """Get the token vector"""
        return self.doc.vocab.vectors[self.orth_]

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

        # Create a Syft/Torch tensor
        vector = torch.Tensor(vector)

        # Encrypt the vector using SMPC
        vector = vector.fix_precision().share(
            *workers, crypto_provider=crypto_provider, requires_grad=requires_grad
        )

        return vector
