import syft as sy
import torch

hook = sy.TorchHook(torch)

from .underscore import Underscore


class Token:
    def __init__(self, doc: "Doc", hash: int, space_after: bool, is_space: bool):
        """Initializes a Token object

        Args:
            doc (Doc): Doc object which stores this token in it's container
            hash (int): hash value of the string stored by the Token object
            space_after (bool): Whether the token is followed by a single white
                space (True) or not (False).
            is_space (bool): Whether the token itself is composed of only white
                spaces (True) or not (false).

        """

        self.doc = doc

        # corresponding hash value of this token
        self.hash = hash

        self.space_after = space_after
        self.is_space = is_space

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()

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
        return str(self.doc.vocab.store[self.hash])

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
