from .utils import hash_string

import syft as sy
import torch

hook = sy.TorchHook(torch)


class Token:
    def __init__(self, doc, token_meta: "TokenMeta"):

        self.doc = doc

        # corresponding hash value of this token
        self.orth = token_meta.orth

        # The start and stop positions of the token in self.text
        # notice that stop_position refers to one position after `token_meta.end_pos`.
        # this is practical for indexing
        self.start_pos = token_meta.start_pos
        self.stop_pos = token_meta.end_pos + 1 if token_meta.end_pos is not None else None
        self.is_space = token_meta.is_space
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
        return str(self.doc.vocab.store[self.orth])

    @property
    def text_with_ws(self):
        """Get the text with trailing whitespace if it exists"""
        if self.space_after:
            return self.text + " "
        else:
            return self.text

    def __repr__(self):
        return self.text

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
