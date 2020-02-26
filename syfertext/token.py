from .utils import hash_string

import syft as sy
import torch

hook = sy.TorchHook(torch)


class Token:
    def __init__(
        self, doc, start_pos: int, stop_pos: int, is_space: bool, space_after: bool
    ):

        self.doc = doc
        self.start_pos = start_pos
        self.stop_pos = stop_pos
        self.is_space = is_space
        self.space_after = space_after

        # Whether this token has a vector or not
        self.has_vector = self.doc.vocab.vectors.has_vector(self.text)
        
    def __str__(self):

        # The call to `str()` in the following is to account for the case
        # when text is of type String or StringPointer (which are Syft string
        # types)
        return self.text

    @property
    def orth(self):
        """Get the corresponding hash value of this token"""
        return hash_string(str(self))

    @property
    def text(self):
        """Get the token text"""
        return str(self.doc.text[self.start_pos : self.stop_pos])
    
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
