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

    def __str__(self):

        # The call to `str()` in the following is to account for the case
        # when text is of type String or StringPointer (which are Syft string
        # types)
        return str(self.doc.text[self.start_pos : self.stop_pos])

    @property
    def orth(self):
        """
           Get the corresponding hash value of this token
        """

        return hash_string(str(self))

    @property
    def vector(self):
        """
           Get the token vector
        """

        return self.doc.vocab.vectors[self.__str__()]

    def getEncryptedVector(self, *workers, crypto_provider=None, requires_grad=True):

        assert (
            len(workers) > 1
        ), "You need at least two workers in order to encrypt the vector with SMPC"

        # Get the vector
        vector = self.doc.vocab.vectors[self.__str__()]

        # Create a Syft/Torch tensor
        vector = torch.Tensor(vector)

        # Encrypt the vector using SMPC
        vector = vector.fix_precision().share(
            *workers, crypto_provider=crypto_provider, requires_grad=requires_grad
        )

        return vector
