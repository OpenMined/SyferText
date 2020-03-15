from .token import Token
import syft
import torch

hook = syft.TorchHook(torch)

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker

from typing import List
from typing import Union
from .underscore import Underscore


class Doc(AbstractObject):
    def __init__(
        self,
        vocab,
        text,
        id: int = None,
        owner: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
    ):

        super(Doc, self).__init__(
            id=id, owner=owner, tags=tags, description=description
        )

        self.vocab = vocab
        self.text = text

        # This list is populated in the __call__ method of the Tokenizer object.
        # Its members are objects of the TokenMeta class defined in the tokenizer.py
        # file
        self.container = list()

        # Initialize the Underscore object (inspired by spaCy)
        # This object will hold all the custom attributes set
        # using the `self.set_attribute` method
        self._ = Underscore()

    def set_attribute(self, name: str, value: object):
        """Creates a custom attribute with the name `name` and
           value `value` in the Underscore object `self._`

        Args:
            name (str): name of the custom attribute .
            value (object): value of the custom named attribute.
        """

        # make sure there is no space in name as well prevent empty name
        assert type(name) is str, "name must be of str type"
        assert (
            len(name) > 0 and (not (" " in name)) and (type(name) == str)
        ), "name cannot be empty or contain space"

        setattr(self._, name, value)

    def __getitem__(self, key: int):
        """Returns a Token object at position `key`.

        Args:
            key (int): the index of the token to return.
                Example: 0 -> first token, 1 -> second token

        Returns:
            Token: the token at index key
        """

        # Get the corresponding TokenMeta object
        token_meta = self.container[key]

        # Create a Token object
        token = Token(doc=self, token_meta=token_meta)

        return token

    @staticmethod
    def create_pointer(
        doc,
        location: BaseWorker = None,
        id_at_location: (str or int) = None,
        register: bool = False,
        owner: BaseWorker = None,
        ptr_id: (str or int) = None,
        garbage_collect_data: bool = True,
    ):
        """Creates a DocPointer object that points to a Doc object living in the the worker 'location'.

        Returns:
            DocPointer: pointer object to a document
        """

        # I put the import here in order to avoid circular imports
        from .pointers.doc_pointer import DocPointer

        if id_at_location is None:
            id_at_location = doc.id

        if owner is None:
            owner = doc.owner

        doc_pointer = DocPointer(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=ptr_id,
            garbage_collect_data=garbage_collect_data,
        )

        return doc_pointer

    def __len__(self):
        """Return the number of tokens in the Doc."""
        return len(self.container)

    def __iter__(self):
        """Allows to loop over tokens in `self.container`"""
        for i in range(len(self.container)):

            # Yield a Token object
            yield self[i]

    def get_encrypted_vector(self, *workers, crypto_provider=None, requires_grad=True):
        """Get the mean of the vectors of each Token in this documents.

        Args:
            self (Doc): current document.
            workers (sequence of BaseWorker): A sequence of remote workers from .
            crypto_provider (BaseWorker): A remote worker responsible for providing cryptography (SMPC encryption) functionalities.
            requires_grad (bool): A boolean flag indicating whether gradients are required or not.

        Returns:
            Tensor: A tensor representing the SMPC-encrypted vector of this document.
        """
        assert (
            len(workers) > 1
        ), "You need at least two workers in order to encrypt the vector with SMPC"

        # Accumulate the vectors here
        vectors = None

        # Count the tokens that have vectors
        vector_count = 0

        for token in self:

            # Get the encypted vector of the token if one exists
            if token.has_vector:

                # Increment the vector counter
                vector_count += 1

                # cumulate token's vector by summing them
                vectors = token.vector if vectors is None else vectors + token.vector

        # if no tokens with vectors were found, just get the default vector (zeros)
        if vector_count == 0:
            doc_vector = self.vocab.vectors.default_vector
        else:
            # Create the final Doc vector
            doc_vector = vectors / vector_count

        # Create a Syft/Torch tensor
        doc_vector = torch.Tensor(doc_vector)

        # Encrypt the vector using SMPC with PySyft
        doc_vector = doc_vector.fix_precision().share(
            *workers, crypto_provider=crypto_provider, requires_grad=requires_grad
        )

        return doc_vector
