from .token import Token
import syft
import torch

hook = syft.TorchHook(torch)

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker

from typing import List
from typing import Union


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

        # The start and stop positions of the token in self.text
        # notice that stop_position refers to one position after `token_meta.end_pos`.
        # this is practical for indexing
        start_pos = token_meta.start_pos
        stop_pos = token_meta.end_pos + 1 if token_meta.end_pos is not None else None

        # Create a Token object
        token = Token(
            doc=self,
            # string = self.text[start_pos:end_pos],
            start_pos=start_pos,
            stop_pos=stop_pos,
            is_space=token_meta.is_space,
            space_after=token_meta.space_after,
        )

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

    @property
    def vector(self):
        """Get document vector as an average of in-vocabulary token's vectors """

        # Accumulate the vectors here
        vectors = None

        # Count the tokens that have vectors
        vector_count = 0

        for token in self:

            # Get the encrypted vector of the token if one exists
            if token.has_vector:

                # Increment the vector counter
                vector_count += 1

                # Cumulate token's vector by summing them
                vectors = token.vector if vectors is None else vectors + token.vector

        # If no tokens with vectors were found, just get the default vector(zeros)
        if vector_count == 0:
            doc_vector = self.vocab.vectors.default_vector
        else:
            # Create the final Doc vector
            doc_vector = vectors / vector_count
                
        return doc_vector
    
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

        
        # Storing the average of vectors of each in-vocabulary token's vectors
        doc_vector = self.vector
        
        # Create a Syft/Torch tensor
        doc_vector = torch.Tensor(doc_vector)

        # Encrypt the vector using SMPC with PySyft
        doc_vector = doc_vector.fix_precision().share(
            *workers, crypto_provider=crypto_provider, requires_grad=requires_grad
        )

        return doc_vector
