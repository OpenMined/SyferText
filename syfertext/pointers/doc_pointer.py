from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import pickle

from typing import List
from typing import Union


class DocPointer(ObjectPointer):
    def __init__(
        self,
        location: BaseWorker = None,
        id_at_location: Union[str, int] = None,
        owner: BaseWorker = None,
        id: Union[str, int] = None,
        garbage_collect_data: bool = True,
        tags: List[str] = None,
        description: str = None,
    ):

        super(DocPointer, self).__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=id,
            garbage_collect_data=garbage_collect_data,
            tags=tags,
            description=description,
        )

    def get_encrypted_vector(self, *workers, crypto_provider=None, requires_grad=True):
        """Get the mean of the vectors of each Token in this documents.

        Args:
            self (DocPointer): current pointer to a remote document.
            workers (sequence of BaseWorker): A sequence of remote workers from .
            crypto_provider (BaseWorker): A remote worker responsible for providing cryptography (SMPC encryption) functionalities.
            requires_grad (bool): A boolean flag indicating whether gradients are required or not.

        Returns:
            Tensor: A tensor representing the SMPC-encrypted vector of the Doc this pointer points to.
        """

        assert (
            len(workers) > 1
        ), "You need at least two workers in order to encrypt the vector with SMPC"

        # Create the command
        kwargs = dict(crypto_provider=crypto_provider, requires_grad=requires_grad)
        command = ("get_encrypted_vector", self.id_at_location, workers, kwargs)

        # Send the command
        doc_vector = self.owner.send_command(self.location, command)

        # I call get because the returned object is a PointerTensor to the AdditiveSharedTensor
        doc_vector = doc_vector.get()

        return doc_vector

    def get_encrypted_token_vectors(self, *workers, crypto_provider=None, requires_grad=True):
        """Get the mean of the vectors of each Token in this documents.

        Args:
            self (DocPointer): current pointer to a remote document.
            workers (sequence of BaseWorker): A sequence of remote workers from .
            crypto_provider (BaseWorker): A remote worker responsible for providing cryptography (SMPC encryption) functionalities.
            requires_grad (bool): A boolean flag indicating whether gradients are required or not.

        Returns:
            Tensor: A SMPC-encrypted tensor representing the array of all vectors in the document this pointer points to.
        """

        assert (
            len(workers) > 1
        ), "You need at least two workers in order to encrypt the vector with SMPC"

        # Create the command
        kwargs = dict(crypto_provider=crypto_provider, requires_grad=requires_grad)
        command = ("get_encrypted_token_vectors", self.id_at_location, workers, kwargs)

        # Send the command
        token_vectors = self.owner.send_command(self.location, command)

        # We call get because the returned object is a PointerTensor to the AdditiveSharedTensor
        token_vectors = token_vectors.get()

        return token_vectors

    def __len__(self):

        # Create the command
        command = ("__len__", self.id_at_location, [], {})

        # Send the command
        length = self.owner.send_command(self.location, command)

        return length
