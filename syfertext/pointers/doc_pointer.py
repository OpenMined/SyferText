from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import pickle

from typing import List
from typing import Union
from typing import Dict
from typing import Set


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

    def get_encrypted_vector(
        self,
        *workers: BaseWorker,
        crypto_provider: BaseWorker = None,
        requires_grad: bool = True,
        excluded_tokens: Dict[str, Set[object]] = None,
    ):
        """Get the mean of the vectors of each Token in this documents.

        Args:
            workers (sequence of BaseWorker): A sequence of remote workers from .
            crypto_provider (BaseWorker): A remote worker responsible for providing cryptography
            (SMPC encryption) functionalities.
            requires_grad (bool): A boolean flag indicating whether gradients are required or not.
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes, the keys are the attributes names and they index, for efficiency,
                sets of values.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            Tensor: A tensor representing the SMPC-encrypted vector of the Doc this pointer points to.
        """

        # You need at least two workers in order to encrypt the vector with SMPC
        assert len(workers) > 1

        # Create the command
        kwargs = dict(
            crypto_provider=crypto_provider,
            requires_grad=requires_grad,
            excluded_tokens=excluded_tokens,
        )
        command = ("get_encrypted_vector", self.id_at_location, workers, kwargs)

        # Send the command
        doc_vector = self.owner.send_command(self.location, command)

        # I call get because the returned object is a PointerTensor to the AdditiveSharedTensor
        doc_vector = doc_vector.get()

        return doc_vector

    def get_encrypted_tokens_set(self):
        """Encrypt doc's tokens using owner's key.
        Returns:
            Set of tokens encrypted using owner's secret key `self.owner.secret`.
        """

        # Create the command
        command = ("get_encrypted_tokens_set", self.id_at_location, [], {})

        # Send the command
        enc_tokens = self.owner.send_command(self.location, command)

        return enc_tokens

    def get_encrypted_token_vectors(
        self,
        *workers: BaseWorker,
        crypto_provider: BaseWorker = None,
        requires_grad: bool = True,
        excluded_tokens: Dict[str, Set[object]] = None,
    ):
        """Get the Numpy array of all the vectors corresponding to the tokens in the `Doc`,
        excluding token according to the excluded_tokens dictionary.


        Args:
            workers (sequence of BaseWorker): A sequence of remote workers from .
            crypto_provider (BaseWorker): A remote worker responsible for providing cryptography
            (SMPC encryption) functionalities.
            requires_grad (bool): A boolean flag indicating whether gradients are required or not.
            excluded_tokens (Dict): A dictionary used to ignore tokens of the document based on values
                of their attributes, the keys are the attributes names and they index, for efficiency,
                sets of values.
                Example: {'attribute1_name' : {value1, value2}, 'attribute2_name': {v1, v2}, ....}

        Returns:
            Tensor: A SMPC-encrypted tensor representing the array of all vectors in the document
                this pointer points to.
        """

        # You need at least two workers in order to encrypt the vector with SMPC
        assert len(workers) > 1

        # Create the command
        kwargs = dict(
            crypto_provider=crypto_provider,
            requires_grad=requires_grad,
            excluded_tokens=excluded_tokens,
        )
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

    def set_indices(self, token_to_index: Dict):
        """Decrypts encrypted tokens using `self.owner`'s key and maps token to
        unique index.

        Args:
            token_to_index (dict): Contains encrypted tokens mapped to
                unique indices.
        """

        # Create the command
        command = ("set_indices", self.id_at_location, [token_to_index], {})

        # Send the command
        self.owner.send_command(self.location, command)
