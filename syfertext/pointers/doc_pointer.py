from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import syft as sy

from .span_pointer import SpanPointer
from typing import List
from typing import Union
from typing import Dict
from typing import Set


class DocPointer(ObjectPointer):
    """An Object Pointer that points to the Doc Object at remote location"""

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
        """Create a Doc Pointer from `location` where the `Doc` object resides and
        `id_at_location`, the id of the `Doc` object at that location.

        Args:
            location (BaseWorker): the worker where the `Doc` object resides that this
                DocPointer will point to.

            id_at_location (int or str): the id of the `Doc` object at the `location` worker.

            owner (BaseWorker): the owner of the this object ie. `DocPointer`

        Returns:
            A `DocPointer` object
        """

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

    def __getitem__(self, item: Union[slice, int]) -> SpanPointer:

        # if item is int, so we are trying to access to token
        assert isinstance(
            item, slice
        ), "You are not authorised to access a `Token` from a `DocPointer`"

        # Create the command
        command = ("__getitem__", self.id_at_location, [item], {})

        # Send the command
        obj_id = self.owner.send_command(self.location, command)

        # we create a SpanPointer from the obj_id
        span = SpanPointer(location=self.location, id_at_location=obj_id, owner=self.owner)

        return span

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

    def set_attribute(self, name: str, value: object):
        """Creates a custom attribute with the name `name` and
        value `value` in the Underscore object of remote `Doc`.

        Args:
            name (str): name of the custom attribute.
            value (object): value of the custom named attribute.
        """

        # Create the command
        command = ("set_attribute", self.id_at_location, [name, value], {})

        # Send the command
        self.owner.send_command(self.location, command)

    def has_attribute(self, name: str) -> bool:
        """Checks if remote `Doc` has custom attribute with name `name`.

        Args:
            name (str): name of the custom attribute.

        Returns:
            attr_exists (bool): Returns `True` if the Underscore object of remote `Doc`
            has an attribute `name`, otherwise returns `False`
        """

        # Create the command
        command = ("has_attribute", self.id_at_location, [name], {})

        # Send the command
        attr_exists = self.owner.send_command(self.location, command)

        return attr_exists

    def remove_attribute(self, name: str):
        """Removes the custom attribute `name` from the remote `Doc`.

        Args:
            name (str): name of the custom attribute.
        """

        # Create the command
        command = ("remove_attribute", self.id_at_location, [name], {})

        # Send the command
        self.owner.send_command(self.location, command)

    def get_attribute(self, name: str):
        """Returns value of custom attribute or remote `Doc` with name `name` if it is present,
         else raises `AttributeError`.

        Args:
            name (str): name of the custom attribute.

        Returns:
            value (obj): value of the custom attribute of remote `Doc` with name `name`.
        """

        # Create the command
        command = ("get_attribute", self.id_at_location, [name], {})

        # Send the command
        value = self.owner.send_command(self.location, command)

        return value
