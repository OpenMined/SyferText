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
        """
           Get the mean of the vectors of each Token in this documents.

           Parameters
           ----------
            self: DocPointer
                pointer to a remote document.
            workers: sequence of BaseWorker
                A sequence of remote workers from .
            crypto_provider: BaseWorker
                A remote worker responsible for providing cryptography (SMPC encryption) functionalities.
            requires_grad:
                A boolean flag indicating whether gradients are required or not.

           Returns
           -------
            encrypted_vector: Tensor
                A tensor representing the SMPC-encrypted vector of the Doc this pointer points to.
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

    def __len__(self):

        # Create the command
        command = ("__len__", self.id_at_location, [], {})

        # Send the command
        length = self.owner.send_command(self.location, command)

        return length

    @staticmethod
    def simplify(worker, doc_pointer):
        """
           This method is used to reduce a `DocPointer` object into a list of simpler objects that can be
           serialized
        """

        # Simplify the attributes
        location_id = pickle.dumps(doc_pointer.location.id)
        tags = (
            [pickle.dumps(tag) for tag in doc_pointer.tags]
            if doc_pointer.tags
            else None
        )
        description = pickle.dumps(doc_pointer.description)

        return (
            location_id,
            doc_pointer.id_at_location,
            doc_pointer.id,
            doc_pointer.garbage_collect_data,
            tags,
            description,
        )

    @staticmethod
    def detail(worker: BaseWorker, simple_obj):
        """
           Create an object of type DocPointer from the reduced representation in `simple_obj`.

           Parameters
           ----------
           worker: BaseWorker
                   The worker on which the new DocPointer object is to be created.
           simple_obj: tuple
                       A tuple resulting from the serialized then deserialized returned tuple
                       from the `_simplify` static method above.

           Returns
           -------
           doc_pointer: DocPointer
                      a DocPointer object, pointing to a Doc object
        """

        # Get the typle elements
        (
            location_id,
            id_at_location,
            id,
            garbage_collect_data,
            tags,
            description,
        ) = simple_obj

        # Unpickle
        location_id = pickle.loads(location_id)
        tags = [pickle.loads(tag) for tag in tags] if tags else None
        description = pickle.loads(description)

        # Get the worker `location` on which lives the pointed-to Doc object
        location = worker.get_worker(id_or_worker=location_id)

        # Create a DocPointer object
        doc_pointer = DocPointer(
            location=location,
            id_at_location=id_at_location,
            owner=worker,
            id=id,
            garbage_collect_data=garbage_collect_data,
            tags=tags,
            description=description,
        )

        return doc_pointer
