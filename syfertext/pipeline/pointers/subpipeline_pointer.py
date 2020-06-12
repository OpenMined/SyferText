from syft.generic.pointers.object_pointer import ObjectPointer
from syft.generic.pointers.string_pointer import StringPointer
from syft.workers.base import BaseWorker

from ...pointers.doc_pointer import DocPointer

from typing import Union, Dict, List


class SubPipelinePointer(ObjectPointer):
    """Use this class to create a pointer to a subpipeline object.
    Such pointers are used to send commands to execute different
    methods of remote subpipeline object.
    """

    def __init__(
        self,
        location: BaseWorker,
        id_at_location: Union[str, int],
        owner: BaseWorker,
        id: Union[str, int],
        garbage_collect_data: bool = True,
    ):
        """Initializes the object.

        Args:
            location (BaseWorker): The worker on which the SubPipeline
                object pointed to by this object is located.
            id_at_location (str, int): The PySyft ID of the SubPipeline
                object referenced by this pointer.
            owner (BaseWorker): The worker that owns this pointer
                object.
            id (str, int): The ID of the pointer object.
            garbage_collect_data (bool): Activate garbage collection or not.
        """

        # Initialize the parent object
        super(SubPipelinePointer, self).__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=id,
            garbage_collect_data=True,  # Overriding passed argument
        )

    def __call__(self, pointer: Union[StringPointer, DocPointer]):
        """Forwards the call to the `__call__` method of the
        `SubPipeline` object it points to.
        This forwarding mecanism is needed when the SubPipeline is
        located on a remote worker.

        Args:
            pointer: A pointer to the PySyft `String` to be tokenized or
                to the `Doc` object to by modified.
        """

        # Make sure that the String of Doc to process is located on the
        # same worker as the SubPipeline object.
        assert (
            pointer.location == self.location
        ), "The `String` or `Doc`  objects to process do not belong to the same worker"

        # Get the ID of the remote object pointed to by `pointer`.
        input_id_at_location = pointer.id_at_location

        # Create the command message to is used to forward the method
        # call.
        args = tuple()
        kwargs = {"input_id": input_id_at_location}

        # Send the command
        response = self.owner.send_command(
            recipient=self.location, cmd_name="__call__", target=self, args_=args, kwargs_=kwargs
        )

        return response
