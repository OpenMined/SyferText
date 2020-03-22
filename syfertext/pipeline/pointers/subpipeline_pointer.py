from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker

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
            garbage_collect_data=garbage_collect_data,
        )
