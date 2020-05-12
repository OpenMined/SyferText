from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker

from typing import Union


class StatePointer(ObjectPointer):
    """This class defines a pointer to a State object. Whenever a State
    object is searched on the grid and found on any of the remote workers,
    a pointer to it, represented by this object, is returned. Then using
    this object, a copy of the state can be downloaded to the worker that
    requested it.
    """

    def __init__(
        self,
        location: BaseWorker,
        id_at_location: str,
        owner: BaseWorker,
        id: Union[str, int],
        garbage_collect_data: bool = False,
    ):
        """Initializes the State object.

        Args:
            location (BaseWorker): The worker on which the State
                object pointed to by this object is located.
            id_at_location (str, int): The ID of the State object
                referenced by this pointer.
            owner (BaseWorker): The worker that owns this pointer object.
            id (str, int): The ID of the pointer object.
            garbage_collect_data (bool): Activate garbage collection or not. 
                default to False meaning that the State object shouldn't
                be GCed once this pointer is removed.
        """

        # Initialize the parent object
        super(StatePointer, self).__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=id,
            garbage_collect_data=garbage_collect_data,
        )

        
    def get_copy(self) -> "State":
        """This method is used to download a copy of the remote 
        State object.

        Returns:
            A State object that is an exact copy of the remote State
                object referenced by this pointer.
        """

        # Create the command message that is used to forward the method call.
        args = []
        kwargs = {"location": location}

        command = ("send_copy", self.id_at_location, args, kwargs)

        # Send the command
        state = self.owner.send_command(self.location, command)

        return state
