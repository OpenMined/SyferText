from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde

from ..utils import msgpack_code_generator

from typing import Union
from typing import Tuple
from typing import Dict


class StatePointer(ObjectPointer):
    """This class defines a pointer to a State object. Whenever a State
    object is searched on the grid and found on any of the remote workers,
    a pointer to it, represented by this object, is returned. Then using
    this object, a copy of the state can be downloaded to the worker that
    requested it.
    """

    def __init__(self, location: BaseWorker, id_at_location: str, owner: BaseWorker):
        """Initializes the State object.

        Args:
            location (BaseWorker): The worker on which the State
                object pointed to by this object is located.
            id_at_location (str, int): The ID of the State object
                referenced by this pointer.
            owner (BaseWorker): The worker that owns this pointer object.
            id (str, int): The ID of the pointer object.
        """

        # Initialize the parent object
        super(StatePointer, self).__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=None,
            garbage_collect_data=False,
        )

    def get_copy(self) -> "State":
        """This method is used to download a copy of the remote
        State object.

        Returns:
            A State object that is an exact copy of the remote State
                object referenced by this pointer.
        """

        # Send the command
        self.owner.send_command(
            recipient=self.location,
            cmd_name="send_copy",
            target=self,
            args_=tuple(),
            kwargs_={"destination": self.owner},
        )

        # Get the state from the local object store
        state = self.owner.get_obj(self.id_at_location)

        return state

    def send_copy(self, destination: Union[str, BaseWorker]) -> None:
        """Calls the `send_copy` method of the underlying State object."""

        # Send the command
        self.owner.send_command(
            recipient=self.location,
            cmd_name="send_copy",
            target=self,
            args_=tuple(),
            kwargs_={"destination": destination},
        )

    @staticmethod
    def simplify(worker: BaseWorker, state_pointer: "StatePointer") -> Tuple[object]:
        """Simplifies a StatePointer object. This method is required by PySyft
        when a StatePointer object is sent to another worker.

        Args:
            worker: The worker on which the simplify operation
                is carried out.
            state_pointer: the StatePointer object to simplify.

        Returns:
            The simplified StatePointer object as a tuple of serialized State
            attributes.

        """

        # Simplify the StatePointer object attributes
        location_simple = serde._simplify(worker, state_pointer.location)
        id_at_location_simple = serde._simplify(worker, state_pointer.id_at_location)

        # create the simple StatePointer object
        state_pointer_simple = (location_simple, id_at_location_simple)

        return state_pointer_simple

    @staticmethod
    def detail(worker: BaseWorker, state_pointer_simple: Tuple[object]) -> "StatePointer":
        """Takes a simplified StatePointer object, details it to create
        a new StatePointer object. This is usually done on a worker where
        the StatePointer object is sent.


        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            state_pointer_simple: The simplified StatePointer object.

        Returns:
            A StatePointer object.
        """

        # Unpack the simple state
        location_simple, id_at_location_simple = state_pointer_simple

        # Detail the attributes
        location = serde._detail(worker, location_simple)
        id_at_location = serde._detail(worker, id_at_location_simple)

        # If the detailing is happening on the worker pointed
        # to by this pointer, there is no point in keeping a
        # pointer. We extract the object pointed to instead.
        # This is actually crucial to the correct functioning
        # of CommandMessages in PySyft.
        if worker.id == location.id:
            return worker.get_obj(id_at_location)

        # Create a State object
        state_pointer = StatePointer(location=location, id_at_location=id_at_location, owner=worker)

        return state_pointer

    @staticmethod
    def get_msgpack_code() -> Dict[str, int]:
        """This is the implementation of the `get_msgpack_code()`
        method required by PySyft's SyftSerializable class.
        It provides a code for msgpack if the type is not present in proto.json.

        The returned object should be similar to:
        {
            "code": int value,
            "forced_code": int value
        }

        Both keys are optional, the common and right way would be to add only the "code" key.

        Returns:
            dict: A dict with the "code" and/or "forced_code" keys.
        """

        # If a msgpack code is not already generated, then generate one
        if not hasattr(StatePointer, "proto_id"):
            StatePointer.proto_id = msgpack_code_generator()

        code_dict = dict(code=StatePointer.proto_id)

        return code_dict
