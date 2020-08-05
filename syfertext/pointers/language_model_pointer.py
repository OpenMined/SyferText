from syft.generic.pointers.object_pointer import ObjectPointer
from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde

from ..utils import search_resource
from ..utils import msgpack_code_generator

from typing import Union
from typing import Tuple
from typing import Dict


class LanguageModelPointer(ObjectPointer):
    """This class defines a pointer to a LanguageModel object. Whenever a 
    LanguageModel object is searched on the grid and found on any of the 
    remote workers, a pointer to it, represented by this object, is returned. 
    Then, using this pointer, a copy of the LanguageModel object could be 
    pulled to the worker that requested it.
    """

    def __init__(
        self, location: BaseWorker, id_at_location: str, owner: BaseWorker,
    ):
        """Initializes the LanguageModel object.

        Args:
            location (BaseWorker): The worker on which the LanguageModel
                object pointed to by this object is located.
            id_at_location (str, int): The ID of the LanguageModel object
                referenced by this pointer.
            owner (BaseWorker): The worker that owns this pointer object.
            id (str, int): The ID of the pointer object.
        """

        # Initialize the parent object
        super(LanguageModelPointer, self).__init__(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
            id=None,
            garbage_collect_data=False,
        )

    def get_copy(self) -> "LanguageModel":
        """This method is used to download a copy of the remote 
        LanguageModel object.

        Returns:
            A LanguageModel object that is an exact copy of the remote LanguageModel
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

        # Get the language_model from the local object store
        language_model = self.owner.get_obj(self.id_at_location)

        return language_model

    def deploy_states(self) -> None:
        """Forwards the call to the `deploy_states` method of the underlying
        `language_model` object which, in turn, searches for the State objects 
        associated with this language model and deploys them on the 
        corresponding workers by sending copies of them.
        """

        # Send the command
        self.owner.send_command(
            recipient=self.location,
            cmd_name="deploy_states",
            target=self,
            args_=tuple(),
            kwargs_={},
        )

    @staticmethod
    def simplify(
        worker: BaseWorker, language_model_pointer: "LanguageModelPointer"
    ) -> Tuple[object]:
        """Simplifies a LanguageModelPointer object. This method is required by PySyft
        when a LanguageModelPointer object is sent to another worker. 

        Args:
            worker: The worker on which the simplify operation is carried out.
            language_model_pointer: the LanguageModelPointer object to simplify.

        Returns:
            The simplified LanguageModelPointer object as a tuple of serialized LanguageModel
            attributes.

        """

        # Simplify the LanguageModelPointer object attributes
        location_simple = serde._simplify(worker, language_model_pointer.location)
        id_at_location_simple = serde._simplify(worker, language_model_pointer.id_at_location)

        # create the simple LanguageModelPointer object
        language_model_pointer_simple = (location_simple, id_at_location_simple)

        return language_model_pointer_simple

    @staticmethod
    def detail(
        worker: BaseWorker, language_model_pointer_simple: Tuple[object]
    ) -> "LanguageModelPointer":
        """Takes a simplified LanguageModelPointer object, details it to create
        a new LanguageModelPointer object. This is usually done on a worker where
        the LanguageModelPointer object is sent.


        Args:
            worker (BaseWorker): The worker on which the detail operation is carried out.
            language_model_pointer_simple: The simplified LanguageModelPointer object.

        Returns:
            A LanguageModelPointer object.
        """

        # Unpack the simple language_model
        location_simple, id_at_location_simple = language_model_pointer_simple

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

        # Create a LanguageModel object
        language_model_pointer = LanguageModelPointer(
            location=location, id_at_location=id_at_location, owner=worker
        )

        return language_model_pointer

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
        if not hasattr(LanguageModelPointer, "proto_id"):
            LanguageModelPointer.proto_id = msgpack_code_generator()

        code_dict = dict(code=LanguageModelPointer.proto_id)

        return code_dict
