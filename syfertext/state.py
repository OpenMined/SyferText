from syft.generic.abstract.sendable import AbstractSendable
from syft.workers.base import BaseWorker
import syft.serde.msgpack.serde as serde

from .pointers import StatePointer
from .utils import msgpack_code_generator

from typing import Union
from typing import Set
from typing import Tuple
from typing import Dict


class State(AbstractSendable):
    
    def __init__(
        self,
        simple_obj: Tuple[object],
        id: str,
        access: Set[str],
        owner: BaseWorker = None,
        tags: Set[str] = None,
        description: str = None,
    ):
        """Initializes the object.

        Args:
            simple_obj: this is a tuple of simplified (serialized)
                objects that define the state of a SyferText object.

            id: The id of the state. This should be a string that
                uniquely identifies the state in terms of what 
                language model it belongs to, and what object it
                saves the state for.
                
                Example: "syfertext_en_core_web_lg:vocab" means that
                    this object saves the state of a Vocab object.

            access: The set of worker ids where this State can be
                sent. if the string '*' is included in the set,
                then all workers are allowed to receive a copy of the state.

            owner: The worker that owns this object. That is, the 
                syft worker on which this object is located.

            tags: Any set of other tags used to search for this state.
            description: Any extra information about this state.
        """

        self.simple_obj = simple_obj

        self.access = access

        super(State, self).__init__(id=id, owner=owner, tags=tags, description=description)

    def send_copy(self, destination: BaseWorker) -> "State":
        """This method is called by a StatePointer using 
        StatePointer.get_copy(). It creates a copy of the current
        object and send it to the pointer on `destination`
        which requested the copy.

        Args:
            location: The worker on which the StatePointer object
                which requested the copy is located.

        Returns:
            A copy of the current state object.
        """

        # Create the copy
        state = State(
            simple_obj=self.simple_obj,
            id=self.id,
            access=self.access,
            tags=self.tags,
            description=self.description,
        )

        # Send the object
        self.owner.send_obj(state, destination)


    def send(self, location: BaseWorker) -> StatePointer:
        """Sends this object to the worker specified by `location`. 

        Args:
            location (BaseWorker): The BaseWorker object to which the state object is 
                to be sent.

            Returns:
                (StatePointer): A pointer to this object.
        """

        assert (
            "*" in self.access or location.id in self.access
        ), f"Worker `{location.id}` does not have the right to download State with ID {self.id} on worker {self.owner}"

        state_pointer = self.owner.send(self, location)

        return state_pointer


    def create_pointer(
        state: "State" = None,
        owner: BaseWorker = None,
        location: BaseWorker = None,
        id_at_location: str = None,
        tags: Set[str] = None,
        register: bool = True,
        garbage_collect_data: bool = False,
    ) -> StatePointer:
        """Creates a SupPipelinePointer object that points to a given
        SupPipeline object.

        Args:
            state (State): The State object to which the pointer refers.
                Although this is an instance method (As opposed to statics),
                this argument is called `state` instead of `self` due
                to the fact that PySyft calls this method, sometimes of 
                the class and sometimes on the object. 
            owner (BaseWorker): The worker that will own the pointer object.
            location (BaseWorker): The worker on which the State
                object pointed to by this object is located.
            id_at_location (str, int): The ID of the State object
                referenced by the pointer.
            register (bool): Whether to register the pointer object 
                in the object store or not. (it is required by the 
                the BaseWorker's object send() method in PySyft, but
                not used for the moment in this method).
            garbage_collect_data (bool): Activate garbage collection or not. 
                default to False meaning that the State object shouldn't
                be GCed once this pointer is removed.

        
        Returns:
            A StatePointer object pointing to this state object.
        """
        
        if location is None:
            location = state.owner

        if id_at_location is None:
            id_at_location = state.id

        # Create the pointer object
        state_pointer = StatePointer(
            location=location,
            id_at_location=id_at_location,
            owner=owner,
        )

        return state_pointer


    @staticmethod
    def simplify(worker: BaseWorker, state: "State") -> Tuple[object]:
        """Simplifies a State object. This method is required by PySyft
        when a State object is sent to another worker. 

        Args:
            worker: The worker on which the simplify operation 
                is carried out.
            state: the State object to simplify.

        Returns:
            The simplified State object as a tuple of serialized State
            attributes.

        """

        # Simplify the State object attributes
        id_simple = serde._simplify(worker, state.id)
        access_simple = serde._simplify(worker, state.access)
        tags_simple = serde._simplify(worker, state.tags)
        description_simple = serde._simplify(worker, state.description)

        # create the simple State object
        state_simple = (id_simple, access_simple, tags_simple, description_simple, state.simple_obj)

        return state_simple

    @staticmethod
    def detail(worker: BaseWorker, state_simple: Tuple[object]) -> "State":
        """Takes a simplified State object, details it to create
        a new State object. This is usually done on a worker where
        the State object is sent.


        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            state_simple: The simplified State object.

        Returns:
            A State object.
        """

        # Unpack the simple state
        id_simple, access_simple, tags_simple, description_simple, simple_obj = state_simple

        # Detail the attributes
        id = serde._detail(worker, id_simple)
        access = serde._detail(worker, access_simple)
        tags = serde._detail(worker, tags_simple)
        description = serde._detail(worker, description_simple)

        # Create a State object
        state = State(
            simple_obj=simple_obj,
            id=id,
            access=access,
            owner=worker,
            tags=tags,
            description=description,
        )

        return state


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
        if not hasattr(State, "proto_id"):
            State.proto_id = msgpack_code_generator()

        code_dict = dict(code=State.proto_id)

        return code_dict
    
