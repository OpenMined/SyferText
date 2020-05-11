from syft.generic.object import AbstractObject
from syft.worker.base import BaseWorker

import syft.serde.msgpack.serde as serde

from typing import Union
from typing import Set
from typing import Tuple

class State(AbstractObject):


    def __init__(self,
                 simple_obj: Tuple(object),                 
                 id: str,
                 owner: BaseWorker = None,
                 tags: set[str] = None,
                 description: str = None
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

            owner: The worker that owns this object. That is, the 
                syft worker on which this object is located.

            tags: Any set of other tags used to search for this state.
            description: Any extra information about this state.
        """

        self.simple_obj = simple_obj

        super(State, self).__init__(id = id, owner = owner, tags = tags, description = description)

        
    def send_copy(self, location: BaseWorker) -> "State":
        """This method is called by a StatePointer using 
        StatePointer.get_copy(). It creates a copy of the current
        object and send it to the pointer on `location`
        which requested the copy.

        Args:
            location: The worker on which the StatePointer object
                which requested the copy is located.

        Returns:
            A copy of the current state object.
        """

        # Create the copy
        state = State(simple_obj = self.simple_obj,
                      id = self.id,
                      tags = self.tags,
                      description = self.description
        )

        return state

    
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
        tags_simple = serde._simplify(worker, state.tags)
        description_simple = serde._simplify(worker, state.description)        

        # create the simple State object
        state_simple = (id_simple, tags_simple, description_simple, state.simple_obj)
        
        return state_simple


    @staticmethod
    def detail(worker: Baseworker, state_simple: Tuple[object]) -> "State":
        """Takes a simplified State object, details it to create
        a new State object. This is usually done on a worker where
        the State object is sent.


        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.

        Returns:
            A State object.
        """

        # Unpack the simple state
        id_simple, tags_simple, description_simple, simple_obj = state_simple
        # Detail the attributes
        id = serde._detail(id_simple)
        tags = serde._detail(tags_simple)
        description = serde._detail(description_simple)

        # Create a State object
        state = State(simple_obj = simple_obj,
                      id = id,
                      owner  = worker,
                      tags = tags,
                      description = description
        )

        return state
        
        
