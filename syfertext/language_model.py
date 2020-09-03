from .pointers.language_model_pointer import LanguageModelPointer
from .utils import msgpack_code_generator
from .utils import create_state_query
from .utils import search_resource

from syft.generic.abstract.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.abstract.sendable import AbstractSendable
import syft.serde.msgpack.serde as serde

from typing import Union
from typing import Set
from typing import Tuple
from typing import Dict
from typing import List


class LanguageModel(AbstractSendable):
    """This class is responsible of serving the pipeline template
    of a language model that allows the Language object to recreate
    the pipeline and distribute it over workers.
    """

    def __init__(
        self,
        name: str,
        pipeline_template: List[dict],
        states: Dict[str, dict],
        owner: BaseWorker = None,
        tags: Set[str] = None,
        description: str = None,
    ):
        """Initializes the object.

        Args:
            name: The name of the language model. This name will
                be also used as the ID of the language model. This
                ID is used to search of the language model over
                PyGrid.
            pipeline_template: A list of dictionaries each describing
                a pipe component of the pipeline in order.
            states: a dictionary of dictionaries containing the
                description of each state needed to reconstruct
                the pipeline. Example:
                    {'tokenizer': {'location_id': 'bob',
                                   'access': {'*'},
                                  },
                     'vocab': {'location_id': 'bob',
                               'access', {'*'},
                              }
                     :
                     :
                    }
            owner: The worker that owns this object. That is, the
                syft worker on which this object is located.
            tags: A set of PyGrid searchable tags that can be
                associated with the language model.
            description: A text that describes the language model,
                 its contents, and any other features.
        """

        self.name = name

        # Set the id to the same as the language model name
        id = name

        # Create properties
        self.pipeline_template = pipeline_template
        self.states = states

        # Initialize the parent class
        super(LanguageModel, self).__init__(id=id, owner=owner, tags=tags, description=description)

    def deploy_states(self) -> None:
        """Search for the State objects associated with this language model and
        deploy them on the corresponding workers by sending copies of them.
        """

        # Loop through all the states
        for state_name in self.states:

            # Get the name of the state
            location_id = self.states[state_name]["location_id"]

            # Construct the state ID that will be used as the search query
            state_id = create_state_query(model_name=self.name, state_name=state_name)

            # Search for the state
            result = search_resource(query=state_id, local_worker=self.owner)

            # If no state is found, pass
            if not result:
                continue

            # Send a copy of the state to the location to be deployed on
            # using its pointer
            state = result.send_copy(destination=location_id)

    def send_copy(self, destination: Union[str, BaseWorker]) -> "LanguageModel":
        """This method is called by a LanguageModelPointer using
        LanguageModelPointer.get_copy(). It creates a copy of the current
        object and sends it to the pointer on `destination`
        which requested the copy.

        Args:
            location: The worker (or its id) on which the LanguageModelPointer object
                which requested the copy is located.
        """

        # Create the copy
        language_model = LanguageModel(
            name=self.name,
            pipeline_template=self.pipeline_template,
            states=self.states,
            owner=self.owner,
            tags=self.tags,
            description=self.description,
        )

        # Send the object
        self.owner.send_obj(language_model, destination)

    def send(self, location: BaseWorker) -> LanguageModelPointer:
        """Sends this object to the worker specified by `location`.

        Args:
            location (BaseWorker): The BaseWorker object to which the LanguageModel
                object is to be sent.

            Returns:
                (LanguageModelPointer): A pointer to this object.
        """

        language_model_pointer = self.owner.send(self, location)

        return language_model_pointer

    def create_pointer(
        language_model: "LanguageModel" = None,
        owner: BaseWorker = None,
        location: BaseWorker = None,
        id_at_location: str = None,
        tags: Set[str] = None,
        ptr_id=None,
        register: bool = True,
        garbage_collect_data: bool = False,
    ) -> LanguageModelPointer:
        """Creates a LanguageModelPointer object that points to a given
        LanguageModel object.

        Args:
            language_model (LanguageModel): The LanguageModel object
                to which the pointer refers.
                Although this is an instance method (As opposed to statics),
                this argument is called `language_model` instead of `self` due
                to the fact that PySyft calls this method, sometimes on
                the class and sometimes on the object.
            owner (BaseWorker): The worker that will own the pointer object.
            location (BaseWorker): The worker on which the LanguageModel
                object pointed to by this object is located.
            id_at_location (str, int): The ID of the LanguageModel object
                referenced by the pointer.
            register (bool): Whether to register the pointer object
                in the object store or not. (it is required by the
                the BaseWorker's object send() method in PySyft, but
                not used for the moment in this method).
            garbage_collect_data (bool): Activate garbage collection or not.
                default to False meaning that the LanguageModel object shouldn't
                be GCed once this pointer is removed.


        Returns:
            A LanguageModelPointer object pointing to this LanguageModel object.
        """

        if location is None:
            location = language_model.owner

        if id_at_location is None:
            id_at_location = language_model.id

        # Create the pointer object
        language_model_pointer = LanguageModelPointer(
            location=location, id_at_location=id_at_location, owner=owner,
        )

        return language_model_pointer

    @staticmethod
    def simplify(worker: BaseWorker, language_model: "LanguageModel") -> Tuple[object]:
        """Simplifies a LanguageModel object. This method is required by PySyft
        when a LanguageModel object is sent to another worker.

        Args:
            worker: The worker on which the simplify operation
                is carried out.
            language_model: the LanguageModel object to simplify.

        Returns:
            The simplified LanguageModel object as a tuple of serialized LanguageModel
            attributes.

        """

        # Simplify the LanguageModel object attributes
        name_simple = serde._simplify(worker, language_model.name)
        pipeline_template_simple = serde._simplify(worker, language_model.pipeline_template)
        states_simple = serde._simplify(worker, language_model.states)
        tags_simple = serde._simplify(worker, language_model.tags)
        description_simple = serde._simplify(worker, language_model.description)

        # create the simple LanguageModel object
        language_model_simple = (
            name_simple,
            pipeline_template_simple,
            states_simple,
            tags_simple,
            description_simple,
        )

        return language_model_simple

    @staticmethod
    def detail(worker: BaseWorker, language_model_simple: Tuple[object]) -> "LanguageModel":
        """Takes a simplified language_model object, details it to create
        a new LanguageModel object. This is usually done on a worker where
        the LanguageModel object is sent.


        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            language_model_simple: The simplified LanguageModel object.

        Returns:
            A LanguageModel object.
        """

        # Unpack the simple language model object
        (
            name_simple,
            pipeline_template_simple,
            states_simple,
            tags_simple,
            description_simple,
        ) = language_model_simple

        # Detail the attributes
        name = serde._detail(worker, name_simple)
        pipeline_template = serde._detail(worker, pipeline_template_simple)
        states = serde._detail(worker, states_simple)
        tags = serde._detail(worker, tags_simple)
        description = serde._detail(worker, description_simple)

        # Create a LanguageModel object
        language_model = LanguageModel(
            name=name,
            pipeline_template=pipeline_template,
            states=states,
            owner=worker,
            tags=tags,
            description=description,
        )

        return language_model

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
        if not hasattr(LanguageModel, "proto_id"):
            LanguageModel.proto_id = msgpack_code_generator()

        code_dict = dict(code=LanguageModel.proto_id)

        return code_dict
