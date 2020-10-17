from .pointers.pipeline_pointer import PipelinePointer
from ..utils import msgpack_code_generator
from ..utils import create_state_query
from ..utils import search_resource

from syft.generic.abstract.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.abstract.sendable import AbstractSendable
import syft.serde.msgpack.serde as serde

from typing import Union
from typing import Set
from typing import Tuple
from typing import Dict
from typing import List


class Pipeline(AbstractSendable):
    """This class is responsible of serving the template
    of a pipeline. This allows the pipeline to be reconstructed
    and distributed it over workers.
    """

    def __init__(
        self,
        name: str,
        template: List[dict],
        states_info: Dict[str, dict],
        owner: BaseWorker = None,
        tags: Set[str] = None,
        description: str = None,
    ):
        """Initializes the object.

        Args:
            name: The name of the pipeline. This name will
                be also used as the ID of the pipeline. This
                ID is used to search of the pipeline over
                PyGrid.
            template: A list of dictionaries each describing
                a pipe component of the pipeline in order.
            states_info: a dictionary of dictionaries containing the
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
                associated with the pipeline.
            description: A text that describes the pipeline,
                 its contents, and any other features.
        """

        self.name = name

        # Set the id to the same as the pipeline name
        id = name

        # Create properties
        self.template = template
        self.states_info = states_info

        # Initialize the parent class
        super(Pipeline, self).__init__(id=id, owner=owner, tags=tags, description=description)

    def deploy_states(self) -> None:
        """Search for the State objects associated with this pipeline and
        deploy them on the corresponding workers by sending copies of them.
        """

        # Loop through all the states
        for state_name in self.states_info:

            # Get the name of the state
            location_id = self.states_info[state_name]["location_id"]

            # Construct the state ID that will be used as the search query
            state_id = create_state_query(pipeline_name=self.name, state_name=state_name)

            # Search for the state
            result = search_resource(query=state_id, local_worker=self.owner)

            # If no state is found, pass
            if not result:
                continue

            # Send a copy of the state to the location to be deployed on
            # using its pointer
            state = result.send_copy(destination=location_id)

    def send_copy(self, destination: Union[str, BaseWorker]) -> "Pipeline":
        """This method is called by a PipelinePointer using
        PipelinePointer.get_copy(). It creates a copy of the current
        object and sends it to the pointer on `destination`
        which requested the copy.

        Args:
            location: The worker (or its id) on which the PipelinePointer object
                which requested the copy is located.
        """

        # Create the copy
        pipeline = Pipeline(
            name=self.name,
            template=self.template,
            states_info=self.states_info,
            owner=self.owner,
            tags=self.tags,
            description=self.description,
        )

        # Send the object
        self.owner.send_obj(pipeline, destination)

    def send(self, location: BaseWorker) -> PipelinePointer:
        """Sends this object to the worker specified by `location`.

        Args:
            location (BaseWorker): The BaseWorker object to which the Pipeline
                object is to be sent.

            Returns:
                (PipelinePointer): A pointer to this object.
        """

        pipeline_pointer = self.owner.send(self, location)

        return pipeline_pointer

    def create_pointer(
        pipeline: "Pipeline" = None,
        owner: BaseWorker = None,
        location: BaseWorker = None,
        id_at_location: str = None,
        tags: Set[str] = None,
        ptr_id=None,
        register: bool = True,
        garbage_collect_data: bool = False,
    ) -> PipelinePointer:
        """Creates a PipelinePointer object that points to a given
        Pipeline object.

        Args:
            pipeline (Pipeline): The Pipeline object
                to which the pointer refers.
                Although this is an instance method (As opposed to statics),
                this argument is called `pipeline` instead of `self` due
                to the fact that PySyft calls this method, sometimes on
                the class and sometimes on the object.
            owner (BaseWorker): The worker that will own the pointer object.
            location (BaseWorker): The worker on which the Pipeline
                object pointed to by this object is located.
            id_at_location (str, int): The ID of the Pipeline object
                referenced by the pointer.
            register (bool): Whether to register the pointer object
                in the object store or not. (it is required by the
                the BaseWorker's object send() method in PySyft, but
                not used for the moment in this method).
            garbage_collect_data (bool): Activate garbage collection or not.
                default to False meaning that the Pipeline object shouldn't
                be GCed once this pointer is removed.


        Returns:
            A PipelinePointer object pointing to this Pipeline object.
        """

        if location is None:
            location = pipeline.owner

        if id_at_location is None:
            id_at_location = pipeline.id

        # Create the pointer object
        pipeline_pointer = PipelinePointer(
            location=location, id_at_location=id_at_location, owner=owner
        )

        return pipeline_pointer

    @staticmethod
    def simplify(worker: BaseWorker, pipeline: "Pipeline") -> Tuple[object]:
        """Simplifies a Pipeline object. This method is required by PySyft
        when a Pipeline object is sent to another worker.

        Args:
            worker: The worker on which the simplify operation
                is carried out.
            pipeline: the Pipeline object to simplify.

        Returns:
            The simplified Pipeline object as a tuple of serialized Pipeline
            attributes.

        """

        # Simplify the Pipeline object attributes
        name_simple = serde._simplify(worker, pipeline.name)
        template_simple = serde._simplify(worker, pipeline.template)
        states_info_simple = serde._simplify(worker, pipeline.states_info)
        tags_simple = serde._simplify(worker, pipeline.tags)
        description_simple = serde._simplify(worker, pipeline.description)

        # create the simple Pipeline object
        pipeline_simple = (
            name_simple,
            template_simple,
            states_info_simple,
            tags_simple,
            description_simple,
        )

        return pipeline_simple

    @staticmethod
    def detail(worker: BaseWorker, pipeline_simple: Tuple[object]) -> "Pipeline":
        """Takes a simplified pipeline object, details it to create
        a new Pipeline object. This is usually done on a worker where
        the Pipeline object is sent.


        Args:
            worker (BaseWorker): The worker on which the
                detail operation is carried out.
            pipeline_simple: The simplified Pipeline object.

        Returns:
            A Pipeline object.
        """

        # Unpack the simple pipeline object
        (
            name_simple,
            template_simple,
            states_info_simple,
            tags_simple,
            description_simple,
        ) = pipeline_simple

        # Detail the attributes
        name = serde._detail(worker, name_simple)
        template = serde._detail(worker, template_simple)
        states_info = serde._detail(worker, states_info_simple)
        tags = serde._detail(worker, tags_simple)
        description = serde._detail(worker, description_simple)

        # Create a Pipeline object
        pipeline = Pipeline(
            name=name,
            template=template,
            states_info=states_info,
            owner=worker,
            tags=tags,
            description=description,
        )

        return pipeline

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
        if not hasattr(Pipeline, "proto_id"):
            Pipeline.proto_id = msgpack_code_generator(Pipeline.__qualname__)

        code_dict = dict(code=Pipeline.proto_id)

        return code_dict
