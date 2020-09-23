import syft
from syft.serde.msgpack.serde import msgpack_global_state
from syft.workers.base import BaseWorker
import torch

# Get a torch hook
HOOK = syft.TorchHook(torch)

# Set the local worker
LOCAL_WORKER = HOOK.local_worker

from .language import Language
from .pipeline import SubPipeline
from .pipeline import Pipeline
from .pipeline.pointers.pipeline_pointer import PipelinePointer
from .utils import search_resource

from typing import Set


def load(pipeline_name: str) -> Language:
    """Searches for `pipeline_name` on PyGrid and loads it as a Language object.

    Args:
        pipeline_name (str): The name of the pipeline to search for  on PyGrid.

    Returns:
        a an object of the Language class, representing the requested pipeline.
    """

    # Search for the pipeline
    result = search_resource(query=pipeline_name, local_worker=LOCAL_WORKER)

    # If no pipeline is found, return
    if not result:
        return

    # If a pipeline is found get either its pointer if it is remote
    # or the pipeline itself if it is local
    elif isinstance(result, PipelinePointer):

        # The ID of the worker on which the pipeline is deployed
        deployed_on = result.location.id

        # Get a copy of the pipeline using its pointer
        pipeline = result.get_copy()

    elif isinstance(result, Pipeline):

        # In this case, the Pipeline object is found on the local worker
        # which is a virtual worker by default as of the current PySyft version
        # 0.2.9. We do not consider that it is officially deployed.
        deployed_on = None

        # Get the pipeline object
        pipeline = result

    # Instantiate a Language object
    nlp = Language(
        pipeline_name, owner=LOCAL_WORKER, tags=pipeline.tags, description=pipeline.description
    )

    # Set the `deployed_on` property
    nlp.deployed_on = deployed_on

    # Load the pipeline into the Language object
    nlp.load_pipeline(template=pipeline.template, states=pipeline.states)

    return nlp


def create(pipeline_name, tags: Set[str] = None, description: str = None):
    """Creates a new Language object. This function is used when a new pipeline
    is constructed from local files.


    Args:
        pipeline_name (str): The name of the pipeline to create.
        tags (set): A set of str that help search for the pipeline across workers.
        description (str): A str that describes the Language object.


    Returns:
        a an object of the Language class, representing the created pipeline.
    """

    # TODO: The create method should first search over pygrid to make sure no other
    #      model has the same name

    # Instantiate a Language object
    nlp = Language(
        pipeline_name=pipeline_name, owner=LOCAL_WORKER, tags=tags, description=description
    )

    return nlp


# Set the default owners of some classes
SubPipeline.owner = LOCAL_WORKER
