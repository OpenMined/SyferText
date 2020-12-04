import syft
from syft.serde.msgpack.serde import msgpack_global_state
from syft.workers.base import BaseWorker
import torch
import dill
import os
from pathlib import Path

# Get a torch hook
HOOK = syft.TorchHook(torch)

# Set the local worker
LOCAL_WORKER = HOOK.local_worker

from .language import Language
from .pipeline import SubPipeline
from .pipeline import Pipeline
from .pipeline.pointers.pipeline_pointer import PipelinePointer
from .pointers.state_pointer import StatePointer
from .utils import search_resource, create_state_query

from typing import Set
from typing import Union


def load(pipeline_name: str) -> Language:
    """Searches for `pipeline_name` on PyGrid and loads it as a Language object.

    Args:
        pipeline_name (str): The name of the pipeline to search for  on PyGrid.

    Returns:
        a an object of the Language class, representing the requested pipeline.
    """

    # Search for the pipeline
    result = search_resource(query=pipeline_name, local_worker=LOCAL_WORKER)

    pipeline = None

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

        # Save the pipeline to local storage
        save(pipeline_name=pipeline_name, pipeline=pipeline, destination="local")

    elif isinstance(result, Pipeline):

        # In this case, the Pipeline object is found on the local worker
        # which is a virtual worker by default as of the current PySyft version
        # 0.2.9. We do not consider that it is officially deployed.
        deployed_on = None

        # Get the pipeline object
        pipeline = result

    elif isinstance(result, tuple):

        # In this case we get a simplified pipeline object,
        # from the stored cache which is a tuple.
        # The following code details it back to a pipeline object.

        pipeline = Pipeline.detail(worker=LOCAL_WORKER, pipeline_simple=result)

        # Since we found the model in local storage, deployed_on = None

        deployed_on = None

    # Instantiate a Language object
    nlp = Language(
        pipeline_name, owner=LOCAL_WORKER, tags=pipeline.tags, description=pipeline.description
    )

    # Set the `deployed_on` property
    nlp.deployed_on = deployed_on

    # Load the pipeline into the Language object
    nlp.load_pipeline(template=pipeline.template, states_info=pipeline.states_info)

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
# SubPipeline.owner = LOCAL_WORKER


def save(pipeline_name: str, pipeline: "Pipeline", destination: Union["local"] = "local") -> None:
    """Saves the pipeline and it's states to storage

    Args:
        pipeline_name (str): The name of the pipeline.
        pipeline (Pipeline): The pipeline object itself
        destination (Union): The location where to save that object, currently storage on local machine is implemented.

    """

    # Path to the home/SyferText/cache/<pipeline_name> directory
    data_path = os.path.join(str(Path.home()), "SyferText", "cache", pipeline_name)

    # Creating a new directory if home/SyferText/cache/<pipeline_name> does not exist
    if not os.path.exists(data_path):
        os.makedirs(data_path, exist_ok=True)

    # Making a target at the file
    target = str("/{}.pkl".format(pipeline_name))

    # Opening cache file
    pipeline_cache = open(data_path + target, "wb")

    # Dumping data
    dill.dump(pipeline.simplify(worker=LOCAL_WORKER, pipeline=pipeline), pipeline_cache)

    # Loading states
    for state in pipeline.states_info:
        # Searching on state
        state_id = create_state_query(pipeline_name=pipeline_name, state_name=state)

        result = search_resource(query=state_id, local_worker=LOCAL_WORKER)

        if isinstance(result, StatePointer):
            # State target

            # Since : is a reserved character for naming files, replacing with - instead
            # Splitting tokens to get the name of the pipeline for the name of the pipeline directory
            # tokens = state_id.split(":")
            # print("Tokens", tokens)
            file_name = state_id.replace(":", "-")

            target = str("/{}.pkl".format(file_name))

            state_obj = result.get_copy()

            state_cache = open(data_path + target, "wb")

            dill.dump(state_obj.simplify(worker=LOCAL_WORKER, state=state_obj), state_cache)

    pipeline_cache.close()
