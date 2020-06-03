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
from typing import Set


def load(
    model_name, owner: BaseWorker, id: int = None, tags: Set[str] = None, description: str = None
):
    """Loads the specified language model `model_name` and returns a Language object.

    Args:
        model_name (str): The name of the language model.
        owner (BaseWorker): The worker that should own the Language object.
        id (int): The Identifier of the Language object in the worker's object registery.
        tags (set): A set of str that help search for the Language object across workers.
        description (str): A str that describes the Language object.


    Returns:
        a an object of the Language class, representing the requested language model.
    """

    # Instantiate a Language object
    nlp = Language(model_name,
                   id=id,
                   owner=owner,
                   tags=tags,
                   description=description)

    return nlp

def create(
    model_name, owner: BaseWorker, id: int = None, tags: Set[str] = None, description: str = None
):
    """Creates a new Language object. This function is used when a new language model
    is constructed from local files.


    Args:
        model_name (str): The name of the language model to create.
        owner (BaseWorker): The worker that should own the Language object.
        id (int): The Identifier of the Language object in the worker's object store.
        tags (set): A set of str that help search for the Language model across workers.
        description (str): A str that describes the Language object.


    Returns:
        a an object of the Language class, representing the created language model.
    """

    #TODO: The create method should first search over pygrid to make sure no other
    #      model has the same name

    # Instantiate a Language object
    nlp = Language(model_name = model_name,
                   id=id,
                   owner=owner,
                   tags=tags,
                   description=description)

    return nlp


# Set the default owners of some classes
SubPipeline.owner = LOCAL_WORKER
