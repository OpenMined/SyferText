from .language import Language
from .pipeline import SubPipeline

import syft

from syft.workers.base import BaseWorker
import torch

from typing import Set


# Get a torch hook
hook = syft.TorchHook(torch)


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
    nlp = Language(model_name, id=id, owner=owner, tags=tags, description=description)

    return nlp


# Set the default owners of some classes
SubPipeline.owner = hook.local_worker
