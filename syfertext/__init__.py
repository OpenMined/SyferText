import syft
from syft.serde.msgpack.serde import msgpack_global_state
from syft.workers.base import BaseWorker
import torch

# Get a torch hook
HOOK = syft.TorchHook(torch)

# Set the local worker
LOCAL_WORKER = HOOK.local_worker

from .language import Language
from .language_model import LanguageModel
from .pointers import LanguageModelPointer
from .pipeline import SubPipeline
from .utils import search_resource
from .typecheck.typecheck import type_hints

from typing import Set

@type_hints
def load(model_name: str) -> Language:
    """Searches for `model_name` on PyGrid and loads it as a Language object.

    Args:
        model_name (str): The name of the language model to search for  on PyGrid.

    Returns:
        a an object of the Language class, representing the requested language model.
    """

    # Search for the language model
    result = search_resource(query=model_name, local_worker = LOCAL_WORKER)

    # If no language model is found, return
    if not result:
        return

    # If a language model is found get either its pointer if it is remote
    # or the language model itself if it is local
    elif isinstance(result, LanguageModelPointer):
        
        # Get a copy of the language model using its pointer
        language_model = result.get_copy()

    elif isinstance(result, LanguageModel):
        language_model = result
    
    # Instantiate a Language object
    nlp = Language(model_name,
                   owner=LOCAL_WORKER,
                   tags=language_model.tags,
                   description=language_model.description)

    # Load the pipeline into the Language object
    nlp.load_pipeline(pipeline_template = language_model.pipeline_template,
                      states = language_model.states)
    
    return nlp

@type_hints
def create(
    model_name, tags: Set[str] = None, description: str = None
) -> Language:
    """Creates a new Language object. This function is used when a new language model
    is constructed from local files.


    Args:
        model_name (str): The name of the language model to create.
        tags (set): A set of str that help search for the Language model across workers.
        description (str): A str that describes the Language object.


    Returns:
        a an object of the Language class, representing the created language model.
    """


    #TODO: The create method should first search over pygrid to make sure no other
    #      model has the same name

    # Instantiate a Language object
    nlp = Language(model_name = model_name,
                   owner=LOCAL_WORKER,
                   tags=tags,
                   description=description)

    return nlp


# Set the default owners of some classes
SubPipeline.owner = LOCAL_WORKER
