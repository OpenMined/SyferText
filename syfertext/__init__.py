from .language import Language
from .tokenizer import Tokenizer
from .pointers.doc_pointer import DocPointer
from .pipeline import SubPipeline
from .pipeline import SimpleTagger

import syft
import syft.serde.msgpack.serde as serde
from syft.workers.base import BaseWorker
import torch

from typing import List, Tuple, Set
import logging
import os

# Get a torch hook
hook = syft.TorchHook(torch)

def load(
    model_name,
    owner: BaseWorker,
    id: int = None,
    tags: Set[str] = None,
    description: str = None,
):
    """Loads the specified language model `model_name` and returns a Language object. 

    If it is the first time the language model `model_name` is loaded, it is downloaded 
    prior to loading.

    Args:
        model_name (str): The name of the language model.
        owner (BaseWorker): The worker that should own the Language object.
        id (int): The Identifier of the Language object in the worker's object registery.
        tags (set): A set of str that help search for the Language object across workers.
        description (str): A str that describes the Language object.


    Returns:
        a an object of the Language class, representing the requested language model.
    """

    # If the language model does not exist, download it
    utils.get_lang_model(model_name)

    # Instantiate a Language object
    nlp = Language(model_name, id=id, owner=owner, tags=tags, description=description)

    return nlp


def register_to_serde(class_type: type):
    """Adds a class `class_type` to the `serde` module of PySyft.

    This is important to enable SyferText types to be sent to remote workers.

    Args:
        class_type (type): The class to register to PySyfts' serde module.
            This enables serde to serialize and deserialize objects of that class.

    Returns:
        (int): The proto ID it is registered with
    """

    # Get the maximum integer index of detailers and add 1 to it
    # to create a new index that does not exist yet
    proto_id = max(list(serde.detailers.keys())) + 1

    # Add the simplifier
    serde.detailers[proto_id] = class_type.detail

    # Add the simplifier
    serde.simplifiers[class_type] = (proto_id, class_type.simplify)


    return proto_id

# Register some types to serde
Tokenizer.proto_id = register_to_serde(Tokenizer)
SubPipeline.proto_id = register_to_serde(SubPipeline)
SimpleTagger.proto_id = register_to_serde(SimpleTagger)

# Set the default owners of some classes
SubPipeline.owner = hook.local_worker




