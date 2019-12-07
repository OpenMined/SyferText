from .language import Language
from syft.workers.base import BaseWorker
import syft.serde as serde
from typing import List, Tuple
from .tokenizer import Tokenizer
from .pointers.doc_pointer import DocPointer
import logging


def load(
    model_name,
    owner: BaseWorker,
    id: int = None,
    tags: List[str] = None,
    description: str = None,
):

    # Instantiate a Language object
    nlp = Language(model_name, id=id, owner=owner, tags=tags, description=description)

    return nlp


def register_to_serde(serde, class_type):
    """
       Adds a Class 'class_type' to the 'serde' module of PySyft.
       This is important to enable SyferText types to be sent to 
       remote workers.
    """

    # Initialize an index in which the detailer method is to be
    # added in serde.detailers
    idx = None

    # In the following loop, an index is chosen
    # and the detailer is added in that index.
    # However, I test again if the detailers was correctly
    # added in that index to protect it from the case
    # where concurrent adds to serde.detailers from other libraries are
    # happening at the same time
    while idx is None or serde.detailers[idx] != class_type.detail:

        idx = len(serde.detailers)
        serde.detailers[idx] = class_type.detail

    # Now add the simplifier
    serde.simplifiers[class_type] = (idx, class_type.simplify)


# Register some types to serde
register_to_serde(serde, Tokenizer)
register_to_serde(serde, DocPointer)
