from .language import Language
from syft.workers.base import BaseWorker
from typing import List, Tuple

def load(model_name,
         owner: BaseWorker,
         id: int = None,         
         tags: List[str] = None,
         description: str = None

         
):

        
    # Instantiate a Language object
    nlp = Language(model_name,
                   id = id,
                   owner = owner,
                   tags = tags,
                   description = description
    )

    return nlp
