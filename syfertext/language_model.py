from syft.generic.object import AbstractObject
from syft.worker.base import BaseWorker

class LanguageModel(AbstractObject):
    """This class is responsible of serving the resources and 
    the configurations that help create a Language object or
    part of it.

    A language model typically consists of a pipeline template
    that describes all of the pipe components and resource files
    such as vectors and vocabulary stored as mapping between 
    hash keys and vector indexes. Those resources and configuration
    data for a given language models are persisted on a PyGrid
    node. An object of this class is sent to live on the object
    store of the worker where the corresponding language model 
    is located. This object is also tagged with the name
    of the language model and the name of its resources and
    contents. Check the documentation of the `__init__` method
    to know more.

    """
    
    def __init__(self,
                 id: Union[str, int] = None,
                 owner: BaseWorker = None,
                 tags: set[str] = None,
                 description: str = None,
    ):
        """Initializes the object.

        Args:
            id: The id of the object. Defaults to `None`. If it is
                `None`, then it will be assigned an automatically
                generated ID value in the parent class.

            owner: The worker that owns this object. That is, the 
                syft worker on which this object is located.

            tags: This is a very important argument. It will hold
                the tags that will define the name of the language
                model served by this object. It will also hold
                the names of the individual object and  pipe components 
                included in this model. For instance, if the language model's
                name is 'syfertext_en_core_web_lg' and if it contains
                a 'vectors' array that stores the embedding vector for
                each word in the vocab, and an array called 'key2row'
                that maps each word hash in the vocab to a row index
                in the 'vectors' array, and  if  its pipeline
                contains a Tokenizer object called 'tokenizer' and 
                a NER model called 'ner_recognizer', that 'tags'
                argument  will look like this:

                {"#syfertext_en_core_web_lg", 
                 "#syfertext_en_core_web_lg:vectors",
                 "#syfertext_en_core_web_lg:key2row",
                 "#syfertext_en_core_web_lg:tokenizer",
                 "#syfertext_en_core_web_lg:ner_recognizer",
                }

                If this argument does not contain the tag
                "#syfertext_en_core_web_lg", and it only
                includes tags of the format:
                "#syfertext_en_core_web_lg:<name>", then this signifies
                that this object is attached to another LanguageModel
                that acts as the master object, or the entry point 
                to the language model retrieval process. 
                This would be the case when the language model 
                contains a pipeline that is distributed
                over several workers due privacy and ownership constrains.
                Exposing component names with tags this way, 
                also allows language models to reuse each other's
                components.

            description: A text that describes the language model,
                 its contents, and any other features.
           
        """ 
 
