from syft.generic.object import AbstractObject
from syft.worker.base import BaseWorker

from typing import Union
from typing import Set


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

    def __init__(
        self,
        source: str,
        name: str,
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

        self.name = name

        # Set the tag ad #<model_name>. This tag will be
        # used to search for this language model on PyGrid
        if tags is None:
            self.tags = set()

        self.tags.add(f"#{name}")

        # Initialize the pipeline template
        self.pipeline = []

        # Initialize the parent class
        super(LanguageModel, self).__init__(id=id, owner=owner, tags=tags, description=description)

    @property
    def pipe_names(self):
        """Get a list of all pipe names included in the pipeline.

        Returns:
            name (List[str]): A list of all pipe name in the pipeline.
        """

        names = set([pipe_template["name"] for pipe_template in self.pipeline_template])

        return names

    def remove_pipe(self, name: str):
        """Remove a pipe template from the pipeline template.

        Args:
            name: The name of the pipe to remove.
        """

        self.pipeline_template = [pipe for pipe in self.pipeline_template if pipe["name"] != name]

    def add_pipe_template(self, name: str, owner: str, access: Union[None, Set[str]]):
        """Adds a pipe template to the pipeline template of 
        the language model.
        """

        assert (
            name not in self.pipe_names
        ), f"A pipe with name '{name}' already exists in the pipeline"

        # Create the pipe template
        pipe_template = dict(name=name, owner=owner, access=access)

        # Add the pipe template to the pipeline template
        self.pipeline_template.append(pipe_template)

        """
        1- To create a brand new language model, we always start by creating
        a Language object, and then calling calling nlp.to_grid(worker) which
        will create the LanguageModel object and the underlying LanguageResource
        objects and push it to PyGrid. We should never create the LanguageModel directly.

        tokenizer = Tokenizer().set_resources(prefixes, ...)  # Should know how to convert itself to LanguageResource

        vocab = Vocab().set_resources(keys = keys, key2row = key2row, vectors = vectors) # Should know ...... get_resources() called by nlp.to_grid()

        nlp = Language(model_name = 'my_new_model')
        nlp.set_tokenizer(tokenizer)
        nlp.set_vocab(vocab)

        nlp.to_grid(worker) # This will put all components on same worker with public access
        nlp.to_grid(tokenizer = {'owner': bob, 'access': None}},
                    vocab = {'owner' : alice, access': {alice, james}}
                    my_tagger = {'owner' alice},
        ) # notice that the component/pipe name is used as kwarg.

        - Factories should be removed and replace by the class name 'SimpleTagger' that
          is known to SyferText. The factory method is replaced by 'load_resources()' and
          'dump_resources() -> LanguageResource'

        - We need a method in PySyft similar to .get() but does not remove object
          from destination, just copy it .copy() to  copy language resources from
          their pointers inside the 'load_resources()' methods. Or in LanguageResourcePointer
          object I implement a method 'pull_resource()' that calls the 'send()' method of
          LanguageResource, which in turn, compresses the simple object and returns it.

        - start by adding dump/load_resources() to Tokenizer and Vocab

        - add set_tokenizer() and set_vocab() to Language

        - when adding any pipe or set_vocab/tokenizer, we call directly dump_resources()
          this stores the corresponding LanguageResources objects in the store.

        - Then I create the pipeline template as (name = 'stop_tagger', class_name = 'SimpleTagger', owner = Union[None, str])

        - in SubPipeline, I remove the factory and use something like
          syfertext.factories['<class_name>'](name = <resource/pipe_name>, model_name = "<language_model_name>")

        - The LanguageResource object contains as properties: resource (simple object style), tag = <language_model>:<resource_name>

        - Predictive models should be plans. encapsulated in object Classifier for instance that calls the plan's build() method. and
          implements dump_resources() and load_resources(). The dump_resources() methods create a LanguageResource object with
          'resource = built plan'


        - I think the method load_state should be called explicitely in the SubPipeline after each object is sent to the
          corresponding worker.

        """
