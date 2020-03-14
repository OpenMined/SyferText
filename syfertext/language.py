from .tokenizer import Tokenizer
from .pointers.tokenizer_pointer import TokenizerPointer
from .vocab import Vocab

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.string import String
from syft.generic.pointers.string_pointer import StringPointer
from typing import List, Union, Tuple


class BaseDefaults(object):
    """
       A class that defines all the defaults of the Language class
    """

    @classmethod
    def create_vocab(cls, model_name):
        """
           Creates the Vocab object that holds the vocabulary along with vocabulary meta data

           Todo:
               I started by a very simple Vocab class that
               contains only a variable called 'vectors' of type DICT to hold word vectors
               vocab.vectors['word'] = float. To be reviewed for more complex functionality.
        """

        # Instantiate the Vocab object
        vocab = Vocab(model_name)

        return vocab

    @classmethod
    def create_tokenizer(
        cls,
        vocab,
        id: int = None,
        owner: BaseWorker = None,
        client_id: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
    ):
        """Creates a Tokenizer object that will be used to create the Doc object, which is the
        main container for annotated tokens.

           Todo:
               this is a minimal Tokenizer object that is not nearly as sophisticated
               as that of spacy. It just creates tokens as space separated strings.
               Something like "string1 string2".split(' '). Of course, this should be changed later.
        """

        # Instantiate the Tokenizer object and return it
        tokenizer = Tokenizer(
            vocab,
            owner=owner,
            client_id=client_id,  # This is the id of the owner of the Language object using this tokenizer
            tags=tags,
            description=description,
        )

        return tokenizer


class Language(AbstractObject):
    """Inspired by spaCy Language class. 

    Orchestrates the interactions between different components of the pipeline
    to accomplish core text-processing task.



    It create the Doc object which is the container into which all text-processing
    pipeline components feed their results.
    """

    def __init__(
        self,
        model_name,
        id: int = None,
        owner: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
    ):

        # Define the default settings
        self.Defaults = BaseDefaults

        # Create the vocabulary
        self.vocab = self.Defaults.create_vocab(model_name)

        # Create a dictionary that associates to the name of each text-processing component
        # of the pipeline, an object that is charged to accomplish the job.
        self.factories = {"tokenizer": self.Defaults.create_tokenizer}

        self.tokenizers = dict()

        # Initialize the pipeline list
        self.pipeline = []

        super(Language, self).__init__(
            id=id, owner=owner, tags=tags, description=description
        )

    @property
    def pipe_names(self):
        """Returns a list of component names in the pipeline in order of execution.

           Returns:
               (list): List of all pipeline components in order of excution.
        """

        return [name for name, _ in self.pipeline]

    def make_doc(self, text: Union[str, String, StringPointer]):
        """Creates a Tokenizer object and uses it to tokenize 'text'. The tokens
        are stored in a Doc object which is then returned.
        """

        # Get the location ID of the "worker" where the string to be tokenized
        # is located
        if isinstance(text, StringPointer):
            location_id = text.location.id
        else:
            location_id = self.owner.id

        # Create a new Tokenizer object if one doesn't already exist on the
        # "worker" where the string is located
        if not location_id in self.tokenizers:
            self.tokenizers[location_id] = self.factories["tokenizer"](
                self.vocab,
                owner=self.owner,
                client_id=self.owner.id,  # This is the id of the owner of the Language object using this tokenizer
            )

            # Send the tokenizer to the "worker" where the string to be tokenized is located
            if isinstance(text, StringPointer) and text.location != self.owner:
                self.tokenizers[location_id] = self.tokenizers[location_id].send(
                    text.location
                )

        doc = self.tokenizers[location_id](text)

        # Return the Doc object containing the tokens
        return doc

    def add_pipe(
        self,
        component: callable,
        name: str = None,
        before: str = None,
        after: str = None,
        first: bool = False,
        last: bool = True,
    ):
        """Adds a pipeline component.


           Args:
               component (callable): This is a callable that takes a Doc object and modifies
                   it inplace.
               name (str): The name of the pipeline component to be added. Defaults to None.
               before (str): The name of the pipeline component before which the new component
                   is to be added. Defaults to None.
               after (str): The name of the pipeline component after which the new component
                   is to be added. Defaults to None.
               first (bool): if set to True, the new pipeline component will be add as the 
                   first element of the pipeline (after the tokenizer). Defaults to False.
               last (bool): if set to True, the new pipeline component will be add as the 
                   last element of the pipeline (after the tokenizer). Defaults to True.

        """

        # The component argument must be callable
        # [TODO] An exception with a custom error message should be thrown
        assert hasattr(component, "__call__"), "Argument `component` is not callable."

        # [TODO] The following requirement should be relaxed and a name should be
        # automatically assigned in case `name` is None. This would be convenient
        # as done by spaCy
        assert (
            isinstance(name, str) and len(name) >= 1
        ), "Argument `name` should be of type `str` with at least one character."

        # [TODO] Add custom error message
        assert (
            name not in self.pipe_names
        ), "Pipeline component name '{}' that you have chosen is already used by another pipeline component.".format(
            name
        )

        # Make sure only one of 'before', 'after', 'first' or 'last' is set
        # [TODO] Add custom error message
        assert (
            sum([bool(before), bool(after), bool(first), bool(last)]) < 2
        ), "Only one among arguments 'before', 'after', 'first' or 'last' should be set."

        # Create a pipe
        pipe = (name, component)

        # Add the pipe at the right place
        if last or not any([before, after, first]):
            self.pipeline.append(pipe)
        elif first:
            self.pipeline.insert(0, pipe)
        elif before in self.pipe_names:
            self.pipeline.insert(self.pipe_names.index(before), pipe)
        elif after in self.pipe_names:
            self.pipeline.insert(self.pipe_names.index(after) + 1, pipe)
        else:
            # [TODO] Raise exception with custom error message
            assert (
                False
            ), "component cannot be added to the pipeline, \
                           please double check argument values for the `add_pipe` method call."

    def remove_pipe(self, name: str) -> Tuple[str, callable]:
        """Removes the pipeline whose name is 'name'

           Args:
               name (str): The name of the pipeline component to remove.

           Returns:
               The removed pipe 

        """

        # [TODO] Add custom error message
        assert (
            name in self.pipe_names
        ), "No pipeline component with the specified name '{}' was found".format(name)

        # Get the index of the pipeline to be removed in the
        # self.pipeline list
        pipe_index = self.pipe_names.index(name)

        # Delete the pipe using its index
        pipe = self.pipeline.pop(pipe_index)

        return pipe

    def __call__(self, text: Union[str, String, StringPointer]):
        """Here is where the real work is done. The pipeline components
        are called here, and the Doc object containing their results is created
        here too.
        """

        # create the Doc object with the tokenized text in it
        doc = self.make_doc(text)

        # Apply the pipeline components sequentially
        # Each component will modify the document `doc` inplace
        for _, component in self.pipeline:
            component(doc)

        # return the Doc object
        return doc
