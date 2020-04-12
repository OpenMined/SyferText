from .tokenizer import Tokenizer
from .vocab import Vocab
from .doc import Doc
from .pointers.doc_pointer import DocPointer
from .pipeline import SubPipeline

from syft.generic.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.string import String
from syft.generic.pointers.string_pointer import StringPointer
from syft.generic.pointers.object_pointer import ObjectPointer
from typing import List, Union, Tuple


class BaseDefaults(object):
    """A class that defines all the defaults of the Language class
    """

    @classmethod
    def create_vocab(cls, model_name) -> Vocab:
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
    def create_tokenizer(cls, vocab,) -> Tokenizer:
        """Creates a Tokenizer object that will be used to create the Doc object, which is the
        main container for annotated tokens.

        """

        # Instantiate the Tokenizer object and return it
        tokenizer = Tokenizer(vocab,)

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
        self.factories = {"tokenizer": self.Defaults.create_tokenizer(self.vocab)}

        # Initialize the subpipeline template
        # It only contains the tokenizer at initialization
        self.pipeline_template = [{"remote": True, "name": "tokenizer"}]

        # Intialize the main pipeline
        self._reset_pipeline()

        super(Language, self).__init__(id=id, owner=owner, tags=tags, description=description)

    @property
    def pipe_names(self) -> List[str]:
        """Returns a list of component names in the pipeline in order of execution.

        Returns:
            (list): List of all pipeline component name in order of execution.
        """

        return [pipe_template["name"] for pipe_template in self.pipeline_template]

    def _parse_pipeline_template(self):
        """Parses the `pipeline_template` property to 
        create the `subpipeline_templates` property.
        """

        # Initialize a subpipeline template with the
        # tokenizer. The tokenizer alway has 'remote' set
        # to True.
        subpipeline_template = dict(
            remote=self.pipeline_template[0]["remote"], names=[self.pipeline_template[0]["name"]],
        )

        # Initialize the subpipeline templates list as a class property
        self.subpipeline_templates = [subpipeline_template]

        # Loop through the pipeline template elements
        for pipe_template in self.pipeline_template[1:]:

            # compare `remote` properties between templates:
            # If the pipe template has the same `remote` value,
            # it is appended to the existing subpipeline template
            if pipe_template["remote"] == subpipeline_template["remote"]:
                subpipeline_template["names"].append(pipe_template["name"])

            # Otherwise, create a new subpipeline template and add the
            # pipe template to it
            else:
                subpipeline_template = dict(
                    remote=pipe_template["remote"], names=[pipe_template["name"]]
                )

                self.subpipeline_templates.append(subpipeline_template)

    def _reset_pipeline(self):
        """Reset the `pipeline` class property.
        """

        # Read the pipeline components from the template and aggregate them into
        # a list of subpipline templates.
        # This method will create the instance variable
        # self.subpipeline_templates
        self._parse_pipeline_template()

        # Get the number of subpipelines
        subpipeline_count = len(self.subpipeline_templates)

        # Initialize a new empty pipeline with as many
        # empty dicts as there are subpipelines
        self.pipeline = [dict() for i in range(subpipeline_count)]

    def add_pipe(
        self,
        component: callable,
        remote: bool = False,
        name: str = None,
        before: str = None,
        after: str = None,
        first: bool = False,
        last: bool = True,
    ):
        """Adds a pipe template to the pipeline template. 
           
        A pipe template is a dict of the form `{'remote': remote, 'name': name}`.
        Few main steps are carried out here:

        1- The new pipe name is added at the right position in the pipeline template.
           Here is an example of how pipeline template list looks like

           self.pipeline_template = [{'remote': True, 'name': 'tokenizer'},
                                     {'remote': True, 'name': <pipe_1_name>},
                                     {'remote': True, 'name': <pipe_2_name>},
                                     {'remote': False, 'name': <pipe_3_name>},
                                     {'remote': False, 'name': <pipe_4_name>}]

        2- The pipeline template is parsed into a list or subpipeline templates.
           Each subpipeline template is an aggregation of adjacent pipes with 
           the same value for 'remote'
           Here is an example of how the subpipeline template list for the above
           pipeline template would look like:
        
           self.subpipeline_templates = [{'remote': True, 'names': ['tokenizer', 
                                                                    'pipe_1_name',
                                                                    'pipe_2_name']},
                                         {'remote': False, 'name': ['pipe_3_name',
                                                                   'pipe_4_name']}
                                        ]

        3- The pipeline is initialize by creating a list with as many empty dicts as
           there are subpipelines:

           self.pipeline = [dict(), dict()]
                                                               

        Args:
            component (callable): This is a callable that takes a Doc object and modifies
                it inplace.
            name (str): The name of the pipeline component to be added. Defaults to None.
            remote (bool): If True, the pipe component will be sent to the remote worker
                where the Doc object resides. If False, the pipe will operate locally,
                either on a Doc object directly, or on a DocPointer returned by the previous
                component in the pipeline. Defaults to False.
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
        assert hasattr(component, "__call__"), "Argument `component` is not a callable."

        # Make sure the `component` argument is an object that has a `factory()` method
        assert hasattr(
            component, "factory"
        ), "Argument `component` should be an object that has a `factory()` method"
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

        # Add the new pipe component to the list of factories
        self.factories[name] = component

        # Create the pipe template that will be added the pipeline
        # template
        pipe_template = dict(remote=remote, name=name)

        # Add the pipe template at the right position
        if last or not any([before, after, first]):
            self.pipeline_template.append(pipe_template)

        elif first:
            # The index 0 is reserved for the tokenizer
            self.pipeline_template.insert(index=1, element=pipe_template)

        elif before in self.pipe_names:
            self.pipeline_template.insert(
                index=self.pipe_names.index(before), element=pipe_template
            )

        elif after in self.pipe_names:
            self.pipeline_template.insert(
                index=self.pipe_names.index(after) + 1, element=pipe_template
            )
        else:
            # [TODO] Raise exception with custom error message
            assert (
                False
            ), "component cannot be added to the pipeline, \
                please double check argument values of the `add_pipe` method call."

        # Reset the pipeline.
        # The instance variable that will be affected is:
        # self.pipeline
        self._reset_pipeline()

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
        pipe = self.pipeline_template.pop(pipe_index)

        # Reset the pipeline.
        self._reset_pipeline()

        return pipe

    def _run_subpipeline_from_template(
        self, template_index: int, input=Union[str, String, StringPointer, Doc, DocPointer],
    ) -> Union[Doc, DocPointer]:
        """Runs the subpipeline at position `template_index` of
        self.pipeline on the appropriate worker. 
 
        The worker on which the subpipeline is run is either the
        the same worker on which `input` lives, if the `remote`
        property of the subpipeline template is True. Or, it is the
        local worker if `remote` is False.

        If no subpipeline is yet created for the specified worker, 
        one is created using the template, and added to the pipeline.

        Args:
            template_index (int): The index of the subpipeline 
                template in `self.subpipelines_templates`
            input (str, String, StringPointer, Doc, DocPointer):
                The input on which the subpipeline operates.
                It can be either the text to tokenize (or a pointer
                to it) for the subpipeline at index 0, or it could
                be the Doc (or its pointer) for all subsequent 
                subpipelines.

        Returns:
            (Doc or DocPointer): The new or updated Doc object or 
               a pointer to a Doc object.
        
        """

        # Get the location ID of the worker where the text to be tokenized,
        # or the Doc to be processed is located
        if isinstance(input, ObjectPointer):
            location_id = input.location.id
        else:
            location_id = self.owner.id

        # Create a new SubPipeline object if one doesn't already exist on the
        # worker where the input is located
        if location_id not in self.pipeline[template_index]:

            # Get the subpipeline template
            subpipeline_template = self.subpipeline_templates[template_index]

            # Is the pipeline a remote one?
            remote = subpipeline_template["remote"]

            # Instantiate a subpipeline and load the subpipeline template
            subpipeline = SubPipeline()

            subpipeline.load_template(template=subpipeline_template, factories=self.factories)

            # Add the subpipeline to the pipeline
            self.pipeline[template_index][location_id] = subpipeline

            # Send the subpipeline to the worker where the input is located
            if (
                isinstance(input, ObjectPointer)
                and input.location != self.owner  # Is the input remote?
                and remote  # Is the subpipeline is sendable?
            ):
                self.pipeline[template_index][location_id] = self.pipeline[template_index][
                    location_id
                ].send(input.location)

        # Apply the subpipeline and get the doc or the Doc id.
        # If a Doc ID is obtained, this signifies the ID of the
        # Doc object on the remote worker.
        doc_or_id = self.pipeline[template_index][location_id](input)

        # If the doc is of type str or int, this means that a
        # DocPointer should be created
        if isinstance(doc_or_id, int) or isinstance(doc_or_id, str):

            doc = DocPointer(location=input.location, id_at_location=doc_or_id, owner=self.owner)

        # This is of type Doc then
        else:
            doc = doc_or_id

        # return the doc
        return doc

    def __call__(self, text: Union[str, String, StringPointer]) -> Union[Doc, DocPointer]:
        """The text is tokenized and  pipeline components are called 
        here, and the Doc object is returned.

        Args:
            text (str, String or StringPointer): the text to be tokenized and 
        processed by the pipeline components.

        Returns:
            (Doc or DocPointer): The Doc object or a pointer to a Doc object.
        This object provides access to all token data.
        """

        # Runs the first subpipeline.
        # The first subpipeline is the one that has the tokenizer
        doc = self._run_subpipeline_from_template(template_index=0, input=text)

        # Apply the the rest of subpipelines sequentially
        # Each subpipeline will modify the document `doc` inplace
        for i, subpipeline in enumerate(self.pipeline[1:], start=1):
            doc = self._run_subpipeline_from_template(template_index=i, input=doc)

        # return the Doc object
        return doc
