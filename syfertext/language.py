from .tokenizer import Tokenizer
from .vocab import Vocab
from .doc import Doc
from .pointers.doc_pointer import DocPointer
from .pipeline import SubPipeline
from .pipeline import SingleLabelClassifier
from .pipeline import SimpleTagger
from .state import State
from .pipeline import Pipeline


from syft.generic.abstract.object import AbstractObject
from syft.workers.base import BaseWorker
from syft.generic.string import String
from syft.generic.pointers.string_pointer import StringPointer
from syft.generic.pointers.object_pointer import ObjectPointer
import torch.nn as nn

from collections import defaultdict
from typing import List
from typing import Union
from typing import Tuple
from typing import Set
from typing import Dict

from collections import defaultdict


class Language(AbstractObject):
    """Inspired by spaCy Language class.

    Orchestrates the interactions between different components of the pipeline
    to accomplish core text-processing task.

    It create the Doc object which is the container into which all text-processing
    pipeline components feed their results.
    """

    def __init__(
        self,
        pipeline_name: str,
        owner: BaseWorker = None,
        tags: List[str] = None,
        description: str = None,
    ):

        # Set the pipeline name
        self.pipeline_name = pipeline_name

        # Initialize the subpipeline template
        self.pipeline_template = []

        # Create a dictionary that associates to the name of each text-processing component
        # of the pipeline, an object that is charged to accomplish the job.
        self.factories = dict()

        # Initialize an empty dict of State object.
        # The keys of this dict are the names of the components
        # whose states are being stored, e.g., 'vocab', 'stopword_tagger', etc.
        self.states = {}

        # Initialize the property that should hold the subpipeline templates
        # list for each worker
        self.subpipeline_templates = defaultdict(list)

        # Initialize the pipeline as an empty dictionary
        self._reset_pipeline()

        super(Language, self).__init__(id=None, owner=owner, tags=tags, description=description)

    @property
    def pipe_names(self) -> List[str]:
        """Returns a list of component names in the pipeline in order of execution.

        Returns:
            (list): List of all pipeline component name in order of execution.
        """

        return [pipe_template["name"] for pipe_template in self.pipeline_template]

    def set_tokenizer(self, tokenizer: Tokenizer, name: str = None, access: Set[str] = None):
        """Set the tokenizer object.

        Args:
            tokenizer: the Tokenizer object.
            name: The name of the Tokenizer. This argument is optional. If it is left
                to its default None value, it will take the 'tokenizer' value.
            access: The set of worker ids where this tokenizer's state could be sent.
                if the string '*' is included in the set,  then all workers are allowed
                to receive a copy of the state. If set to None, then only the worker where this
                component is saved will be allowed to get a copy of the state.
        """

        # Set the most restrictive access by default
        if access is None:
            access = {self.owner.id}

        # Set the name of the tokenizer
        name = tokenizer.__class__.__name__.lower() if name is None else name

        # Add the tokenizer to the pipeline
        self.add_pipe(component=tokenizer, name=name, access=access)

    def set_vocab(self, vocab: Vocab, access: Set[str] = None) -> None:
        """Load a new vocab to the Language object. This methods modifies the
        `vocab` propery.

        Args:
            vocab: The Vocab object.
            access: The set of worker ids where this Vocab's state could be sent.
                if the string '*' is included in the set,  then all workers are allowed
                to receive a copy of the state. If set to None, then only the worker where this
                component is saved will be allowed to get a copy of the state.
        """

        # Set the most restrictive access by default
        if access is None:
            access = {self.owner.id}

        # Set some properties
        vocab.pipeline_name = self.pipeline_name
        vocab.access = access
        vocab.name = "vocab"

        # Get the state of the vocab object
        state = vocab.dump_state()

        # Save the state in the object store

        self._save_state(state=state, name=vocab.name, access=access)

    def load_pipeline(self, template, states):

        # Load the states
        self.states = states

        # Load the pipeline template
        self.pipeline_template = template

        # Create the factories;
        # Create and entry for each pipe component
        self.factories = dict()

        for pipe_template in self.pipeline_template:

            # Get the pipe name and its class name
            name = pipe_template["name"]
            class_name = pipe_template["class_name"]

            self.factories[name] = globals()[class_name]

    def _save_state(self, state: State, name: str, access: Set[str] = None):
        """Saves a State object in the object store of the local worker.
        Make sure that the local workers `is_client_worker` is set to False.

        Args:
            state: The State object to save to the object store of the local
                worker.
            name: The name of the component associated with the state.
            access: The set of worker ids where the state could be sent.
        """

        # Add to the list of State objects known to this Language object
        self.states[name] = dict(state=state, access=access)

        # Register it in the object store
        self.owner.register_obj(state)

    def _parse_pipeline_template(self, location_id: str) -> None:
        """Parses the `pipeline_template` property to
        create the subpipeline templates for `worker`.

        Args:
            location_id: The ID of the worker according to
                which the subpipeline template should be parsed.
        """

        # If the pipeline template is already parsed for this
        # location, return.
        if location_id in self.pipeline:
            return

        # Create an entry of `location_id` in the pipeline
        self.pipeline[location_id] = []

        # Initialize a subpipeline template
        subpipeline_template = dict(names=[])

        # Create the subpipeline templates for that location
        self.subpipeline_templates[location_id].append(subpipeline_template)

        # Loop through the pipeline template elements
        for pipe_template in self.pipeline_template:

            # Check out whether the designated worker whose ID is
            # `location_id` has access to the pipe component,
            # If it does, append the pipe template to the currently
            # processed subpipeline template
            # Notice that PySyft models should never be sent to a
            # remote worker even if their `access` property of the corresponding
            # state allows for this
            # The reason I use .get() here is just for better code lisibility
            access = self.states.get(pipe_template["name"]).get("access")

            if {"*", location_id} & access and pipe_template["class_name"] != nn.Module.__name__:

                subpipeline_template["names"].append(pipe_template["name"])

            # Otherwise, create a new subpipeline template and add the
            # pipe template to it
            else:
                subpipeline_template = dict(names=[pipe_template["name"]])

                self.subpipeline_templates[location_id].append(subpipeline_template)

        # Now create the subpipeline objects
        for subpipeline_template in self.subpipeline_templates[location_id]:

            # Instantiate a subpipeline and load the subpipeline template
            subpipeline = SubPipeline(pipeline_name=self.pipeline_name)

            subpipeline.load_template(template=subpipeline_template, factories=self.factories)

            # Send the subpipeline to the worker where the input is located
            # if the destination worker is different from the local one
            if location_id != self.owner.id:
                subpipeline = subpipeline.send(location_id)

            # Add the subpipeline to the pipeline
            self.pipeline[location_id].append(subpipeline)

        # Now load the state of each pipe in the subpipelines.
        # I could have done this step in the previous loop, but
        # I do it here in order to separated parsing from loading
        # states that might take significant longer time.
        for subpipeline in self.pipeline[location_id]:

            subpipeline.load_states()

    def _reset_pipeline(self):
        """Reset the `pipeline` class property."""

        # Initialize a new empty pipeline with as an empty dict
        self.pipeline = {}

        # Initialize a new `subpipelins_template` property
        self.subpipeline_templates = defaultdict(list)

    def add_pipe(
        self,
        component: callable,
        access: Set[str] = None,
        name: str = None,
        before: str = None,
        after: str = None,
        first: bool = False,
        last: bool = True,
    ):

        """Adds a pipe template to the pipeline template.

        Args:
            component (callable): This is a callable that takes a Doc object and modifies
                it inplace.
            access: The set of worker ids where this component's state could be sent.
                if the string '*' is included in the set,  then all workers are allowed
                to receive a copy of the state. If set to None, then only the worker where this
                component is saved will be allowed to get a copy of the state.
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
        assert hasattr(component, "__call__"), "Argument `component` is not a callable."

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

        if access is None:
            access = {self.owner.id}

        # Add the new pipe component to the list of factories
        self.factories[name] = globals()[component.__class__.__name__]

        # Set the pipeline name to which this tokenizer belongs.
        component.pipeline_name = self.pipeline_name

        # Set the component name
        component.name = name

        # Set the component access rules
        component.access = access

        # Get the component's state
        state = component.dump_state()

        # Save the component's state
        self._save_state(state=state, name=name, access=access)

        # Create the pipe template that will be added the pipeline
        # template
        pipe_template = dict(name=name, class_name=component.__class__.__name__)

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
        ), f"No pipeline component with the specified name '{name}' was found"

        # Get the index of the pipeline to be removed in the
        # self.pipeline list
        pipe_index = self.pipe_names.index(name)

        # Delete the pipe using its index
        pipe = self.pipeline_template.pop(pipe_index)

        del self.factories[name]

        # Reset the pipeline.
        self._reset_pipeline()

        return pipe

    def _run_subpipeline_from_template(
        self,
        template_index: int,
        location_id: str,
        input=Union[str, String, StringPointer, Doc, DocPointer],
    ) -> Union[Doc, DocPointer]:
        """Creates a `subpipeline` object and sends it to the appropriate
        worker if `input` is remote. Then runs the subpipeline at position
        `template_index` of self.pipeline on the appropriate worker.

        The worker on which the subpipeline is run is either the
        the same worker on which `input` lives, if the `remote`
        property of the subpipeline template is True. Or, it is the
        local worker if `remote` is False.

        The self.pipeline is a list of dicts which stores
        the references to Subpipeline (or SubpipelinePointer) objects.
        It is created by parsing self.subpipeliline_template.

        At each index in self.pipeline we have a dictionary which
        holds the location id as the key and a reference to the
        SubPipeline (or SubpipelinePointer) object as the value.

        The text (or StringPointer) is processed by the tokenizer to create
        Doc (or DocPointer) object which is then processed in place by
        the subpipeline components in the relative order in which
        they are stored in self.pipeline.

        If no subpipeline is yet created for the specified worker,
        one is created using the template in self.subpipeline_templates
        and added to the pipeline.

        Example:
            If a text is to be processed on the local machine, then subpipelines
            in the pipeline are initialized on the local machine.
            So if there are 3 subpipelines in the pipeline then

            self.pipeline = [{'me': SubPipeline[pipe_1_name > pipe_2_name]},
                             {'me': SubPipeline[pipe_3_name]},
                             {'me': SubPipeline[pipe_4_name, pipe_5_name]},
                            ]

            Now any more local texts that are to be processed by the pipeline are
            processed with the subpipeline objects already present on the local machine.

            Then if we pass a StringPointer referring to a text located on a remote
            machine, new SubPipeline objects whose templates in `self.subpipeline_templates`
            has "remote = True" are created on remote machines and their references
            are stored in dictionaries at appropriate indices in `self.pipeline` with
            remote location id as the key.

            Hence in our example, if subpipeline templates at the first and third indices
            in `self.subpipeline_templates` have "remote = True". Then upon passing a
            StringPointer to the pipeline, new subpipeline objects are initialized on
            bob's machine and their references are stored in dictionaries at the first
            and third indices in `self.pipeline` with `bob` as the key.

            self.pipeline = [{'me' : SubPipeline[pipe_1_name > pipe_2_name],
                              'bob': SubPipelinePointer[pipe_1_name > pipe_2_name]},
                             {'me' : SubPipeline[pipe_3_name]},
                             {'bob': SubPipelinePointer[pipe_4_name > pipe_5_name],
                              'me' : SubPipeline[pipe_4_name > pipe_5_name]}
                            ]

            Note:
                The order of workers stored in the dictionary in self.pipeline is arbitrary.

        Args:
            template_index (int): The index of the subpipeline template in
                `self.subpipelines_templates`
            location_id: The ID of the worker on which processing will take place.
            input (str, String, StringPointer, Doc, DocPointer):
                The input on which the subpipeline operates. It can be either the text
                to tokenize (or a pointer to it) for the subpipeline at index 0, or it
                could be the Doc (or its pointer) for all subsequent subpipelines.

        Returns:
            (Doc or DocPointer): The new or updated Doc object or
               a pointer to a Doc object.

        """

        # Apply the subpipeline and get the doc or the Doc id.
        # If a Doc ID is obtained, this signifies the ID of the
        # Doc object on the remote worker.
        doc_or_id = self.pipeline[location_id][template_index](input)

        # If the doc is of type (str or int), this means that a
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

        # Get the location ID of the worker where the text to be tokenized,
        # or the Doc to be processed is located
        if isinstance(text, ObjectPointer):
            location_id = text.location.id
        else:
            location_id = self.owner.id

        # Create a subpipeline templates list for the worker where `input` is located
        # If it does not already exist
        self._parse_pipeline_template(location_id=location_id)

        # Runs the first subpipeline.
        # The first subpipeline is the one that has the tokenizer
        doc = self._run_subpipeline_from_template(
            template_index=0, location_id=location_id, input=text
        )

        # Apply the the rest of subpipelines sequentially
        # Each subpipeline will modify the document `doc` inplace
        for i in range(1, len(self.pipeline[location_id])):
            doc = self._run_subpipeline_from_template(
                template_index=i, location_id=location_id, input=doc
            )

        # return the Doc object
        return doc

    def deploy(self, worker: BaseWorker) -> None:
        """Deploys the pipeline to PyGrid by creating a LanguageModel
        object and sending it to the worker where it is to be deployed.
        The State objects of every pipe component and every resource are
        also deployed to the same worker.

        Args:
            worker: The worker on which the pipeline is to be deployed.
        """

        # Set the `location_id` property of each state to
        # the worker on which the pipeline is deployed
        states = self.states.copy()

        for pipe_name in self.states:

            # Change the pipe's location
            states[pipe_name]["location_id"] = worker.id

        # Create a Pipeline object
        pipeline = Pipeline(
            name=self.pipeline_name,
            template=self.pipeline_template,
            states=states,
            owner=self.owner,
            tags=self.tags,
            description=self.description,
        )

        # Send the Pipeline object to the destination worker
        pipeline_pointer = pipeline.send(location=worker)

        # Tell the Pipelin object to deploy all State objects
        pipeline_pointer.deploy_states()
