import syft as sy
import torch
import syfertext

from syft.generic.string import String
from syfertext.pipeline import SimpleTagger

from syfertext.exceptions import DuplicateNameError
from syfertext.exceptions import ObjectNotCallableError
from syfertext.exceptions import InvalidPositionError
from syfertext.exceptions import PipelineComponentNotFoundError
from syfertext.exceptions import SubPipelineNotCollocatedError
from syfertext.exceptions import ObjectNotCollocatedError

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)

# initialise workers
bob = sy.VirtualWorker(hook, id="bob")
alice = sy.VirtualWorker(hook, id="alice")

# Create a simple tagger
tagger = SimpleTagger(attribute="pronoun", lookups=["he", "she"])


def test_duplicate_name_error():

    try:
        name = "my tagger"
        nlp.add_pipe(component=tagger, name=name)

        # adding component with duplicate name
        nlp.add_pipe(component=tagger, name=name)

    except Exception as e:
        assert isinstance(e, DuplicateNameError)


def test_object_not_callable_error():

    try:
        not_callable_object = "a string"

        # adding component str which can't be called
        nlp.add_pipe(component=not_callable_object)

    except Exception as e:
        assert isinstance(e, ObjectNotCallableError)


def test_pipeline_component_not_found_error():

    try:
        # component with passed name is not available in pipeline
        nlp.remove_pipe("not my tagger")

    except Exception as e:
        assert isinstance(e, PipelineComponentNotFoundError)


def test_invalid_position_error():

    try:
        # Setting last as False but not providing any other position
        nlp.add_pipe(component=tagger, name="my new tagger", last=False)

    except Exception as e:
        assert isinstance(e, InvalidPositionError)

    try:
        # Passing more than one position as True
        nlp.add_pipe(component=tagger, name="my new tagger", last=True, first=True)

    except Exception as e:
        assert isinstance(e, InvalidPositionError)


def test_object_not_collocated_error():

    try:
        nlp = syfertext.load("en_core_web_lg", owner=me)
        tagger = SimpleTagger(attribute="pronoun", lookups=["he", "she"])

        # Add a simple tagger with remote = False
        nlp.add_pipe(tagger, name="my tagger", remote=False)

        text_ptr = String("Building SyferText").send(bob)

        # The tokenizer is initialised and sent to bob's machine
        # while the tagger resides on local machine
        # cause it was added with remote = False.
        # Hence after processing the text the tokenizer returns a
        # DocPointer which is passed to tagger.
        # The tagger can't process a DocPointer object, hence
        # it will throw an ObjectNotCollocatedError
        _ = nlp(text_ptr)

    except Exception as e:
        assert isinstance(e, ObjectNotCollocatedError)
