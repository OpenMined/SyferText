import syft as sy
import syfertext
import torch
from syft.generic.string import String
from syfertext.doc import Doc
from syfertext.pointers import DocPointer
from syfertext.pipeline import SubPipeline, SimpleTagger
from syfertext.pipeline.pointers import SubPipelinePointer


hook = sy.TorchHook(torch)
me = hook.local_worker


def test_subpipeline_is_not_recreated_in_remote_workers():
    """
    Test that the a subpipeline at a given index is not recreated in remote workers after it
    has been initialized once. Each worker contains a single subpipeline, with multiple components.
    """

    nlp = syfertext.load("en_core_web_lg", owner=me)

    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")

    # Create 4 PySyft Strings and send them to remote workers
    # (3 to Bob, 1 to Alice)
    texts = [String(text) for text in ["hello", "syfertext", "private", "nlp"]]

    texts_ptr = [texts[0].send(bob), texts[1].send(bob), texts[2].send(alice), texts[3].send(bob)]

    # The first time a text owned by `bob` is tokenized, a `SubPipeline` object is
    # created by the `nlp` object and sent to `bob`. The `nlp` object keeps a
    # pointer to the tokenizer object in `bob`'s machine.
    doc1 = nlp(texts_ptr[0])
    subpipelines = [v for v in bob._objects.values() if isinstance(v, SubPipeline)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1
    assert len(documents) == 1
    assert len(nlp.pipeline[0].keys()) == 1
    assert list(nlp.pipeline[0].keys())[0] == "bob"
    assert isinstance(nlp.pipeline[0]["bob"], SubPipelinePointer)

    # The second time a text owned by `bob` is tokenized, no new `SubPipeline`
    # objects are created. Only a new document on `bob`'s machine.
    doc2 = nlp(texts_ptr[1])
    subpipelines = [v for v in bob._objects.values() if isinstance(v, SubPipeline)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1
    assert len(documents) == 2
    assert len(nlp.pipeline[0].keys()) == 1

    # The first time a text owned by `alice` is tokenized, a new `SubPipeline` object
    # is created by the `nlp` object and sent to `alice`. Now the `nlp` object has
    # a second pointer to a `SubPipeline`.
    doc3 = nlp(texts_ptr[2])
    subpipelines = [v for v in alice._objects.values() if isinstance(v, SubPipeline)]
    documents = [v for v in alice._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1
    assert len(documents) == 1
    assert len(nlp.pipeline[0].keys()) == 2
    assert list(nlp.pipeline[0].keys())[1] == "alice"
    assert isinstance(nlp.pipeline[0]["alice"], SubPipelinePointer)

    # The third time a text owned by `bob` is tokenized, no new `SupPipeline`
    # objects are created. The `nlp` object still has the same previous pointer
    # to a `SubPipeline` on `bob`'s machine and `bob` now has a third document.
    doc4 = nlp(texts_ptr[3])
    subpipelines = [v for v in bob._objects.values() if isinstance(v, SubPipeline)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1
    assert len(documents) == 3
    assert len(nlp.pipeline[0].keys()) == 2


def test_subpipeline_component_addition_and_removal_from_pipeline():
    """Test that on addition of a pipeline component with remote
    set as True or False, it is added to appropriate subpipeline on
    remote/local worker"""

    nlp = syfertext.load("en_core_web_lg", owner=me)

    tagger = SimpleTagger(attribute="noun", lookups=["he", "she"], tag=True)

    # Note: If a pipeline component is added to the pipeline with remote set as True,
    # then it can be sent to a remote worker, if the doc/string object to be processed is on a remote machine

    # Add tagger to the pipeline
    nlp.add_pipe(tagger, name="my tagger")  # Note: remote = False (default)

    # Now, remote is set to True by default for Tokenizer, hence it is created where the doc/strings is located
    # Since the remote values of Tokenizer and tagger(added above) are different, tagger is added in another subpipeline
    # Assert nlp pipeline has two subpipelines
    assert len(nlp.pipeline) == 2  # Tokenizer is added by default

    # Remove tagger from the pipeline
    nlp.remove_pipe(name="my tagger")

    # Assert pipeline has single subpipeline
    assert len(nlp.pipeline) == 1

    # Add tagger to the pipeline
    nlp.add_pipe(tagger, name="my tagger", remote=True)  # Note: remote = True

    # Now, remote = True by default for Tokenizer, hence it is created where the doc/strings is located
    # Since the remote values of Tokenizer and tagger are same, tagger is added in the same subpipeline as tokenizer
    # Assert nlp pipeline has only one subpipeline
    assert len(nlp.pipeline) == 1

    # Now, If we add another tagger, with remote = False
    nlp.add_pipe(tagger, name="my_new_tagger", remote=False)

    # Assert nlp pipeline has two subpipelines
    assert len(nlp.pipeline) == 2


def test_pipeline_output():

    nlp = syfertext.load("en_core_web_lg", owner=me)

    james = sy.VirtualWorker(hook, id="james")

    # Create a PySyft String and send it to remote worker james
    text_ptr = String("building syfertext").send(james)

    # Add tagger with remote = True
    tagger = SimpleTagger(attribute="noun", lookups=["syfertext"], tag=True)
    nlp.add_pipe(tagger, name="noun_tagger", remote=True)

    # Now, Upon processing the text present on james's machine,
    # pipeline Should return a DocPointer to the doc on james's machine
    doc = nlp(text_ptr)
    assert isinstance(doc, DocPointer)

    # assert only one document object on james's machine
    documents = [v for v in james._objects.values() if isinstance(v, Doc)]
    assert len(documents) == 1

    # assert returned doc_pointer points to document object on james's machine
    assert doc.id_at_location == documents[0].id

    # assert only one subpipeline object on james's machine
    subpipelines = [v for v in james._objects.values() if isinstance(v, SubPipeline)]
    assert len(subpipelines) == 1

    # Make sure subpipeline object contains tokenizer and tagger
    pipes = subpipelines[0].pipe_names
    assert len(pipes) == 2
    assert pipes[0] == "tokenizer"
    assert pipes[1] == "noun_tagger"

    # nlp.pipeline stores pointers to subpipeline objects on remote machines
    # assert subpipeline pointer stored in nlp.pipeline points to the subpipeline on james machine
    assert nlp.pipeline[0]["james"].id_at_location == subpipelines[0].id
