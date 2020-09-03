import syft as sy
import syfertext
import torch
from syft.generic.string import String
from syfertext.doc import Doc
from syfertext.pointers import DocPointer
from syfertext.pipeline import SubPipeline, SimpleTagger
from syfertext.pipeline.pointers import SubPipelinePointer
from syfertext.local_pipeline import get_test_language_model


hook = sy.TorchHook(torch)
me = hook.local_worker

alice = sy.VirtualWorker(hook, id="alice")
bob = sy.VirtualWorker(hook, id="bob")
james = sy.VirtualWorker(hook, id="james")


# Create components which can be added to a text processing pipeline
noun_tagger = SimpleTagger(attribute="noun", lookups=["syerftext", "pysyft"], tag=True)
verb_tagger = SimpleTagger(attribute="verb", lookups=["build", "learn"], tag=True)
pronoun_tagger = SimpleTagger(attribute="pronoun", lookups=["he", "she"], tag=True)
adjective_tagger = SimpleTagger(attribute="adjective", lookups=["secure"], tag=True)


def test_addition_and_removal_of_pipeline_components():
    """Test the add_pipe and remove_pipe methods.
    """

    nlp = get_test_language_model()

    # Add the pipeline components to SyferText pipeline
    nlp.add_pipe(noun_tagger, name="noun tagger")
    nlp.add_pipe(verb_tagger, name="verb tagger")

    # Note: Tokenizer is always the first component in any pipeline and
    # is added by default to the nlp.pipeline_template.
    # So the current state of the nlp.pipeline_template should be like this
    # nlp.pipeline_template = [{'remote': True, 'name': 'tokenizer'},
    #                          {'remote': False, 'name': 'noun tagger'},
    #                          {'remote': False, 'name': 'verb tagger'}]
    assert len(nlp.pipeline_template) == 3

    # Remove noun tagger from the pipeline
    nlp.remove_pipe(name="noun tagger")

    # Assert pipeline has two components
    assert len(nlp.pipeline_template) == 2




def test_subpipeline_is_not_recreated_in_remote_workers():
    """Test that the a subpipeline at a given index is not recreated in remote workers after it
    has been initialized once. Each worker contains a single subpipeline, with multiple components.
    """

    nlp = get_test_language_model()

    
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
    assert len(nlp.pipeline.keys()) == 1
    assert "bob" in nlp.pipeline.keys()

    # The second time a text owned by `bob` is tokenized, no new `SubPipeline`
    # objects are created. Only a new document on `bob`'s machine.
    doc2 = nlp(texts_ptr[1])
    subpipelines = [v for v in bob._objects.values() if isinstance(v, SubPipeline)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1
    assert len(documents) == 2
    assert len(nlp.pipeline.keys()) == 1

    # The first time a text owned by `alice` is tokenized, a new `SubPipeline` object
    # is created by the `nlp` object and sent to `alice`. Now the `nlp` object has
    # a second pointer to a `SubPipeline`.
    doc3 = nlp(texts_ptr[2])
    subpipelines = [v for v in alice._objects.values() if isinstance(v, SubPipeline)]
    documents = [v for v in alice._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1
    assert len(documents) == 1
    assert len(nlp.pipeline.keys()) == 2
    assert "alice" in nlp.pipeline.keys()

    # The third time a text owned by `bob` is tokenized, no new `SupPipeline`
    # objects are created. The `nlp` object still has the same previous pointer
    # to a `SubPipeline` on `bob`'s machine and `bob` now has a third document.
    doc4 = nlp(texts_ptr[3])
    subpipelines = [v for v in bob._objects.values() if isinstance(v, SubPipeline)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1
    assert len(documents) == 3
    assert len(nlp.pipeline.keys()) == 2


def test_pipeline_output():

    nlp = get_test_language_model()

    
    # Create a PySyft String and send it to remote worker james
    text_ptr = String("building SyferText").send(james)

    # Add tagger with remote = True
    tagger = SimpleTagger(attribute="noun", lookups=["SyferText"], tag=True)
    nlp.add_pipe(tagger, name="noun_tagger",access = {'*'})

    # Upon processing the text present on james's machine,
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
    print(pipes)
    assert len(pipes) == 2
    assert pipes[0] == "tokenizer"
    assert pipes[1] == "noun_tagger"

    # nlp.pipeline stores pointers to subpipeline objects on remote machines
    # assert subpipeline pointer stored in nlp.pipeline points to the subpipeline on james machine
    assert nlp.pipeline['james'][0].id_at_location == subpipelines[0].id
