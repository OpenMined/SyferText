import syft as sy
import syfertext
import torch
from syft.generic.string import String
from syfertext.doc import Doc
from syfertext.pipeline import SubPipeline


hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_subpipeline_is_not_recreated_in_remote_workers():
    """Test that the a subpipeline at a given index is not recreated in remote workers after it
    has been initialized once.
    """
    alice = sy.VirtualWorker(hook, id="alice")
    bob = sy.VirtualWorker(hook, id="bob")

    # Create 4 PySyft Strings and send them to remote workers
    # (3 to Bob, 1 to Alice)
    texts = [String(text) for text in ["hello", "syfertext", "private", "nlp"]]

    texts_ptr = [
        texts[0].send(bob),
        texts[1].send(bob),
        texts[2].send(alice),
        texts[3].send(bob),
    ]

    # The first time a text owned by `bob` is tokenized, a `SubPipeline` object is
    # created by the `nlp` object and sent to `bob`. The `nlp` object keeps a
    # pointer to the tokenizer object in `bob`'s machine.
    doc1 = nlp(texts_ptr[0])
    subpipelines = [v for v in bob._objects.values() if isinstance(v, SubPipeline)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1
    assert len(documents) == 1
    assert len(nlp.pipeline[0].keys()) == 1

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
    
    # The third time a text owned by `bob` is tokenized, no new `SupPipeline`
    # objects are created. The `nlp` object still has the same previous pointer
    # to a `SubPipeline` on `bob`'s machine and `bob` now has a third document.
    doc4 = nlp(texts_ptr[3])
    subpipelines = [v for v in bob._objects.values() if isinstance(v, SubPipeline)]    
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(subpipelines) == 1    
    assert len(documents) == 3
    assert len(nlp.pipeline[0].keys()) == 2
