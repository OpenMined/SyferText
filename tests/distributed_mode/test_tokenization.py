import syft as sy
import syfertext
import torch
from syft.generic.string import String
from syfertext.doc import Doc
from syfertext.tokenizer import Tokenizer


hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_tokenizer_is_not_recreated_in_remote_workers():
    """Test that the tokenizers are not recreated in remote workers after they
    have been initialized once.
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

    # The first time a text owned by `bob` is tokenized, a `Tokenizer` object is
    # created by the `nlp` object and sent to `bob`. The `nlp` object keeps a
    # pointer to the tokenizer object in `bob`'s machine.
    nlp(texts_ptr[0])
    tokenizers = [v for v in bob._objects.values() if isinstance(v, Tokenizer)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(tokenizers) == 1
    assert len(documents) == 1
    assert len(nlp.tokenizers.keys()) == 1

    # The second time a text owned by `bob` is tokenized, no new `Tokenizer`
    # objects are created. Only a new document on `bob`'s machine.
    nlp(texts_ptr[1])
    tokenizers = [v for v in bob._objects.values() if isinstance(v, Tokenizer)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(tokenizers) == 1
    assert len(documents) == 2
    assert len(nlp.tokenizers.keys()) == 1

    # The first time a text owned by `alice` is tokenized, a new Tokenizer object
    # is created by the `nlp` object and sent to `alice`. Now the `nlp` object has
    # a second pointer to a `Tokenizer`.
    nlp(texts_ptr[2])
    tokenizers = [v for v in alice._objects.values() if isinstance(v, Tokenizer)]
    documents = [v for v in alice._objects.values() if isinstance(v, Doc)]
    assert len(tokenizers) == 1
    assert len(documents) == 1
    assert len(nlp.tokenizers.keys()) == 2

    # The third time a text owned by `bob` is tokenized, no new `Tokenizer`
    # objects are created. The `nlp` object still has the same previous pointer
    # to a `Tokenizer` on `bob`'s machine and `bob` now has a third document.
    nlp(texts_ptr[3])
    tokenizers = [v for v in bob._objects.values() if isinstance(v, Tokenizer)]
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(tokenizers) == 1
    assert len(documents) == 3
    assert len(nlp.tokenizers.keys()) == 2
