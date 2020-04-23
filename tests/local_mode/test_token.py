import syft as sy
import torch
import syfertext

from syft.generic.string import String

from syfertext.doc import Doc
from syfertext.span import Span
from syfertext.token import Token
from syfertext.pointers.doc_pointer import DocPointer
from syfertext.pointers.span_pointer import SpanPointer
from syfertext.pointers.token_pointer import TokenPointer

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_remote_token_from_doc():
    """Test that token is initialized at remote worker and TokenPointer is returned from a DocPointer"""

    bob = sy.VirtualWorker(hook, id="bob")

    # Send doc to a remote machine
    remote_text = String("the quick brown fox jumps over a lazy dog").send(bob)
    doc = nlp(remote_text)

    token = doc[1]

    # Assert TokenPointer is returned
    assert isinstance(token, TokenPointer)

    # check the length token text is 5
    assert len(token) == 5

    # Assert only one Token object on bob's machine
    tokens = [v for v in bob._objects.values() if isinstance(v, Token)]
    assert len(tokens) == 1

    # Assert returned TokenPointer points to Token object on bob's machine
    assert token.id_at_location == tokens[0].id


def test_remote_token_from_span():
    """Test that token is initialized at remote worker and TokenPointer is returned from a SpanPointer"""

    alice = sy.VirtualWorker(hook, id="alice")

    # Send doc to a remote machine
    remote_text = String("the quick brown fox jumps over a lazy dog").send(alice)
    doc = nlp(remote_text)

    span = doc[1:]

    token = span[0]

    # Assert TokenPointer is returned
    assert isinstance(token, TokenPointer)

    # check the length token text is 5
    assert len(token) == 5

    # Assert only one Token object on alice's machine
    tokens = [v for v in alice._objects.values() if isinstance(v, Token)]
    assert len(tokens) == 1

    # Assert returned TokenPointer points to Token object on alice's machine
    assert token.id_at_location == tokens[0].id
