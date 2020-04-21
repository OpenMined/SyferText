import syft as sy
import torch
import syfertext

from syfertext.doc import Doc
from syfertext.span import Span
from syfertext.pointers.doc_pointer import DocPointer

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)
doc = nlp("the quick brown fox jumps over lazy dog")


def test_creation_of_basic_span():
    """Test the __get_item__() method of doc which returns
    a Span when passed in a slice."""

    # Test positive slicing
    span = doc[1:5]
    actual_tokens = ["quick", "brown", "fox", "jumps"]

    assert len(span) == len(actual_tokens)

    for token, actual_token in zip(span, actual_tokens):
        assert token.text == actual_token

    # Test negative slicing
    span_ = doc[-4:-1]
    actual_tokens = ["jumps", "over", "lazy", "dog"]

    assert len(span) == len(actual_tokens)

    for token, actual_token in zip(span_, actual_tokens):
        assert token.text == actual_token


def test_span_of_span():
    """Test creation of a span from another span."""

    span_ = doc[1:5]

    span = span_[1:3]

    actual_tokens = ["brown", "fox"]

    assert isinstance(span, Span)

    assert len(span) == len(actual_tokens)

    for token, actual_token in zip(span, actual_tokens):
        assert token.text == actual_token


def test_doc_from_span():
    """Test doc is returned upon calling `as_doc()`
     method of Span.
     """

    span = doc[1:5]

    doc_ = span.as_doc()

    assert isinstance(doc_, Doc)

    # Assert by default they are on same owner
    assert doc_.owner == span.owner

    actual_tokens = ["quick", "brown", "fox", "jumps"]

    for token, actual_token in zip(doc_, actual_tokens):
        assert token.text == actual_token


def test_doc_pointer_from_span():
    """Test doc is initialized on a remote machine and
    doc pointer is returned when owner passed to `as_doc()`
    is a remote machine.
    """

    bob = sy.VirtualWorker(hook, id='bob')

    span = doc[1:5]

    doc_ = span.as_doc(owner=bob)

    assert isinstance(doc_, DocPointer)

    # assert only one document object on bob's machine
    documents = [v for v in bob._objects.values() if isinstance(v, Doc)]
    assert len(documents) == 1

    # assert returned doc_pointer points to Doc object on bob's machine
    assert doc_.id_at_location == documents[0].id

