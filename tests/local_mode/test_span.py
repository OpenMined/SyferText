import syft as sy
import torch
import syfertext

from syft.generic.string import String

from syfertext.doc import Doc
from syfertext.span import Span
from syfertext.pointers.doc_pointer import DocPointer
from syfertext.pointers.span_pointer import SpanPointer

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_creation_of_basic_span():
    """Test the __getitem__() method of doc returns
    a Span when passed in a slice."""

    doc = nlp("the quick brown fox jumps over lazy dog")

    span = doc[1:5]

    actual_tokens = ["quick", "brown", "fox", "jumps"]

    assert len(span) == len(actual_tokens)

    for token, actual_token in zip(span, actual_tokens):
        assert token.text == actual_token

    # Test negative slicing
    span_ = doc[-4:-1]

    actual_tokens = ["jumps", "over", "lazy"]

    assert len(span_) == len(actual_tokens)

    for token, actual_token in zip(span_, actual_tokens):
        assert token.text == actual_token


def test_remote_span_from_remote_doc():
    """Test span is initialized on a remote machine and a pointer
    is returned when Doc from which Span is created is on a
    remote machine."""

    bob = sy.VirtualWorker(hook, id="bob")

    # Send doc to a remote machine
    remote_text = String("the quick brown fox jumps over a lazy dog").send(bob)
    doc = nlp(remote_text)

    span = doc[1:5]

    # Assert SpanPointer is returned
    assert isinstance(span, SpanPointer)

    # check the length is 4
    assert len(span) == 4

    # Assert only one Span object on bob's machine
    spans = [v for v in bob._objects.values() if isinstance(v, Span)]
    assert len(spans) == 1

    # Assert returned SpanPointer points to Span object on bob's machine
    assert span.id_at_location == spans[0].id


def test_span_of_span():
    """Test creation of a span from another span."""

    doc = nlp("the quick brown fox jumps over lazy dog")

    span_ = doc[1:5]

    span = span_[1:3]

    actual_tokens = ["brown", "fox"]

    assert isinstance(span, Span)

    assert len(span) == len(actual_tokens)

    for token, actual_token in zip(span, actual_tokens):
        assert token.text == actual_token


def test_span_as_doc():
    """Test span is returned as Doc upon calling `as_doc()`
     method of Span.
    """
    doc_ = nlp("the quick brown fox jumps over lazy dog")

    span = doc_[1:5]

    doc = span.as_doc()

    assert isinstance(doc, Doc)

    # Assert by default they are on same owner
    assert doc.owner == span.owner

    actual_tokens = ["quick", "brown", "fox", "jumps"]

    for token, actual_token in zip(doc, actual_tokens):
        assert token.text == actual_token


def test_remote_span_as_remote_doc():
    """Test a pointer to Doc object with a copy of `Span`'s tokens
     is returned upon calling `as_doc()` on a Span residing on a remote
     machine.
    """

    james = sy.VirtualWorker(hook, id="james")

    # Send doc to a remote machine
    remote_text = String("the quick brown fox jumps over a lazy dog").send(james)
    doc_ = nlp(remote_text)

    remote_span = doc_[1:5]

    doc = remote_span.as_doc()

    # Assert a DocPointer is returned
    assert isinstance(doc, DocPointer)

    # assert two document object on bob's machine
    # One created from the String on Jame's machine and
    # other created from remote_span by `remote_span.as_doc()`
    documents = [v for v in james._objects.values() if isinstance(v, Doc)]
    assert len(documents) == 2

    # get the new doc object on james's machine
    new_doc = [x for x in documents if len(x) == 4][0]

    # assert returned doc_pointer points to Doc object on bob's machine
    assert doc.id_at_location == new_doc.id


def test_remote_span_from_remote_span():
    """Test span is initialized on a remote machine and a pointer
    is returned when a span is created from Span on a
    remote machine."""

    alice = sy.VirtualWorker(hook, id="alice")

    # Send doc to a remote machine
    remote_text = String("the quick brown fox jumps over a lazy dog").send(alice)
    doc = nlp(remote_text)

    span = doc[1:5]

    # Assert SpanPointer is returned
    assert isinstance(span, SpanPointer)

    # check the length is 4
    assert len(span) == 4

    # create a span from a span
    new_span = span[1:3]

    # Assert SpanPointer is returned
    assert isinstance(new_span, SpanPointer)

    # check the length is 2
    assert len(new_span) == 2

    # Assert only two Span objects on alice's machine
    spans = [v for v in alice._objects.values() if isinstance(v, Span)]
    assert len(spans) == 2

    # get the new span object at alice
    new_span_alice = [x for x in spans if len(x) == 2][0]

    # assert returned span_pointer points to Span object on alice's machine
    assert new_span.id_at_location == new_span_alice.id
