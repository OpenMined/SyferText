import syft as sy
import torch
import syfertext

from syfertext.doc import Doc
from syfertext.span import Span

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
    actual_tokens = ["over", "lazy", "dog"]

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
    """Test the `as_doc()` method of span"""

    span = doc[1:5]

    doc_ = span.as_doc()

    assert isinstance(doc_, Doc)

    actual_tokens = ["quick", "brown", "fox", "jumps"]

    for token, actual_token in zip(doc_, actual_tokens):
        assert token.text == actual_token

