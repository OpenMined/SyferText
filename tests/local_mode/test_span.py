import syft as sy
import torch
import syfertext

import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_basic_span():

    doc = nlp("the quick brown fox jumps over lazy dog")

    span = doc[1:5]
    actual_token = ["quick", "brown", "fox", "jumps"]

    assert len(span) == 4

    for i, token in enumerate(span):
        assert token.text == actual_token[i]


def test_span_of_span():

    doc = nlp("the quick brown fox jumps over lazy dog")

    span_ = doc[1:5]

    span = span_[1:3]

    actual_token = ["brown", "fox"]

    assert len(span) == 2

    for i, token in enumerate(span):
        assert token.text == actual_token[i]
