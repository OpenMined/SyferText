import syft as sy
import torch
import syfertext
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_valid_token_norm_is_not_zero():
    """Test that the norm of a valid token is not zero"""

    doc = nlp("banana")
    norm = doc[0].vector_norm.item()

    # check that norm is not zero for valid token
    assert norm != 0.0


def test_oov_token_norm_is_zero():
    """Test that the norm of an out-of-vocab oken is zero"""

    doc = nlp("notvalidtoken")
    norm = doc[0].vector_norm.item()

    # check that norm is zero for invalid token
    assert norm == 0.0


def test_similarity_tokens():
    """Test that the similarity of valid tokens"""

    doc = nlp("hello banana")
    token1 = doc[0]
    token2 = doc[1]

    # check commutativity of similarity
    assert token1.similarity(token2) == token2.similarity(token1)

    # check if similarity is in valid range
    assert -1.0 <= token1.similarity(token2).item() <= 1.0
