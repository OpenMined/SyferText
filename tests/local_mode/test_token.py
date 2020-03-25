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
    norm = doc[0].vector_norm

    # check that norm is not zero for valid token
    assert norm != 0.0


def test_non_valid_token_norm_is_zero():
    """Test that the norm of a invalid token is zero"""
    doc = nlp("not-valid-token")
    norm = doc[0].vector_norm

    # check that norm is zero for invalid token
    assert norm == 0.0


def test_similarity_bw_valid_tokens():
    """Test that the similarity of valid tokens is not zero"""
    doc = nlp("hello banana")
    token1 = doc[0]
    token2 = doc[1]

    sim = token1.similarity(token2)

    # check that similarity is not zero for two valid unequal token
    assert sim != 0.0


def test_similarity_bw_same_valid_tokens():
    """Test that the similarity of valid equal tokens is one"""
    doc = nlp("banana banana")
    token1 = doc[0]
    token2 = doc[1]

    sim = token1.similarity(token2)

    # check that norm is one for equal valid tokens
    assert sim == 1.0
