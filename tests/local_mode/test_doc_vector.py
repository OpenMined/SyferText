import syft as sy
import torch
import syfertext
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_avg_vector_valid_token_is_not_zero():
    """Test that the average vector of valid tokens is not all zeros"""
    doc = nlp("on your left")
    actual = doc.vector
    zeros = np.zeros(actual.shape)
    # check that at least one cell in actual average vector is not zero
    assert (actual != zeros).any() == True


def test_avg_vector_non_valid_token_is_zero():
    """Test that the average vector of non valid token is all zeros"""
    doc = nlp("non-valid-token")
    actual = doc.vector
    zeros = np.zeros(actual.shape)
    # check that all cells in actual average vector are zeros
    assert (actual == zeros).all() == True
