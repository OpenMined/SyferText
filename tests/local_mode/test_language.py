import syft as sy
import torch
import syfertext
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_sm", owner=me)


def test_vector_valid_token_is_not_zero():
    """Test that the vector of a valid token is not all zeros"""
    doc = nlp("love")
    actual = doc[0].vector
    zeros = np.zeros(actual.shape)
    # check that at least one cell in actual vector is not zero
    assert (actual != zeros).any() == True


def test_vector_non_valid_token_is_zero():
    """Test that the vector of non valid token is all zeros"""
    doc = nlp("non-valid-token")
    actual = doc[0].vector
    zeros = np.zeros(actual.shape)
    # check that all cells in actual vector are zeros
    assert (actual == zeros).all() == True

if __name__ == "__main__":
    test_vector_valid_token_is_not_zero()
    test_vector_non_valid_token_is_zero()