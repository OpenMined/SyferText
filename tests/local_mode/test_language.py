import syft as sy
import torch
import syfertext
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_vector_valid_token_is_not_zero():
    """Test that the vector of a valid token is not all zeros"""
    doc = nlp("banana")
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


def test_length_of_doc():
    """Test that tokenizer creates right number of tokens are created"""
    doc = nlp("I #love  app-le.")
    actual = 8
    tokens_created = len(doc)
    # check that all cells in actual vector are zeros
    assert actual == tokens_created


def test_correctness_of_tokens_created():
    """Test that tokens created by tokenizer are correct"""
    doc = nlp("I #love  app-le!")
    actual_tokens = ["I", "#", "love", " ", "app", "-", "le", "!"]
    for i, token_created in enumerate(doc):
        assert str(token_created) == actual_tokens[i]
print(test_length_of_doc())
print(test_correctness_of_tokens_created())