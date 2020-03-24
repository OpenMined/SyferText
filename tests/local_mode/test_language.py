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

    doc = nlp("I 'love  app-le!")
    actual_tokens = ["I", "'", "love", " ", "app", "-", "le", "!"]

    # check that tokenizer is correctly tokenizing the text.
    for i, token_created in enumerate(doc):
        assert str(token_created) == actual_tokens[i]

    # check correct infix are tokenized only
    str1 = "Hell#o"  # this string does not contains infix.
    str2 = "Hell-o"  # This string contains logical infix.

    assert len(nlp(str1)) == 1  # ['Hell#o']
    assert len(nlp(str2)) == 3  # ['Hell', '-', 'o']

    # check exception cases are tokenized correctly
    excep = "U.S.A"

    assert len(nlp(excep)) == 1  # ['U.S.A']
