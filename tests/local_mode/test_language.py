import os
import syft as sy
import torch
import syfertext
from pathlib import Path
from syfertext.local_pipeline import get_test_language_model

hook = sy.TorchHook(torch)
me = hook.local_worker
me.is_client_worker = False

nlp = get_test_language_model()


def test_vector_valid_token_is_not_zero():
    """Test that the vector of a valid token is not all zeros"""

    doc = nlp("possible")
    actual = doc[0].vector
    zeros = torch.zeros(actual.shape)

    # check that at least one cell in actual vector is not zero
    assert (actual != zeros).any() == True


def test_vector_non_valid_token_is_zero():
    """Test that the vector of non valid token is all zeros"""

    doc = nlp("outofvocabularytoken")
    actual = doc[0].vector
    zeros = torch.zeros(actual.shape)

    # check that all cells in actual vector are zeros
    assert (actual == zeros).all() == True


def test_length_of_doc():
    """Test that the tokenizer creates the right number of tokens."""

    doc = nlp("I #love  app-le.")
    actual = 8
    tokens_created = len(doc)

    # check that all cells in actual vector are zeros
    assert actual == tokens_created


def test_correctness_of_tokens_created():
    """Test that tokens created by the tokenizer are correct"""

    doc = nlp("I 'love  app-le!")
    actual_tokens = ["I", "'", "love", " ", "app", "-", "le", "!"]

    # check that tokenizer is correctly tokenizing the text.
    for i, token_created in enumerate(doc):
        assert str(token_created) == actual_tokens[i]

    # check correct infix are tokenized only
    str1 = "Hell#o"  # this string does not contain infix.
    str2 = "Hell-o"  # This string contains logical infix.

    assert len(nlp(str1)) == 1  # ['Hell#o']
    assert len(nlp(str2)) == 3  # ['Hell', '-', 'o']

    # check exception cases are tokenized correctly
    excep = "U.S.A"

    assert len(nlp(excep)) == 1  # ['U.S.A']
