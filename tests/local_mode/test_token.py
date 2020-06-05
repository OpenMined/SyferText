import syft as sy
import torch
import syfertext
from syfertext.attrs import Attributes
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_check_flag():
    """ Test the check flag method for tokens"""

    doc = nlp("token {")
    token1 = doc[0]
    token2 = doc[1]

    # check same attribute value is returned using check_flag method
    # and token attribute
    assert token1.is_digit == token1.check_flag(Attributes.IS_DIGIT)
    assert token2.is_bracket == token2.check_flag(Attributes.IS_BRACKET)


def test_set_flag():
    """ Test if you can set/update the existing token attribute"""

    doc = nlp("hello banana")
    token1 = doc[0]
    token2 = doc[1]

    # override an attribute value for a token
    token1.set_flag(flag_id=Attributes.IS_DIGIT, value=True)

    # the actual token is not digit but you can override the flag to set it True
    assert token1.is_digit


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
