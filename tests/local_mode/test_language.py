import os
import syft as sy
import torch
import syfertext
from pathlib import Path
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


def test_language_model_download_path():
    """Tests that downloading a model returns a valid & correct path"""

    # Retrieve the model's path from its vocabulary
    model_path = nlp.vocab.model_path

    # Set the expected path for the `en_core_web_lg` model
    expected_model_path = os.path.join(str(Path.home()), "SyferText", "en_core_web_lg")

    # check if the model path exists
    assert os.path.exists(model_path)

    # check if the model path is the same as the expected path
    assert model_path == expected_model_path
