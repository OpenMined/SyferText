import syft as sy
import torch
import syfertext
import numpy as np

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)


def test_avg_vector_valid_token():
    """Test that the average vector of valid tokens match the expected result"""
    doc = nlp("on your left")
    actual = doc.vector
    vectors = None

    # Count the tokens that have vectors
    vector_count = 0
    for token in doc:
        # Get the encypted vector of the token if one exists
        if token.has_vector:
            # Increment the vector counter
            vector_count += 1
            # cumulate token's vector by summing them
            vectors = token.vector if vectors is None else vectors + token.vector
    # if no tokens with vectors were found, just get the default vector (zeros)
    if vector_count == 0:
        expected_vector = doc.vocab.vectors.default_vector
    else:
        # Create the final Doc vector
        expected_vector = vectors / vector_count

    # assert that actual and expected results match
    assert (actual == expected_vector).all()


def test_avg_vector_non_valid():
    """Test that the average vector of invalid tokens match the expected result"""
    doc = nlp("non-valid-token")
    actual = doc.vector
    vectors = None

    # Count the tokens that have vectors
    vector_count = 0
    for token in doc:
        # Get the encypted vector of the token if one exists
        if token.has_vector:
            # Increment the vector counter
            vector_count += 1
            # cumulate token's vector by summing them
            vectors = token.vector if vectors is None else vectors + token.vector
    # if no tokens with vectors were found, just get the default vector (zeros)
    if vector_count == 0:
        expected_vector = doc.vocab.vectors.default_vector
    else:
        # Create the final Doc vector
        expected_vector = vectors / vector_count

    # assert that actual and expected results match
    assert (actual == expected_vector).all()