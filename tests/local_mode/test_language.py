import syft as sy
import torch
import syfertext
import numpy as np
import argparse


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', default='en_core_sci_md', help='Model to be tested')
    return parser.parse_args()


def test_vector_valid_token_is_not_zero(nlp):
    """Test that the vector of a valid token is not all zeros"""
    doc = nlp("love")
    actual = doc[0].vector
    zeros = np.zeros(actual.shape)
    # check that at least one cell in actual vector is not zero
    assert (actual != zeros).any() == True


def test_vector_valid_dim(nlp):
    """Test that the vector has valid dimensions"""
    doc = nlp("love")
    print('Vector shape:', doc[0].vector.shape)


def test_vector_non_valid_token_is_zero(nlp):
    """Test that the vector of non valid token is all zeros"""
    doc = nlp("non-valid-token")
    actual = doc[0].vector
    zeros = np.zeros(actual.shape)
    # check that all cells in actual vector are zeros
    assert (actual == zeros).all() == True

if __name__ == "__main__":

    hook = sy.TorchHook(torch)
    me = hook.local_worker

    args = parse_args() # get model name
    nlp = syfertext.load(args.model, owner=me)

    # test
    test_vector_valid_dim(nlp)
    test_vector_valid_token_is_not_zero(nlp)
    test_vector_non_valid_token_is_zero(nlp)