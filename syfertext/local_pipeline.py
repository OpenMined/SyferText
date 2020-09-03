import random
import numpy as np
import syfertext

from syfertext.tokenizer import Tokenizer
from syfertext.vocab import Vocab
from syfertext.utils import hash_string
from syfertext.test_vocab import TEST_VOCAB


def get_test_language_model():
    nlp = syfertext.create(model_name="syfertext_sentiment")
    # Create the tokenizer
    tokenizer = Tokenizer()

    # total number of words taken
    vocab_size = len(TEST_VOCAB)
    # creating index for vocab
    indexes = list(range(vocab_size))
    # create hash2row empty dict
    hash2row = {}
    # create hash2row dict
    for i in range(vocab_size):
        index = random.choice(indexes)
        indexes.remove(index)
        hash2row[hash_string(TEST_VOCAB[index])] = index

    # create vectors from words taken
    vectors = np.random.rand(vocab_size, 300)
    # create vocab from hash2row and vectors
    vocab = Vocab(hash2row=hash2row, vectors=vectors)
    nlp.set_tokenizer(tokenizer, access={"*"})

    nlp.set_vocab(vocab, access={"*"})

    return nlp
