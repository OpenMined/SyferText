import syft
import torch
import utils
import syfertext
from syfertext.string_store import StringStore

# Create a torch hook for PySyft
hook = syft.TorchHook(torch)

# Initialise a StringStore object with a list of strings
strings = ["I", "have", "an", "apple"]
store = StringStore(strings=strings)


def test_add_init_strings_to_store():

    # Iterate over words in strings
    for word in strings:

        # Check if word is added to store
        assert word in store

        hash_key = store[word]

        # Check hash value for corresponding word
        assert isinstance(hash_key, int)


def test_add_new_string_to_store():

    new_string = "banana"

    # Add new string
    key = store.add(new_string)

    # Check if appropriate hash_key returned
    assert isinstance(key, int)

    # Check if word is added to store
    assert new_string in store


def test_add_string_if_string_not_in_store():

    string_not_in_store = "string_not_in_store"
    key = store[string_not_in_store]

    # Check if appropriate hash_key returned
    assert isinstance(key, int)

    # Check if corresponding string is added to store
    assert string_not_in_store in store


def test_get_string_for_key():

    string = "get_me_back_using_key"
    key = store[string]

    # Check if appropriate hash_key returned
    assert isinstance(key, int)

    returned_string = store[key]

    # Check if both strings are same
    assert string == returned_string


def test_dynamic_store_size():
    """Tests size of StringStore in vocab after creating doc"""

    me = hook.local_worker
    nlp = utils.get_test_language_model()

    # Check that no string is present in store
    assert len(nlp.vocab.store) == 0

    doc = nlp("quick brown fox jumps")

    # Check that 4 strings have been added in store
    assert len(nlp.vocab.store) == 4
