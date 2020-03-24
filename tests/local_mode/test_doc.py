import syft as sy
import torch
import pytest
import syfertext
from syfertext.pipeline import SimpleTagger

hook = sy.TorchHook(torch)
me = hook.local_worker

nlp = syfertext.load("en_core_web_lg", owner=me)

dict_lookups = [
    ({"on": "preposition", "your": "pronoun", "left": "noun"}, True, {"custom": "noun"}),
    ({"on": "preposition", "your": "pronoun", "left": "noun"}, False, {"custom": "noun"}),
]


@pytest.mark.parametrize("lookups, with_get_vector, exclude", dict_lookups)
def test_avg_vector_valid_token(lookups, with_get_vector, exclude):
    """Test that the average vector of valid tokens match the expected result"""

    if with_get_vector:
        doc = nlp("on your left")
    else:
        doc = nlp("on your left", exclude)

    tagger = SimpleTagger("custom", lookups, default_tag="test")
    tagger(doc)

    if with_get_vector:
        actual = doc.get_vector(excluded_tokens=exclude)
    else:
        actual = doc.vector

    vectors = None

    # Count the tokens that have vectors
    vector_count = 0

    for token in doc:
        values = [getattr(token._, attribute, "") for attribute in exclude.keys()]

        bools = [value == exclude_value for value, exclude_value in zip(values, exclude.values())]
        if any(bools):
            continue

        # Get the vector of the token if one exists
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

        # Get the vector of the token if one exists
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


def test_add_custom_attr_doc():
    """Test adding custom attribute to Doc objects"""

    doc = nlp("Joey doesnt share food")

    # add new custom attribute
    doc.set_attribute(name="my_custom_tag", value="tag")

    # check custom attribute has been added
    assert hasattr(doc._, "my_custom_tag") and doc._.my_custom_tag == "tag"


def test_update_custom_attr_doc():
    """Test updating custom attribute of Doc objects"""

    doc = nlp("Joey doesnt share food")

    # add new custom attribute
    doc.set_attribute(name="my_custom_tag", value="tag")

    # check custom attribute has been added
    assert hasattr(doc._, "my_custom_tag") and doc._.my_custom_tag == "tag"

    # now update the attribute
    doc.set_attribute(name="my_custom_tag", value="new_tag")

    # now check the updated attribute
    assert hasattr(doc._, "my_custom_tag") and doc._.my_custom_tag == "new_tag"
