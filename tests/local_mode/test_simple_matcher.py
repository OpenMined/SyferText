import pytest
import syft
import torch
import syfertext
from syfertext.matcher import SimpleMatcher

# Create a torch hook for PySyft
hook = syft.TorchHook(torch)

# Get a reference to the local worker
me = hook.local_worker

# Get a SyferText Language object
nlp = syfertext.load("en_core_web_lg", owner=me)

shared_vocab = nlp.vocab


def test_addition_and_deletion_of_patterns():

    matcher = SimpleMatcher(vocab=shared_vocab)

    pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]

    # Note: pattern needs to be a list of lists
    matcher.add("helloworld", [pattern])

    assert "helloworld" in matcher

    patterns = [
        [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "nlp"}],
        [{"LOWER": "hello"}, {"LOWER": "nlp"}],
    ]
    matcher.add("hellonlp", patterns)

    assert "hellonlp" in matcher

    new = matcher.get("hellonlp")

    assert "helloworld" not in matcher

    matcher.remove("hellonlp")

    assert "hellonlp" not in matcher


def test_preprocess_patterns():

    matcher = SimpleMatcher(shared_vocab)

    patterns = [
        [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "nlp"}],
        [{"LOWER": "hello"}, {"LOWER": "nlp"}],
    ]

    # Add patterns to the matcher
    matcher.add("helloworld", patterns)

    doc = nlp("Hello, nlp! Hello nlp!")

    # tagging tokens
    for token in doc:
        token.set_attribute("lower", value=token.text.lower())

    matches = matcher(doc)

    for match_id, start, end in matches:
        string_id = nlp.vocab.store[match_id]  # Get string representation
        # span = doc[start:end]  # The matched span
        print(match_id, string_id, start, end)
