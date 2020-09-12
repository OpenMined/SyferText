import pytest
import syft
import torch
import syfertext
from syfertext.pipeline import SimpleTagger
from syfertext.local_pipeline import get_test_language_model

# Create a torch hook for PySyft
hook = syft.TorchHook(torch)

# Get a reference to the local worker
me = hook.local_worker
me.is_client_worker = False

# Get a SyferText Language object
nlp = get_test_language_model()

# Define a text to tag
text = "The quiCk broWn Fox jUmps over thE lazY Dog . I will tokenizE thiS phrase wiTh SyferText . I Will do it myselF"

# Create dictionary lookups
dict_lookups = [
    ({"The": True, "myselF": True, "jumps": "verb", "with": "prep"}, False, True),
    ({"The": True, "myselF": True, "jumps": "verb", "with": "prep"}, False, False),
    ({}, "tag me", True),  # This is supposed to tag all words with the same tag 'tag me'
    ({}, "tag me", False),  # This is supposed to tag all words with the same tag 'tag me'
]

list_set_lookups = [
    (["The", "myselF", "jumps", "with"], True, False, True),
    (["The", "myselF", "jumps", "with"], "hey", False, False),
    ([], "tag1", "tag2", True),
    ([], "tag1", "tag2", False),
    ({"The", "myselF", "jumps", "with"}, True, False, True),
    ({"The", "myselF", "jumps", "with"}, "hey", False, False),
    ({}, "tag1", "tag2", True),
    ({}, "tag1", "tag2", False),
]


@pytest.mark.parametrize("lookups,default_tag,case_sensitive", dict_lookups)
def test_simple_tagger_with_dict(lookups, default_tag, case_sensitive):

    # Create the document
    doc = nlp(text)

    # Create the tagger
    tagger = SimpleTagger(
        attribute="custom_tag",
        lookups=lookups,
        case_sensitive=case_sensitive,
        tag="don't care",  # This will be ignored since a dict lookup is used
        default_tag=default_tag,
    )

    # Tag the document
    tagger(doc)

    # Adjust the lookups for case insensitive cases
    if not case_sensitive:
        lookups = {key.lower(): lookups[key] for key in lookups}

    # Check if tags were successfully stored in the specified attribute
    for token in doc:

        # Get the token text
        token_text = token.text if case_sensitive else token.text.lower()

        if token_text in lookups:
            assert token._.custom_tag == lookups[token_text]
        else:
            assert token._.custom_tag == default_tag


@pytest.mark.parametrize("lookups,tag,default_tag,case_sensitive", list_set_lookups)
def test_simple_tagger_with_lists_sets(lookups, tag, default_tag, case_sensitive):

    # Create the document
    doc = nlp(text)

    # Create the tagger
    tagger = SimpleTagger(
        attribute="custom_tag",
        lookups=lookups,
        case_sensitive=case_sensitive,
        tag=tag,
        default_tag=default_tag,
    )

    # Tag the document
    tagger(doc)

    # Adjust the lookups for case insensitive cases
    if not case_sensitive:
        lookups = {string.lower() for string in lookups}

    # Check if tags were successfully stored in the specified attribute
    for token in doc:

        # Get the token text
        token_text = token.text if case_sensitive else token.text.lower()

        if token_text in lookups:
            assert token._.custom_tag == tag
        else:
            assert token._.custom_tag == default_tag
