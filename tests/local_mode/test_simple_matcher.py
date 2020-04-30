import pytest
import syft
import torch
import syfertext
from syfertext.matcher import SimpleMatcher
from syfertext.pipeline.simple_tagger import SimpleTagger

# Create a torch hook for PySyft
hook = syft.TorchHook(torch)

# Get a reference to the local worker
me = hook.local_worker


def test_addition_and_deletion_of_patterns():

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)
    shared_vocab = nlp.vocab

    matcher = SimpleMatcher(vocab=shared_vocab)

    pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]

    # Note: Pattern needs to be a list of lists
    matcher.add("helloworld", [pattern])

    assert "helloworld" in matcher

    patterns = [
        [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "nlp"}],
        [{"LOWER": "hello"}, {"LOWER": "nlp"}],
    ]
    matcher.add("hellonlp", patterns)

    assert "hellonlp" in matcher

    matcher.remove("helloworld")

    # Assert pattern has been removed from matcher
    assert "helloworld" not in matcher

    matcher.remove("hellonlp")

    # Assert pattern has been removed from matcher
    assert "hellonlp" not in matcher


def test_preprocess_patterns():

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)

    # For tagging punctuations
    punctuation_tagger = SimpleTagger("is_punct", [",", "."], tag=True)
    nlp.add_pipe(punctuation_tagger, name="punct_tagger")

    doc = nlp("Hello, nlp! Hello nlp!")

    # Set lower attributes to all the tokens
    # TODO: Can this be done using SimpleTagger ?
    for token in doc:
        token.set_attribute("lower", token.text.lower)

    # Initialize the matcher class
    matcher = SimpleMatcher(nlp.vocab)

    patterns = [
        [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "nlp"}],
        [{"LOWER": "hello"}, {"LOWER": "nlp"}],
    ]

    # Add patterns to the matcher
    matcher.add("hellonlp", patterns)

    # Find the matches in the doc
    matches = matcher(doc)

    true_matches = [
        ("hellonlp", 0, 3),  # 0: "Hello", 1: ",", 2: "nlp"
        # Matching the first pattern in patterns
        ("hellonlp", 4, 6),  # 4: "Hello", 5: "nlp"
        # Matching the second pattern in patterns
    ]

    for (match_id, start, end), true_match in zip(matches, true_matches):

        string_id = nlp.vocab.store[match_id]  # Get string representation

        assert (string_id, start, end) == true_match


def test_callback_on_match():

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)

    # For tagging pronouns
    pronoun_tagger = SimpleTagger("pro_noun", ["I", "He", "She"], tag=True)
    nlp.add_pipe(pronoun_tagger, name="pron_tagger")

    doc1 = nlp("I love to code")
    doc2 = nlp("He hates companies which invade the privacy of it's users")  # facebook, google etc.

    # TODO: Can this be done using SimpleTagger ?
    for token in doc1:
        token.set_attribute("lower", token.text.lower())

    for token in doc2:
        token.set_attribute("lower", token.text.lower())

    positive_docs = [doc1]
    negative_docs = [doc2]

    pos_patterns = [{"PRO_NOUN": True}, {"LOWER": "love"}]
    neg_patterns = [{"PRO_NOUN": True}, {"LOWER": "hates"}]  # After adding lemmatizer,
    # We can just have {"LEMMA", "hate"}

    def tag_doc_sentiment(matcher, doc, i, matches):
        """Callback function to be passed to matcher."""

        key, start, end = matches[i]
        if nlp.vocab.store[key] == "positive":
            doc.set_attribute("sentiment", 1.0)

        elif nlp.vocab.store[key] == "negative":
            doc.set_attribute("sentiment", -1.0)

    matcher = SimpleMatcher(nlp.vocab)

    # Add patterns to the matcher, along with callback function
    matcher.add("positive", [pos_patterns], on_match=tag_doc_sentiment)
    matcher.add("negative", [neg_patterns], on_match=tag_doc_sentiment)

    # Find the matches in the doc
    matcher(doc1)
    matcher(doc2)

    for doc in positive_docs:

        # assert callback function sets the right sentiment value
        assert doc._.sentiment > 0

    for doc in negative_docs:

        # assert callback function sets the right sentiment value
        assert doc._.sentiment < 0
