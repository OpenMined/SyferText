import pytest
import syft
import torch
import syfertext
from syfertext.matcher import Matcher
from syfertext.pipeline.simple_tagger import SimpleTagger

# Create a torch hook for PySyft
hook = syft.TorchHook(torch)

# Get a reference to the local worker
me = hook.local_worker


def test_addition_and_deletion_of_patterns():
    """Test addition and deletion of patterns in Matcher. Each pattern
    is referenced by a key assigned during adding it to the matcher."""

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)
    shared_vocab = nlp.vocab

    matcher = Matcher(vocab=shared_vocab)

    pattern = [{"LOWER": "hello"}, {"IS_PUNCT": True}, {"LOWER": "world"}]

    # Here "helloworld" is the key which references to this pattern
    matcher.add("helloworld", [pattern])  # Note: Pattern needs to be a list of lists

    # Check availability of pattern using key
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


def test_matching_patterns():
    """Test matcher to search for a basic pattern in the document."""

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)

    # For tagging punctuations
    punctuation_tagger = SimpleTagger("is_punct", [",", "."], tag=True)
    nlp.add_pipe(punctuation_tagger, name="punct_tagger")

    doc = nlp("Hello, nlp! Hello nlp!")

    # Set lower attributes to all the tokens
    # TODO: Can this be done using SimpleTagger ?
    for token in doc:
        token.set_attribute("lower", token.text.lower())

    # Initialize the matcher class
    matcher = Matcher(nlp.vocab)

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


def test_matching_regex():
    """When searching for many possible ways of representing a word/token we can pass
    in regex patterns to Matcher. This allows us to avoid adding multiple patterns to
    cover all ways of representing that token.
    """

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)

    doc = nlp("How to refer to United States of America? USA or U.S.A. or US or U.S.")

    # Set lower attributes to all the tokens
    # TODO: Can this be done using SimpleTagger ?
    for token in doc:
        token.set_attribute("lower", token.text.lower())

    # Initialize the matcher class
    matcher = Matcher(nlp.vocab)

    pattern = [
        {"LOWER": {"REGEX": "^[Uu](nited|\.?) ?[Ss](tates|\.?)( ?(of )?[Aa](merica|\.?))?$"}}
    ]

    # Add patterns to the matcher
    matcher.add("america", [pattern])

    # Find the matches in the doc
    matches = matcher(doc)

    true_matches = [
        # ("america", 4, 6),      # 4: "United", 5: "States", 6: "of", 7: "America"
        ("america", 9, 10),  # 9: "USA"
        ("america", 11, 12),  # 11: U.S.A.
        ("america", 13, 14),  # 14: US
        ("america", 15, 16),  # 15: U.S.
    ]

    for (match_id, start, end), true_match in zip(matches, true_matches):

        string_id = nlp.vocab.store[match_id]  # Get string representation

        assert (string_id, start, end) == true_match


def test_callback_on_match():
    """We can pass callback functions to match, which are called by the matcher upon a
    successful match against the patterns. This test demonstrates how it can be used to
    determine the overall sentiment of a document.
    """

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

    matcher = Matcher(nlp.vocab)

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


def test_matching_against_list_of_possible_values():
    """Matcher allows token's attributes to be compared against list of possible
    values. For example, we can search for a token with it's lemma being
    any one among 'love' or 'like' or 'respect'. This can be achieved by
    adding {"LEMMA": {"IN": ["love", "like", "respect"]}} to the patterns.
    """

    # Get a SyferText Language object
    nlp = syfertext.load("en_core_web_lg", owner=me)

    # For tagging pronouns
    pronoun_tagger = SimpleTagger("pro_noun", ["I", "He", "She"], tag=True)
    nlp.add_pipe(pronoun_tagger, name="pron_tagger")

    doc = nlp("I love to code. I like to dance")

    # TODO: Can this be done using SimpleTagger ?
    for token in doc:
        token.set_attribute("lower", token.text.lower())

    hobbies = list()

    def extract_hobbies(matcher, doc, i, matches):
        """Callback function to be passed to matcher."""

        key, start, end = matches[i]
        if nlp.vocab.store[key] == "hobbies":
            hobbies.append(doc[end].text)

    # Passing list of possible values for LOWER (Later, we can move to LEMMA, making it more powerful)
    pattern = [{"PRO_NOUN": True}, {"LOWER": {"IN": ["love", "like"]}}, {"LOWER": "to"}]

    matcher = Matcher(nlp.vocab)

    # Add patterns to the matcher, along with callback function
    matcher.add("hobbies", [pattern], on_match=extract_hobbies)

    # Find the matches in the doc
    matcher(doc)

    true_hobbies = ["code", "dance"]

    # Assert extracted hobbies and true hobbies have the same content
    assert len(hobbies) == len(true_hobbies)
    for hobby in true_hobbies:
        assert hobby in hobbies
