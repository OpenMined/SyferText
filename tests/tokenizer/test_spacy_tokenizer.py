import pytest
import syfertext.tokenizers as tokenizer
import os


def test_tokenizer_handles_no_word(tokenizer_spacy):
    tokens = tokenizer_spacy("")
    assert len(tokens) == 0


@pytest.mark.parametrize("text", ["lorem"])
def test_tokenizer_handles_single_word(tokenizer_spacy, text):
    tokens = tokenizer_spacy(text)
    assert tokens[0].text == text


def test_tokenizer_handles_punct(tokenizer_spacy):
    text = "Lorem, ipsum."
    tokens = tokenizer_spacy(text)
    assert len(tokens) == 4
    assert tokens[0].text == "Lorem"
    assert tokens[1].text == ","
    assert tokens[2].text == "ipsum"
    assert tokens[1].text != "Lorem"


def test_tokenizer_handles_punct_braces(tokenizer_spacy):
    text = "Lorem, (ipsum)."
    tokens = tokenizer_spacy(text)
    assert len(tokens) == 6


def test_tokenizer_handles_digits(tokenizer_spacy):
    exceptions = ["hu", "bn"]
    text = "Lorem ipsum: 1984."
    tokens = tokenizer_spacy(text)

    assert len(tokens) == 5
    assert tokens[0].text == "Lorem"
    assert tokens[3].text == "1984"


@pytest.mark.parametrize(
    "text",
    ["google.com", "python.org", "spacy.io", "explosion.ai", "http://www.google.com"],
)
def test_tokenizer_keep_urls(tokenizer_spacy, text):
    tokens = tokenizer_spacy(text)
    assert len(tokens) == 1


@pytest.mark.parametrize("text", ["NASDAQ:GOOG"])
def test_tokenizer_colons(tokenizer_spacy, text):
    tokens = tokenizer_spacy(text)
    assert len(tokens) == 3


@pytest.mark.parametrize(
    "text", ["hello123@example.com", "hi+there@gmail.it", "matt@explosion.ai"]
)
def test_tokenizer_keeps_email(tokenizer_spacy, text):
    tokens = tokenizer_spacy(text)
    assert len(tokens) == 1


def test_tokenizer_handles_long_text(tokenizer_spacy):
    text = """Lorem ipsum dolor sit amet, consectetur adipiscing elit
Cras egestas orci non porttitor maximus.
Maecenas quis odio id dolor rhoncus dignissim. Curabitur sed velit at orci ultrices sagittis. Nulla commodo euismod arcu eget vulputate.
Phasellus tincidunt, augue quis porta finibus, massa sapien consectetur augue, non lacinia enim nibh eget ipsum. Vestibulum in bibendum mauris.
"Nullam porta fringilla enim, a dictum orci consequat in." Mauris nec malesuada justo."""

    tokens = tokenizer_spacy(text)
    assert len(tokens) > 5


@pytest.mark.parametrize("file_name", ["tests/tokenizer/sun.txt"])
def test_tokenizer_handle_text_from_file(tokenizer_spacy, file_name):
    text = open(file_name, "r", encoding="utf8").read()
    assert len(text) != 0
    tokens = tokenizer_spacy(text)
    assert len(tokens) > 100


def test_tokenizer_suspected_freeing_strings(tokenizer_spacy):
    text1 = "Lorem dolor sit amet, consectetur adipiscing elit."
    text2 = "Lorem ipsum dolor sit amet, consectetur adipiscing elit."
    tokens1 = tokenizer_spacy(text1)
    tokens2 = tokenizer_spacy(text2)
    assert tokens1[0].text == "Lorem"
    assert tokens2[0].text == "Lorem"
