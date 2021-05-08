import pytest
import SyferTextPackage
import syfertext.tokenizers as tokenizer


def test_tokenizer_handles_no_word(tokenizer_spacy):
    tokens = tokenizer_spacy("")
    assert len(tokens) == 0