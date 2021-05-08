"""
    Dummy conftest.py for syfertext.

    If you don't know what this is for, just leave it empty.
    Read more about conftest.py under:
    - https://docs.pytest.org/en/stable/fixture.html
    - https://docs.pytest.org/en/stable/writing_plugins.html
"""

import pytest
import syfertext.tokenizers as tokenizers

@pytest.fixture(scope="module")
def tokenizer_spacy():
    return tokenizers.SpacyTokenizer()