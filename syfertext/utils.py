import mmh3
import os
import re
from pathlib import Path

from typing import Pattern
from typing import Match
from typing import Tuple

import tempfile
import shutil


def hash_string(string: str) -> int:
    """Create a hash for a given string. 
    Hashes created by this functions will be used everywhere by
    SyferText to represent tokens.
    """

    key = mmh3.hash64(string, signed=False, seed=1)[0]

    return key


# The following three functions for compiling prefix, suffix and infix regex are adapted
# from Spacy  https://github.com/explosion/spaCy/blob/master/spacy/util.py.
def compile_prefix_regex(entries: Tuple) -> Pattern:
    """Compile a sequence of prefix rules into a regex object.

    Args:
        entries (tuple): The prefix rules, e.g. syfertext.punctuation.TOKENIZER_PREFIXES.

    Returns:
        The regex object. to be used for Tokenizer.prefix_search.
    """

    if "(" in entries:
        # Handle deprecated data
        expression = "|".join(["^" + re.escape(piece) for piece in entries if piece.strip()])
        return re.compile(expression)
    else:
        expression = "|".join(["^" + piece for piece in entries if piece.strip()])
        return re.compile(expression)


def compile_suffix_regex(entries: Tuple) -> Pattern:
    """Compile a sequence of suffix rules into a regex object.
    
    Args:
        entries (tuple): The suffix rules, e.g. syfertext.punctuation.TOKENIZER_SUFFIXES.

    Returns:
        The regex object. to be used for Tokenizer.suffix_search.
    """

    expression = "|".join([piece + "$" for piece in entries if piece.strip()])
    return re.compile(expression)


def compile_infix_regex(entries: Tuple) -> Pattern:
    """Compile a sequence of infix rules into a regex object.

    Args:
        entries (tuple): The infix rules, e.g. syfertext.punctuation.TOKENIZER_INFIXES.

    Returns:
        The regex object. to be used for Tokenizer.infix_finditer.
    """

    expression = "|".join([piece for piece in entries if piece.strip()])
    return re.compile(expression)
    prog_bar.close()


def normalize_slice(length, start, stop, step=None):

    assert step is None or step == 1, "Not a valid Slice"

    if start is None:
        start = 0

    elif start < 0:
        start += length

    start = min(length, max(0, start))

    if stop is None:
        stop = length

    elif stop < 0:
        stop += length

    stop = min(length, max(start, stop))

    assert start < stop, "Empty range"

    return start, stop