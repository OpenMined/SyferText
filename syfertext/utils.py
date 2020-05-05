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


def normalize_slice(length: int, start: int, stop: int, step: int = None):
    """This function is used to convert the negative slice boundaries to positive values.
    eg. start = -4, stop = -1, length = 6 gets converted to start = 2, stop = 5

    Args:
        length (int): the length of the document to slice
        start (int): the start index of the slice
        stop (int): the stop index of the slice
        step (int): the step value for the slice

    Returns:
        (start, stop) : pair of non-negative integer values signifying the
            normalized values of the slice
    """
    assert step is None or step == 1, "Stepped slices with steps greater than one are not supported"

    # if start is none, that means we need to start from 0 index
    if start is None:
        start = 0

    # if start is negative, we add the length to get its actual index
    elif start < 0:
        start += length

    # start should not exceed the length of the document
    # also max(0,start) ensures the start is never negative
    start = min(length, max(0, start))

    # stop is None, that means we need stop to be the last index+1
    if stop is None:
        stop = length

    # add the length to get the actual positive index for stop if
    # is negative
    elif stop < 0:
        stop += length

    # stop should be less than or equal to length. Also max(start,stop) ensures that start <= stop
    stop = min(length, max(start, stop))

    # if this occurs, that means we have an empty range.
    if start == stop:
        return 0, 0

    return start, stop


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
    return tmp_model_path
