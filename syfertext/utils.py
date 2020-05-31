import mmh3
import os
import re
from pathlib import Path

from typing import Pattern
from typing import Match
from typing import Tuple

import tempfile
import shutil

from quickumls_simstring import simstring
import unicodedata

import math


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

def safe_unicode(s):
    return u'{}'.format(unicodedata.normalize('NFKD', s))

class SimstringDBReader(object):
    """Used to get candidates for medical extraction. Uses CPMerge
    Alogorithm to find approximate matches.

    Inspired from : https://github.com/Georgetown-IR-Lab/QuickUMLS/blob/master/quickumls/toolbox.py
    For more info refer to paper : <CPmerge paper>
    """
    def __init__(self, database_umls : str, similarity_name : str, threshold : float):
        self.db = simstring.reader(database_umls)
        self.db.measure = getattr(simstring, similarity_name)
        self.db.threshold = threshold

    def get(self, term):
        term = safe_unicode(term)
        return self.db.retrieve(term)

def make_ngrams(s, n):
    # s = u'{t}{s}{t}'.format(s=safe_unicode(s), t=('$' * (n - 1)))
    n = len(s) if len(s) < n else n
    return (s[i:i + n] for i in xrange(len(s) - n + 1))


def get_similarity(x, y, n, similarity_name):
    if len(x) == 0 or len(y) == 0:
        # we define similarity between two strings
        # to be 0 if any of the two is empty.
        return 0.

    X, Y = set(make_ngrams(x, n)), set(make_ngrams(y, n))
    intersec = len(X.intersection(Y))

    if similarity_name == 'dice':
        return 2 * intersec / (len(X) + len(Y))
    elif similarity_name == 'jaccard':
        return intersec / (len(X) + len(Y) - intersec)
    elif similarity_name == 'cosine':
        return intersec / math.sqrt(len(X) * len(Y))
    elif similarity_name == 'overlap':
        return intersec
    else:
        msg = 'Similarity {} not recognized'.format(similarity_name)
        raise TypeError(msg)

class Intervals(object):
    def __init__(self):
        self.intervals = []

    def _is_overlapping_intervals(self, a, b):
        if b[0] < a[1] and b[1] > a[0]:
            return True
        elif a[0] < b[1] and a[1] > b[0]:
            return True
        else:
            return False

    def __contains__(self, interval):
        return any(
            self._is_overlapping_intervals(interval, other)
            for other in self.intervals
        )

    def append(self, interval):
        self.intervals.append(interval)
