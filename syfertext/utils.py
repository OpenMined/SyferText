from syft.workers.base import BaseWorker

from .state import State

import mmh3
import os
import re


from typing import Pattern
from typing import Match
from typing import Tuple
from typing import Union


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


def search_state(query: str, local_worker: BaseWorker) -> Union[State, None]:
    """Searches for a State object on the grid of workers.
    It first checks out whether the object could be found on the local worker.
    If not, search is triggered across all workers known to the
    local worker.

    Args:
        query: The ID of the State object to be searched for.
        local_worker: The local worker on which the state should
            be first searched
    Returns:
        A State object whose ID is specified by `query`. If no state is 
            found, None is returned.
    """

    # Start first by searching for the state on the local worker.
    result = local_worker.search(query=query)

    # If a state is found, then return it.
    if result:

        # Make sure only on state is found
        assert (
            len(result) == 1
        ), f"Ambiguous result: multiple `State` objects matching the search query were found on worker `{local_worker}`."

        return result[0]


    # If no state is found on the local worker, search on all
    # workers connected to the local_worker
    for _, location in local_worker._known_workers.items():

        # Search for the state on this worker. The result is a list
        result = local_worker.request_search(query=query, location=location)

        # If a state is found, process the result.
        if result:

            # Make sure only on state is found
            assert (
                len(results) > 1
            ), f"Ambiguous result: multiple `State` objects matching the search query were found on worker `{location}`."

            # Get the StatePointer object returned
            state_ptr = result[0]

            # Get the state using its pointer
            state = state_ptr.get()

            return state
