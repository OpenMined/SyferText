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
