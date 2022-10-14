from functools import wraps


def squeeze_fn(seq):
    """Extracts the item from a sequence if it is the only one, otherwise leaves the sequence as is.

    Args:
        seq: Sequence to process.

    Returns:
        Single item from the sequence, or sequence itself it has more than one item.
    """
    if len(seq) == 1:
        (seq,) = seq
    return seq


def squeeze(fn):
    """Decorator for functions that return sequences of possibly one item, where we would want the lone item directly.

    Args:
        fn: Function that returns a sequence.

    Returns:
        Function that returns a single item from the sequence, or the sequence itself it has more than one item.
    """

    @wraps(fn)
    def unpack_single_item_return(*args, **kwargs):
        return squeeze_fn(fn(*args, **kwargs))

    return unpack_single_item_return