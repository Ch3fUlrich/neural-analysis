def flatten(l: list) -> list:  # type: ignore
    return [item for sublist in l for item in sublist]


def unique(l: list) -> list:  # type: ignore
    return list(set(l))


def is_array_like(obj) -> bool:  # type: ignore
    return hasattr(obj, "__iter__") and not isinstance(obj, (str, bytes))


def make_list_ifnot(obj) -> list:  # type: ignore
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    return [obj]


def mean_diff(x, y, axis=0):  # type: ignore
    import numpy as np

    return np.mean(x, axis=axis) - np.mean(y, axis=axis)


def do_critical(exception_type: Exception, message: str, logger=None):  # type: ignore
    if logger:
        logger.critical(message)
    raise exception_type(message)  # type: ignore
