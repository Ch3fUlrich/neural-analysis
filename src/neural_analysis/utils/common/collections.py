def flatten(l: list) -> list:
    return [item for sublist in l for item in sublist]

def unique(l: list) -> list:
    return list(set(l))

def is_array_like(obj) -> bool:
    return hasattr(obj, '__iter__') and not isinstance(obj, (str, bytes))

def make_list_ifnot(obj) -> list:
    if isinstance(obj, list):
        return obj
    elif isinstance(obj, tuple):
        return list(obj)
    return [obj]

def mean_diff(x, y, axis=0):
    import numpy as np
    return np.mean(x, axis=axis) - np.mean(y, axis=axis)

def do_critical(exception_type: Exception, message: str, logger=None):
    if logger:
        logger.critical(message)
    raise exception_type(message)
