import functools
from enum import Enum
from functools import singledispatch
from time import time


def clock(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(f"Finished {func.__name__} in {end - start:.2f} seconds.")
        return result

    return wrapper
