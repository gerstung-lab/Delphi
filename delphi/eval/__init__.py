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


@clock
@singledispatch
def eval_task(task_args: object, **kwargs) -> None:

    raise ValueError(
        f"task type {task_args.__class__.__name__} not supported. "
        f"please register a new task with the @eval_task.register decorator."
    )
