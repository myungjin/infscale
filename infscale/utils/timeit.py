"""Decorator to measure function execution time."""
import time
from functools import wraps

from infscale import get_logger

logger = get_logger()


def timeit(func):
    """Measure function execution time as decorator.

    Usage example:
    @timeit
    def do_something(num):
        total = sum((x for x in range(0, num)))
        return total

    To access the execution time outside the function is achieved as follows.
    e.g.: print(do_simething.elapsed)
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.perf_counter()
        result = func(*args, **kwargs)
        end_time = time.perf_counter()

        wrapper.elapsed = end_time - start_time

        logger.debug(f"elapsed time = {wrapper.elapsed}")

        return result

    wrapper.elaspsed = 0

    return wrapper
