from functools import wraps
from inspect import signature
from pathlib import Path
from typing import Callable

from loguru import logger


def cache(out_path_name: str, load_func):

    def decorator(func: Callable):

        @wraps(func)
        def wrapped(*args, **kwargs):
            sign = signature(func)
            out_path_str = sign.bind(*args, **kwargs).arguments[out_path_name]
            out_path = Path(out_path_str)

            if out_path.exists():
                logger.info(f'{out_path} already exists, directly loading from it.')
                return load_func(out_path_str)

            return func(*args, **kwargs)

        return wrapped

    return decorator
