import sys
from tqdm import tqdm as _tqdm


def tqdm(*args, **kwargs):
    kwargs.setdefault("disable", sys.stderr is None)
    return _tqdm(*args, **kwargs)
