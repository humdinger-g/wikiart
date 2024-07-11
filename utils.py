import numpy as np
from torch import nn

from typing import List, Union



def gini_index(arr: Union[List[int], np.ndarray]) -> float:
    '''
    Measure of the class disbalance, takes values in [0,1].
    Higher value indicates higher disbalance.

    '''
    arr = np.cumsum(sorted(arr))
    X = np.linspace(0, 1, len(arr) + 1)
    y = np.hstack([[0], np.array(arr) / arr[-1]])
    delta = 1 / (len(X) + 1)
    return 1 - 2 * sum(y * delta)


def num_params(model: nn.Module) -> int:
    return sum([p.numel() for p in model.parameters()])