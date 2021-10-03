import torch

from .ada_cluster import _ada_cluster
from .broadcast import _broadcast
from .weighted_sum import _weighted_sum

def ada_cluster(hashes):
    pos = hashes.argsort(dim=-1)
    B, N = hashes.shape
    groups = torch.zeros((B, N), dtype=torch.int, device=hashes.device)
    counts = torch.zeros((B, N), dtype=torch.int, device=hashes.device)
    _ada_cluster(hashes, pos, groups, counts)
    clusters = groups.max()
    return groups, counts[:, :clusters+1].contiguous()


def weighted_sum(x, groups, weights):
    assert x.size(1) == groups.size(1)
    B, N, D = x.shape
    C = weights.size(1)
    y = torch.zeros((B, C, D), dtype=torch.float, device=x.device)
    _weighted_sum(x, groups, weights, y)
    return y


def broadcast(y, groups, weights):
    assert y.size(1) == weights.size(1)
    B, C, D = y.shape
    N = groups.size(1)
    x = torch.zeros((B, N, D), dtype=torch.float, device=y.device)
    _broadcast(y, groups, weights, x)
    return x
