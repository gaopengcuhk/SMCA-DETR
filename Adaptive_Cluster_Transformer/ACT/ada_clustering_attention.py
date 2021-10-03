from .extensions import ada_cluster, broadcast, weighted_sum

import torch
from math import fabs
from torch.nn import Module, Dropout

class WeightedSoftMax(Module):
    def __init__(self):
        super(WeightedSoftMax, self).__init__()
    
    def forward(self, x, dim=None, weight=None):
        ret = torch.softmax(x, dim=dim)
        if weight is not None:
            ret = ret * weight.unsqueeze(1)
            ret = ret / ret.sum(dim=-1, keepdim=True)
        return ret


class CalcCenter(torch.autograd.Function):
    @staticmethod
    def forward(ctx, x, clusters, counts):
        weights = 1/counts.float()
        center = weighted_sum(x, clusters, weights)
        ctx.save_for_backward(clusters, weights)

        return center

    @staticmethod
    def backward(ctx, grad_center):
        clusters, weights = ctx.saved_tensors
        grad = broadcast(grad_center, clusters, weights)

        return grad, None, None


class Broadcast(torch.autograd.Function):
    @staticmethod
    def forward(ctx, center, clusters):
        B, C, D = center.shape
        weights = torch.ones((B, C), dtype=torch.float, device=center.device)
        x = broadcast(center, clusters, weights)
        ctx.save_for_backward(clusters)
        
        return x

    @staticmethod
    def backward(ctx, grad):
        B, N, D = grad.shape
        clusters = ctx.saved_tensors[0]
        C = clusters.max() + 1
        weights = torch.ones((B, C), dtype=torch.float, device=grad.device)
        grad_center = weighted_sum(grad, clusters, weights)

        return grad_center, None, None

class AdaClusteringAttention(Module):
    """Use E2LSH to adaptively cluster queries or keys

    Arguments
    ---------
        group_Q: If true, use E2LSH to adaptively cluster queries
        group_K: If true, use E2LSH to adaptively cluster keys
        q_hashes: If group_Q is true, q_hashes represents the rounds of 
                  E2LSH that appled to queries
        k_hashes: If group_K is true, k_hashes represents the rounds of 
                  E2LSH that appled to keys
        softmax_temp: The temperature to use for the softmax attention
        attention_dropout: The dropout rate to apply to the attention
    """
    def __init__(self, group_Q=False, group_K=False, q_hashes=32,
                 k_hashes=8, softmax_temp=1, attention_dropout=0.0):
        super(AdaClusteringAttention, self).__init__()
        self.group_Q = group_Q
        self.group_K = group_K
        self.q_hashes = q_hashes
        self.k_hashes = k_hashes
        self.softmax_temp = softmax_temp
        if attention_dropout > 0.0:
            self.dropout = Dropout(attention_dropout)
        else:
            self.dropout = None
        self.softmax = WeightedSoftMax()

    @torch.no_grad()
    def _compute_hashes(self, V, r=8, n_hashes=32):
        B, N, D = V.shape
        weights = torch.randn((1, D, n_hashes), dtype=torch.float, 
                                device=V.device).expand(B, -1, -1)
        vecs = torch.bmm(V, weights).permute(0, 2, 1)
        bias = torch.rand((1, n_hashes, 1), dtype=torch.float, 
                                device=V.device).expand(B, -1, -1)
        vecs = (vecs + bias)/r
        hashes = torch.floor(vecs).long()
        hashes = (hashes - hashes.min(dim=-1)[0].unsqueeze(-1)).long()
        offsets = torch.arange(n_hashes, device=V.device, dtype=torch.long)*2
        hashes = (hashes << offsets.view(1, -1, 1)).sum(dim=1)
        return hashes.contiguous()


    def _create_clusters(self, V, n_hashes=32):
        B, N, D = V.shape
        hashes = self._compute_hashes(V)
        groups, counts =  ada_cluster(hashes, n_hashes=n_hashes)
        return groups, counts

    def forward(self, queries, keys, values, key_padding_mask=None):
        queries = queries.contiguous()
        keys = keys.contiguous()
        values = values.contiguous()
        
        if self.group_Q:
            q_groups, q_counts = self._create_clusters(queries, self.q_hashes)
            Q_center = CalcCenter.apply(queries, q_groups, q_counts)
            self.Q_clusters = q_counts.size(-1) # number of clusters
        else:
            Q_center = queries

        if self.group_K:
            k_groups, k_counts = self._create_clusters(keys, self.k_hashes)
            K_center = CalcCenter.apply(keys, k_groups, k_counts)
            V_center = CalcCenter.apply(values, k_groups, k_counts)
            self.K_clusters = k_counts.size(-1) # number of clusters
        else:
            K_center = keys
            V_center = values

        QK = torch.bmm(Q_center, K_center.permute(0, 2, 1))
        if key_padding_mask is not None:
            assert self.group_K is not True
            QK = QK.view(key_padding_mask.size(0), -1, Q_center.size(1), keys.size(1))
            QK = QK.masked_fill(key_padding_mask.unsqueeze(1).unsqueeze(2), float('-inf'))
            QK = QK.view(-1, Q_center.size(1), keys.size(1))

        softmax_weight = k_counts if self.group_K else None
        A = self.softmax(self.softmax_temp * QK, dim=-1, weight=softmax_weight)
        if self.dropout:
            A = self.dropout(A)
        
        V = torch.bmm(A, V_center)
        if self.group_Q:
            V = Broadcast.apply(V, q_groups)

        return V.contiguous()
