"""Auxiliary losses vendored verbatim from xxlya/BrainGNN_Pytorch 03-main.py
(commit 1e337e7).

Only change vs upstream: the consistency-loss Laplacian is placed on the
*input tensor's* device instead of a module-global ``device``. The math is
identical. See PROVENANCE.md.
"""
import torch

EPS = 1e-10


def topk_loss(s, ratio):
    """Per-graph pooling-score regularizer (upstream `topk_loss`).

    ``s`` has shape [B, n_kept]; sort within each graph (dim=1) and push the
    top fraction toward 1 and the bottom fraction toward 0.
    """
    if ratio > 0.5:
        ratio = 1 - ratio
    s = s.sort(dim=1).values
    res = -torch.log(s[:, -int(s.size(1) * ratio):] + EPS).mean() \
          - torch.log(1 - s[:, :int(s.size(1) * ratio)] + EPS).mean()
    return res


def consist_loss(s):
    """Group-consistency Laplacian regularizer (upstream `consist_loss`).

    ``s`` has shape [n_subjects_in_group, n_kept]. NOTE: applies sigmoid to
    ``s`` (this is the third sigmoid in the upstream score path — faithful,
    intentional).
    """
    if len(s) == 0:
        return 0
    s = torch.sigmoid(s)
    W = torch.ones(s.shape[0], s.shape[0], device=s.device)
    D = torch.eye(s.shape[0], device=s.device) * torch.sum(W, dim=1)
    L = D - W
    res = torch.trace(torch.transpose(s, 0, 1) @ L @ s) / (s.shape[0] * s.shape[0])
    return res
