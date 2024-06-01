import torch
from dgl import DGLGraph
from torch import nn
import torch.nn.functional as F
import dgl.function as fn
from sklearn.metrics import roc_auc_score


class DotPredictor(nn.Module):
    def forward(self, g: DGLGraph, h):
        with g.local_scope():
            g.ndata["h"] = h
            g.apply_edges(fn.u_dot_v("h", "h", "score"))
            return g.edata["score"][:, 0]

    def compute_loss(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score])
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        )
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_loss_only_pos(self, pos_score):
        scores = torch.cat([pos_score])
        labels = torch.cat([torch.ones(pos_score.shape[0])])
        return F.binary_cross_entropy_with_logits(scores, labels)

    def compute_auc(self, pos_score, neg_score):
        scores = torch.cat([pos_score, neg_score]).numpy()
        labels = torch.cat(
            [torch.ones(pos_score.shape[0]), torch.zeros(neg_score.shape[0])]
        ).numpy()
        return roc_auc_score(labels, scores)