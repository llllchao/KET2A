import torch
import torch.nn as nn

"""
multiple task loss
"""


class MTLoss(nn.Module):
    def __init__(self):
        super(MTLoss, self).__init__()
        self.w = nn.Parameter(torch.tensor([1, 1], dtype=torch.float32), requires_grad=True)

    def forward(self, l1, l2):
        w = torch.softmax(self.w, dim=0)
        l = torch.stack([l1, l2])
        return torch.sum(w * l) + 0.001 / torch.prod(w)
