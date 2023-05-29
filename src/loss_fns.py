"""
This script records all attacking losses that we use
in attackers and Upsilon evaluations.
"""
import torch

EPS = 1e-12


class CE(torch.nn.Module):
    loss_fn = torch.nn.CrossEntropyLoss(reduction="none")
    loss_name = "CE"

    def __init__(self):
        super(CE, self).__init__()

    def forward(self, *args):
        return self.loss_fn(*args)

    def __str__(self) -> str:
        return self.loss_name


class DLR(torch.nn.Module):
    loss_name = "DLR"

    def __init__(self):
        super(DLR, self).__init__()

    def forward(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return -(x[u, y] - x_sorted[:, -2] * ind - x_sorted[:, -1] * (1.0 - ind)) / (
            x_sorted[:, -1] - x_sorted[:, -3] + EPS
        )

    def __str__(self) -> str:
        return self.loss_name


class ReDLR(torch.nn.Module):
    loss_name = "ReDLR"

    def __init__(self):
        super(ReDLR, self).__init__()

    def forward(self, x, y):
        x_sorted, ind_sorted = x.sort(dim=1)
        ind = (ind_sorted[:, -1] == y).float()
        u = torch.arange(x.shape[0])

        return (
            -(x[u, y] - x_sorted[:, -2])
            / (x_sorted[:, -1] - x_sorted[:, -3] + EPS)
            * ind
        )

    def __str__(self) -> str:
        return self.loss_name
