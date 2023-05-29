import torch
from torch.utils.data import TensorDataset, DataLoader
from data import get_cifar10_split, get_cifar10_loader, BATCH_SIZE


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loss_grad(network, X=None, y=None, loss_fn=None):
    """
    Calculate the gradient of loss with respect to the input.

    return:
    grads: a tensor of gradient of each data point, shape: dataset_size * 3 * 32 * 32

    """
    if X == None:
        loader = get_cifar10_loader()
    else:
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )
    network.to(device).eval()
    grad = torch.Tensor([]).to("cpu")
    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        X_b.requires_grad = True
        loss_b = loss_fn(network(X_b), y_b)  # shape: batch_size
        # take sum of loss to get a scalar.
        # we want to obtain the "unscaled" gradient for each image.
        grad_b = torch.autograd.grad(outputs=loss_b.sum(), inputs=X_b)[0]
        # shape: batch_size * 3 * 32 * 32
        X_b.requires_grad = False
        grad = torch.cat((grad, grad_b.detach().cpu()))
    return grad.to("cpu")  # shape: data_size * 3 * 32 * 32


def comp_upsilon(network, X=None, y=None, loss_fn=None, q=1, s=1):
    """
    Compute Upsilon for a given dataset.
    """

    grad = loss_grad(network, X, y, loss_fn)  # shape: data_size * 3 * 32 * 32
    grad_norm = grad.norm(p=s, dim=(1, 2, 3))  # shape: data_size
    return grad_norm.pow(q).mean().pow(1 / q).item()


def eval(network, X=None, y=None, loss_fn=None, cond=False):
    """
    Calculate the loss and accuracy on a given dataset.

    return: a dictionary with keys:
    loss: loss of network on (X,y) with loss_fn
    acc: accuracy of network on (X,y)
    loss_cond: conditional loss of network on misclassifed images in (X,y) if cond=True
    """
    if X == None:
        loader = get_cifar10_loader()
        X, y = get_cifar10_split()
    else:
        loader = DataLoader(
            TensorDataset(X, y),
            batch_size=BATCH_SIZE,
            shuffle=False,
        )

    network.to(device).eval()
    l = torch.Tensor([]).to("cpu")
    pred = torch.Tensor([]).to("cpu")

    for X_b, y_b in loader:
        X_b, y_b = X_b.to(device), y_b.to(device)
        logit = network(X_b)
        l_b = loss_fn(logit, y_b).detach().cpu()
        _, pred_b = torch.max(logit, dim=1)
        pred_b = pred_b.detach().cpu()
        l = torch.cat((l, l_b))
        pred = torch.cat((pred, pred_b))

    loss = l.mean()
    acc = (pred == y.long()).float().mean()
    res = {
        f"loss": loss.item(),
        "acc": acc.item(),
    }
    if cond:
        loss_cond = (l * (pred != y.long()).detach()).sum() / (pred != y.long()).sum()
        res[f"loss_cond"] = loss_cond.item()
    return res
