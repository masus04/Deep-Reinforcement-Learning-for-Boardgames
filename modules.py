from torch.nn import Module


def swish(x):
    return x * x.sigmoid()
