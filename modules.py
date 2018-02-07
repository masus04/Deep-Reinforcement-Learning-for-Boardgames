from torch.nn import Module


class LegalSoftMax(Module):
    
    def __init__(self):
        super(LegalSoftMax, self).__init__()

    def forward(self, input, legal_map):

        x = input.exp()
        x = x * legal_map
        x = x / x.sum()

        return x
