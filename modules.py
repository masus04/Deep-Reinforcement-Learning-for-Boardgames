from torch.nn import Module


class LegalSoftMax(Module):
    
    def __init__(self):
        super(LegalSoftMax, self).__init__()
        self.HARD_LIMIT = 150

    def forward(self, input, legal_map):

        x = input.tanh()
        x = x.exp()
        x = x * legal_map
        x = x / x.sum()

        return x
