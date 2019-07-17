import torch.nn as nn
import torch.nn.functional as F

## network
class BOWMLP(nn.Module):
    def __init__(self, in_size, out_size, hidden_size):
        super(BOWMLP, self).__init__()
        self.in_size = in_size
        self.hidden_size = hidden_size
        #
        self.aff1 = nn.Linear(in_size, hidden_size)
        self.aff2 = nn.Linear(hidden_size, out_size)

    def forward(self, x):
        x = x.view(-1, self.in_size)
        x = F.relu(self.aff1(x))
        x = F.dropout(x, p=0.5, training=self.training)
        # x = F.relu(self.aff2(x))
        x = self.aff2(x) # no activation pre-softmax
        # x = F.dropout(x, p=0.5, training=self.training)
        # x = F.softmax(x, dim=1)
        return x

    def name(self):
        return "BOW-MLP"
