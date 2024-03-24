import torch
import torch.nn as nn

x = 
y = 

x = torch.tensor(x).float
y = 

device = 'cuda' if torch.cuda_is_available() else 'cpu'
x.to(device)
y.to(device)

class MyNeuraNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.input_to_hidden_layer = nn.Linear(2, 8)
        self