import torch
import torch.utils.data
from torch import nn
from torch.nn import functional as F
import torch.nn.init as init



class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, tensor):
        return tensor.view(self.size)

def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def normal_init(m, mean, std):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        m.weight.data.normal_(mean, std)
        if m.bias.data is not None:
            m.bias.data.zero_()
    elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
        m.weight.data.fill_(1)
        if m.bias.data is not None:
            m.bias.data.zero_()

# MLP sender
class Sender(nn.Module):
    def __init__(self, message_dim, image_size, hidden_size=400):
        super(Sender, self).__init__()
        self.fc1 = nn.Linear(image_size, hidden_size)
        self.fc21 = nn.Linear(hidden_size, message_dim)
        self.fc22 = nn.Linear(hidden_size, message_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        mu, logvar = self.fc21(x), self.fc22(x)

        return mu, logvar

# MLP receiver
class Receiver(nn.Module):
    def __init__(self, message_dim, image_size, hidden_size=400):
        super(Receiver, self).__init__()
        self.fc3 = nn.Linear(message_dim, hidden_size)
        self.fc4 = nn.Linear(hidden_size, image_size)

    def forward(self, x):
        x = F.relu(self.fc3(x))
        return torch.sigmoid(self.fc4(x))

"""
Beta-VAE 
"""


# CNN sender
class VisualSender(nn.Module):
    def __init__(self, channels=1, z_dim=10):
        super(VisualSender, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(channels, 32, 4, 2, 1),  # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),  # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),  # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),  # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),  # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256 * 1 * 1)),  # B, 256
            nn.Linear(256, z_dim * 2),  # B, z_dim*2
        )

        self.weight_init()

    def forward(self, x):
        return self.encoder(x)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

# CNN receiver
class VisualReceiver(nn.Module):
    def __init__(self, z_dim=10, channels=1):
        super(VisualReceiver, self).__init__()

        self.decoder = nn.Sequential(
            nn.Linear(z_dim, 256),               # B, 256
            View((-1, 256, 1, 1)),               # B, 256,  1,  1
            nn.ReLU(True),
            nn.ConvTranspose2d(256, 64, 4),      # B,  64,  4,  4
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 64, 4, 2, 1), # B,  64,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, channels, 4, 2, 1),  # B, nc, 64, 64
        )
        self.weight_init()

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


    def forward(self, x):
        return self.decoder(x)




# TODO: Model from the understanding beta-VAE paper
