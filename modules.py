from abc import ABCMeta, abstractmethod, abstractstaticmethod

import torch
import torch.nn as nn
import egg.core as core

from utils import View, kaiming_init


class BaseGame(metaclass=ABCMeta):
    hidden_size = 256
    emb_size = 256
    vocab_size = 7
    max_len = 3
    game_size = 2
    batch_size = 64
    
    # parameters for symbolic game
    n_attributes = 3
    n_values = 6

    def __init__(self):
        self.opts = core.init()

    def train(self, num_epochs:int):
        self.trainer.train(num_epochs)


class BaseSender(nn.Module):
    def __init__(self, hidden_dim=256) -> None:
        super().__init__()
        self.encoder = None
        self.hidden_dim = hidden_dim

    def forward(self, x):
        return self.encoder(x)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)


class DspritesSenderCNN(BaseSender):
    """The specific CNN sender for dSprite images."""
    
    def __init__(self, hidden_dim=256) -> None:
        super().__init__(hidden_dim)

        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 4, 2, 1),          # B,  32, 32, 32
            nn.ReLU(True),
            nn.Conv2d(32, 32, 4, 2, 1),          # B,  32, 16, 16
            nn.ReLU(True),
            nn.Conv2d(32, 64, 4, 2, 1),          # B,  64,  8,  8
            nn.ReLU(True),
            nn.Conv2d(64, 64, 4, 2, 1),          # B,  64,  4,  4
            nn.ReLU(True),
            nn.Conv2d(64, 256, 4, 1),            # B, 256,  1,  1
            nn.ReLU(True),
            View((-1, 256*1*1)),                 # B, 256
            nn.Linear(256, hidden_dim),          # B, hidden_dim
        )

        self.weight_init()


class SymbolicSenderMLP(BaseSender):
    """"""

    def __init__(self, input_dim=18, hidden_dim=256) -> None:
        super().__init__(hidden_dim)

        self.input_dim = input_dim

        self.encoder = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim),
            nn.ReLU(True),
            nn.Linear(self.hidden_dim, self.hidden_dim)
        )

        self.weight_init()
