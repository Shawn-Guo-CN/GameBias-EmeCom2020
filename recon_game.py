import torch
import torch.nn as nn
import torch.nn.functional as F
import egg.core as core

from utils import View, kaiming_init, TopographicSimilarityLatents, ConsoleFileLogger
from modules import BaseGame, DspritesSenderCNN, SymbolicSenderMLP
from data_loader import get_dsprites_dataloader, get_symbolic_dataloader


class DSpritesReceiverCNN(nn.Module):
    """The specific CNN receiver for dSprite images."""
    
    def __init__(self, hidden_size=256) -> None:
        super().__init__()

        self.hidden_size = hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, 256),         # B, 256
            nn.ReLU(True),
            nn.Linear(256, 256),                 # B, 256
            nn.ReLU(True),
            nn.Linear(256, 32*4*4),              # B, 512
            nn.ReLU(True),
            View((-1, 32, 4, 4)),                # B,  32,  4,  4
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32,  8,  8
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 16, 16
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 32, 4, 2, 1), # B,  32, 32, 32
            nn.ReLU(True),
            nn.ConvTranspose2d(32, 1, 4, 2, 1), # B,  nc, 64, 64
            # nn.Sigmoid(),
            View((-1, 64*64))
        )

        self.weight_init()

    def forward(self, x, input=None):
        return torch.squeeze(self.decoder(x), 1)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

class DspritesReconGame(BaseGame):

    def __init__(self, data_path:str=None) -> None:
        super().__init__()

        self.train_loader, self.test_loader = \
            get_dsprites_dataloader(batch_size=self.batch_size, path_to_data=data_path)

        self.sender = core.RnnSenderGS(
            DspritesSenderCNN(self.hidden_size), 
            self.vocab_size,
            self.emb_size,
            self.hidden_size,
            max_len=self.max_len,
            cell="lstm", 
            temperature=1.0
        )

        self.receiver = core.RnnReceiverGS(
            DSpritesReceiverCNN(self.hidden_size),
            self.vocab_size,
            self.emb_size,
            self.hidden_size,
            cell="lstm"
        )

        self.game = core.SenderReceiverRnnGS(self.sender, self.receiver, self.loss)

        self.optimiser = core.build_optimizer(self.game.parameters())
        self.callbacks = []
        self.callbacks.append(core.ConsoleLogger(as_json=True,print_train_loss=True))
        self.callbacks.append(core.TemperatureUpdater(agent=self.sender, decay=0.9, minimum=0.1))
        self.callbacks.append(TopographicSimilarityLatents('euclidean', 'edit'))
        self.trainer = core.Trainer(
            game=self.game, optimizer=self.optimiser, train_data=self.train_loader, validation_data=self.test_loader,
            callbacks=self.callbacks
        )

    @staticmethod
    def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
        # loss = F.binary_cross_entropy(receiver_output, sender_input.view(-1, 4096), reduction='none').sum(dim=1)
        loss = F.mse_loss(receiver_output, sender_input.view(-1, 4096), reduction = 'none').sum(dim=1)
        return loss, {}


class SymbolicReceiverMLP(nn.Module):
    """The specific MLP receiver for symbolic dataset."""
    
    def __init__(self, output_dim=18, hidden_size=256) -> None:
        super().__init__()

        self.output_dim = output_dim
        self.hidden_size = hidden_size

        self.decoder = nn.Sequential(
            nn.Linear(hidden_size, hidden_size), # B, 256
            nn.ReLU(True),
            nn.Linear(hidden_size, hidden_size), # B, 256
            nn.ReLU(True),
            nn.Linear(hidden_size, output_dim),  # B, out_dim
            nn.Sigmoid()
        )

        self.weight_init()

    def forward(self, x, input=None):
        return torch.squeeze(self.decoder(x), 1)

    def weight_init(self):
        for block in self._modules:
            for m in self._modules[block]:
                kaiming_init(m)

class SymbolicReconGame(BaseGame):

    def __init__(self, training_log=None) -> None:
        super().__init__()

        self.training_log = training_log if training_log is not None else core.get_opts().training_log_path

        self.train_loader, self.test_loader = \
            get_symbolic_dataloader(
                n_attributes=self.n_attributes,
                n_values=self.n_values,
                batch_size=self.batch_size
            )

        self.sender = core.RnnSenderGS(
            SymbolicSenderMLP(input_dim=self.n_attributes*self.n_values, hidden_dim=self.hidden_size), 
            self.vocab_size,
            self.emb_size,
            self.hidden_size,
            max_len=self.max_len,
            cell="lstm", 
            temperature=1.0
        )

        self.receiver = core.RnnReceiverGS(
            SymbolicReceiverMLP(self.n_attributes * self.n_values, self.hidden_size),
            self.vocab_size,
            self.emb_size,
            self.hidden_size,
            cell="lstm"
        )

        self.game = core.SenderReceiverRnnGS(self.sender, self.receiver, self.loss)

        self.optimiser = core.build_optimizer(self.game.parameters())
        self.callbacks = []
        self.callbacks.append(ConsoleFileLogger(as_json=True,print_train_loss=True,logfile_path=self.training_log))
        self.callbacks.append(core.TemperatureUpdater(agent=self.sender, decay=0.9, minimum=0.1))
        self.callbacks.append(TopographicSimilarityLatents('hamming', 'edit', log_path=core.get_opts().topo_path))
        self.trainer = core.Trainer(
            game=self.game, optimizer=self.optimiser, train_data=self.train_loader, validation_data=self.test_loader,
            callbacks=self.callbacks
        )

    @staticmethod
    def loss(sender_input, _message, _receiver_input, receiver_output, _labels):
        # loss_f = nn.BCELoss()
        # loss = loss_f(receiver_output, sender_input)
        loss = F.mse_loss(receiver_output, sender_input)
        return loss, {}
