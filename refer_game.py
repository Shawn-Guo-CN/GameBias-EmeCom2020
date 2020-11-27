from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F
import egg.core as core

from utils import View, kaiming_init, TopographicSimilarityLatents, ConsoleFileLogger
from modules import BaseGame, DspritesSenderCNN, SymbolicSenderMLP
from data_loader import get_dsprites_dataloader, get_symbolic_dataloader


class DSpritesReceiverCNN(nn.Module):
    def __init__(self, game_size, embedding_size, hidden_size, reinforce=False):
        super().__init__()
        self.game_size = game_size # number of candidates
        self.embedding_size = embedding_size # size of messages embeddings
        self.hidden_size = hidden_size # size of hidden representation

        self.encoder = DspritesSenderCNN(hidden_size)
        if reinforce:
            self.lin2 = nn.Embedding(embedding_size, hidden_size)
        else:
            self.lin2 = nn.Linear(embedding_size, hidden_size, bias=False)

    def forward(self, signal, candidates):
        """
        Parameters
        ----------
        signal : torch.tensor
            Tensor for the embedding of received messages whose shape is $Batch_size * Hidden_size$
        candidates : list
            A list containing multiple torch.tensor, every tensor is a candidate image.
        """

        # embed each image (left or right)
        embs = self.return_embeddings(candidates)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.squeeze(dim=-1)
        h_s = F.relu(self.lin2(signal))
        # h_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(embs, h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size
        #log_probs = F.log_softmax(out, dim=1)
        log_probs = out
        return log_probs

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            h_i = self.encoder(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h


@dataclass
class DspritesReferGame(BaseGame):

    def __init__(self, data_path:str) -> None:
        super().__init__()

        self.train_loader, self.test_loader = \
            get_dsprites_dataloader(
                batch_size=self.batch_size,
                path_to_data=data_path,
                game_size=self.game_size,
                referential=True
            )

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
            DSpritesReceiverCNN(self.game_size, self.emb_size, self.hidden_size, reinforce=False),
            self.vocab_size,
            self.emb_size,
            self.hidden_size,
            cell="lstm"
        )

        self.game = core.SenderReceiverRnnGS(self.sender, self.receiver, self.loss)

        self.optimiser = core.build_optimizer(self.game.parameters())
        self.callbacks = []
        self.callbacks.append(core.ConsoleLogger(as_json=True,print_train_loss=True))
        self.callbacks.append(TopographicSimilarityLatents('euclidean', 'edit'))
        #self.callbacks.append(core.TemperatureUpdater(agent=self.sender, decay=0.9, minimum=0.1))
        self.trainer = core.Trainer(
            game=self.game, optimizer=self.optimiser, train_data=self.train_loader, validation_data=self.test_loader,
            callbacks=self.callbacks
        )
    
    @staticmethod
    def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(receiver_output, labels.squeeze(dim=1))
        acc = (labels.squeeze(dim=1) == receiver_output.argmax(dim=1)).float().mean().unsqueeze(dim=-1)
        return loss, {'acc': acc}


class SymbolicReceiverMLP(nn.Module):
    def __init__(self, game_size, embedding_size, hidden_size, input_dim=18):
        super().__init__()
        self.game_size = game_size # number of candidates
        self.embedding_size = embedding_size # size of messages embeddings
        self.hidden_size = hidden_size # size of hidden representation
        self.input_dim = input_dim

        self.encoder = SymbolicSenderMLP(input_dim, hidden_size)
        self.lin2 = nn.Linear(embedding_size, hidden_size, bias=False)

    def forward(self, signal, candidates):
        """
        Parameters
        ----------
        signal : torch.tensor
            Tensor for the embedding of received messages whose shape is $Batch_size * Hidden_size$
        candidates : list
            A list containing multiple torch.tensor, every tensor is a candidate image.
        """

        # embed each image (left or right)
        embs = self.return_embeddings(candidates)
        # embed the signal
        if len(signal.size()) == 3:
            signal = signal.squeeze(dim=-1)
        h_s = self.lin2(signal)
        # h_s is of size batch_size x embedding_size
        h_s = h_s.unsqueeze(dim=1)
        # h_s is of size batch_size x 1 x embedding_size
        h_s = h_s.transpose(1, 2)
        # h_s is of size batch_size x embedding_size x 1
        out = torch.bmm(embs, h_s)
        # out is of size batch_size x game_size x 1
        out = out.squeeze(dim=-1)
        # out is of size batch_size x game_size
        #log_probs = F.log_softmax(out, dim=1)
        log_probs = out
        return log_probs

    def return_embeddings(self, x):
        # embed each image (left or right)
        embs = []
        for i in range(self.game_size):
            h = x[i]
            h_i = self.encoder(h)
            # h_i are batch_size x embedding_size
            h_i = h_i.unsqueeze(dim=1)
            # h_i are now batch_size x 1 x embedding_size
            embs.append(h_i)
        h = torch.cat(embs, dim=1)
        return h


class SymbolicReferGame(BaseGame):

    def __init__(self, training_log=None) -> None:
        super().__init__()

        self.training_log = training_log if training_log is not None else core.get_opts().training_log_path

        self.train_loader, self.test_loader = \
            get_symbolic_dataloader(
                n_attributes=self.n_attributes,
                n_values=self.n_values,
                batch_size=self.batch_size,
                game_size=self.game_size,
                referential=True
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
            SymbolicReceiverMLP(self.game_size, self.emb_size, self.hidden_size, 
                input_dim=self.n_attributes*self.n_values
            ),
            self.vocab_size,
            self.emb_size,
            self.hidden_size,
            cell="lstm"
        )

        self.game = core.SenderReceiverRnnGS(self.sender, self.receiver, self.loss)

        self.optimiser = core.build_optimizer(self.game.parameters())
        self.callbacks = []
        self.callbacks.append(ConsoleFileLogger(as_json=True,print_train_loss=True,logfile_path=self.training_log))
        self.callbacks.append(TopographicSimilarityLatents(
            'hamming', 'edit', referential=True, log_path=core.get_opts().topo_path))
        #self.callbacks.append(core.TemperatureUpdater(agent=self.sender, decay=0.9, minimum=0.1))
        self.trainer = core.Trainer(
            game=self.game, optimizer=self.optimiser, train_data=self.train_loader, validation_data=self.test_loader,
            callbacks=self.callbacks
        )
    
    @staticmethod
    def loss(_sender_input, _message, _receiver_input, receiver_output, labels):
        loss_fun = nn.CrossEntropyLoss()
        loss = loss_fun(receiver_output, labels[1].squeeze(dim=1))
        acc = (labels[1].squeeze(dim=1) == receiver_output.argmax(dim=1)).float().mean().unsqueeze(dim=-1)
        return loss, {'acc': acc}
