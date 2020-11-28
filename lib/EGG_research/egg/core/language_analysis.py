# Copyright (c) Facebook, Inc. and its affiliates.

# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

from typing import Union, Callable
import os
from collections import defaultdict

import editdistance
from scipy.spatial import distance
from scipy.stats import spearmanr

import numpy as np
import torch
from .callbacks import Callback
from .interaction import Interaction
import json


def mutual_info(xs, ys):
    """
    I[x, y] = E[x] + E[y] - E[x,y]
    """
    e_x = calc_entropy(xs)
    e_y = calc_entropy(ys)

    xys = []

    for x, y in zip(xs, ys):
        xy = (x[0].item(), y[0].item())
        xys.append(xy)

    #xys = torch.from_numpy(np.array(xys))
    #print(xys)
    e_xy = calc_entropy_tuples(xys)

    return e_x + e_y - e_xy


def entropy_dict(freq_table):
    """
    >>> d = {'a': 1, 'b': 1}
    >>> np.allclose(entropy_dict(d), 1.0)
    True
    >>> d = {'a': 1, 'b': 0}
    >>> np.allclose(entropy_dict(d), 0.0, rtol=1e-5, atol=1e-5)
    True
    """
    t = torch.tensor([v for v in freq_table.values()]).float()
    return torch.distributions.Categorical(probs=t).entropy().item() / np.log(2)


def calc_entropy(messages):
    """
    >>> messages = torch.tensor([[1, 2], [3, 4]])
    >>> np.allclose(calc_entropy(messages), 1.0)
    True
    """
    freq_table = defaultdict(float)

    for m in messages:
        m = _hashable_tensor(m)
        freq_table[m] += 1.0

    return entropy_dict(freq_table)

def calc_entropy_tuples(messages):
    """
    >>> messages = torch.tensor([[1, 2], [3, 4]])
    >>> np.allclose(calc_entropy(messages), 1.0)
    True
    """
    freq_table = defaultdict(float)

    for m in messages:
        freq_table[m] += 1.0

    return entropy_dict(freq_table)


def _hashable_tensor(t):
    t = tuple(t.tolist())
    return t


class MessageEntropy(Callback):
    def __init__(self, print_train: bool = True, is_gumbel: bool = False):
        super().__init__()
        self.print_train = print_train
        self.is_gumbel = is_gumbel

    def print_message_entropy(self, logs: Interaction, tag: str, epoch: int):
        message = logs.message.argmax(
            dim=-1) if self.is_gumbel else logs.message
        entropy = calc_entropy(message)

        output = json.dumps(dict(entropy=entropy, mode=tag, epoch=epoch))
        print(output, flush=True)

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if self.print_train:
            self.print_message_entropy(logs, 'train', epoch)

    def on_test_end(self, loss, logs, epoch):
        self.print_message_entropy(logs, 'test', epoch)


def histogram(strings, vocab_size):
    batch_size = strings.size(0)

    histogram = torch.zeros(batch_size, vocab_size, device=strings.device)

    for v in range(vocab_size):
        histogram[:, v] = strings.eq(v).sum(dim=-1)

    return histogram


class PosDisent(Callback):
    """
    Positional disentanglement metric, introduced in "Compositionality and Generalization in Emergent Languages",
    Chaabouni et al., ACL 2020.
    """

    def __init__(self, print_train: bool = False, print_test: bool = True, is_gumbel: bool = False):
        super().__init__()
        assert print_train or print_test, 'At least on of "print_train" and "print_train" must be enabled'

        self.print_train = print_train
        self.print_test = print_test
        self.is_gumbel = is_gumbel

    @staticmethod
    def posdis(attributes, messages):
        """
        Two-symbol messages representing two-attribute world. One symbol encodes on attribute:
        in this case, the metric should be maximized:
        >>> samples = 1_000
        >>> _ = torch.manual_seed(0)
        >>> attribute1 = torch.randint(low=0, high=10, size=(samples, 1))
        >>> attribute2 = torch.randint(low=0, high=10, size=(samples, 1))
        >>> attributes = torch.cat([attribute1, attribute2], dim=1)
        >>> messages = attributes
        >>> PosDisent.posdis(attributes, messages)
        0.9786556959152222
        >>> messages = torch.cat([messages, torch.zeros_like(messages)], dim=1)
        >>> PosDisent.posdis(attributes, messages)
        0.9786556959152222
        """
        gaps = torch.zeros(messages.size(1))
        non_constant_positions = 0.0

        for j in range(messages.size(1)):
            symbol_mi = []
            h_j = None
            for i in range(attributes.size(1)):
                xs, ys = attributes[:, i], messages[:, j]
                # discretize y based on its histogram
                ys_sorted = sorted(ys)
                bins_count = 20 # TODO: make this settable somehow
                histogram = np.histogram(ys, bins=bins_count)
                left = histogram[0][0]
                width = histogram[1][1] - histogram[1][0]
                ys_disc = []
                for y in ys:
                    assigned_symbol = np.trunc((y - left) / width)
                    # max value needs special treatment
                    if assigned_symbol >= bins_count:
                        assigned_symbol = bins_count - 1
                    ys_disc.append([assigned_symbol])

                # compute mutual information I(X;Y) and entropy H(Y)

                ys_disc = torch.from_numpy(np.array(ys_disc))

                info = mutual_info(xs, ys_disc)
                symbol_mi.append(info)

                if h_j is None:
                    h_j = calc_entropy(ys_disc)

            symbol_mi.sort(reverse=True)

            if h_j > 0.0:
                gaps[j] = (symbol_mi[0] - symbol_mi[1]) / h_j
                non_constant_positions += 1

        score = gaps.sum() / non_constant_positions
        return score.item()

    def print_message(self, logs: Interaction, tag: str, epoch: int):
        message = logs.message.argmax(
            dim=-1) if self.is_gumbel else logs.message

        posdis = self.posdis(logs.sender_input, message)

        output = json.dumps(dict(posdis=posdis, mode=tag, epoch=epoch))
        print(output, flush=True)

    def on_epoch_end(self, _loss, logs: Interaction, epoch: int):
        if self.print_train:
            self.print_message(logs, 'train', epoch)

    def on_test_end(self, loss, logs, epoch):
        if self.print_test:
            self.print_message(logs, 'test', epoch)


class TopographicSimilarity(Callback):
    distances = {'edit': lambda x, y: editdistance.eval(x, y) / ((len(x) + len(y)) / 2),
                 'cosine': distance.cosine,
                 'hamming': distance.hamming,
                 'jaccard': distance.jaccard,
                 'euclidean': distance.euclidean,
                 }

    def __init__(self,
                 sender_input_distance_fn: Union[str, Callable] = 'cosine',
                 message_distance_fn: Union[str, Callable] = 'edit',
                 compute_topsim_train_set: bool = False,
                 compute_topsim_test_set: bool = True):

        self.sender_input_distance_fn = self.distances.get(sender_input_distance_fn, None) \
            if isinstance(sender_input_distance_fn, str) else sender_input_distance_fn
        self.message_distance_fn = self.distances.get(message_distance_fn, None) \
            if isinstance(message_distance_fn, str) else message_distance_fn
        self.compute_topsim_train_set = compute_topsim_train_set
        self.compute_topsim_test_set = compute_topsim_test_set

        assert self.sender_input_distance_fn and self.message_distance_fn, f"Cannot recognize {sender_input_distance_fn} or {message_distance_fn} distances"
        assert compute_topsim_train_set or compute_topsim_test_set

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_test_set:
            self.compute_similarity(
                sender_input=logs.sender_input, messages=logs.message, epoch=epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.compute_topsim_train_set:
            self.compute_similarity(
                sender_input=logs.sender_input, messages=logs.message, epoch=epoch)

    def compute_similarity(self, sender_input: torch.Tensor, messages: torch.Tensor, epoch: int):
        def compute_distance(_list, distance):
            return [distance(el1, el2)
                    for i, el1 in enumerate(_list[:-1])
                    for j, el2 in enumerate(_list[i+1:])
                    ]

        messages = [msg.tolist() for msg in messages]

        input_dist = compute_distance(
            sender_input.numpy(), self.sender_input_distance_fn)
        message_dist = compute_distance(messages, self.message_distance_fn)
        topsim = spearmanr(input_dist, message_dist, nan_policy='raise').correlation

        output_message = json.dumps(dict(topsim=topsim, epoch=epoch))


