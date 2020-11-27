import os
from os import truncate
from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
from egg.core.language_analysis import TopographicSimilarity
from egg.core import Callback
from egg.core.interaction import Interaction
import json
from typing import Union, Callable
from scipy.stats import spearmanr
import numpy


class TopographicSimilarityLatents(TopographicSimilarity):
    def __init__(self,
                 sender_input_distance_fn: Union[str, Callable] = 'cosine',
                 message_distance_fn: Union[str, Callable] = 'edit',
                 compute_topsim_train_set: bool = True,
                 compute_topsim_test_set: bool = False,
                 referential=False,
                 log_path='log/top_sim.txt',
                ):

        super().__init__(sender_input_distance_fn, message_distance_fn, compute_topsim_train_set, compute_topsim_test_set)
        self.referential = referential
        self.log_path = log_path

        create_dir_for_file(self.log_path)

    @staticmethod
    def compute_distance(_list, distance):
        return [distance(el1, el2)
                for i, el1 in enumerate(_list[:-1])
                for j, el2 in enumerate(_list[i+1:])
               ]

    def compute_similarity(self, sender_input: torch.Tensor, messages: torch.Tensor, epoch: int):
        message_argmax = messages.argmax(-1)
        messages_argmax = [msg.tolist() for msg in message_argmax]
        if self.referential:
            sender_input = sender_input[:, :3] # To tackle the labels in referential game.
        input_dist = self.compute_distance(
            sender_input.detach().cpu().numpy(), self.sender_input_distance_fn)
        message_dist = self.compute_distance(messages_argmax, self.message_distance_fn)
        topsim = spearmanr(input_dist, message_dist, nan_policy='raise').correlation

        output_message = json.dumps(dict(topsim=topsim, epoch=epoch))
        print(output_message, flush=True)
        with open(self.log_path, 'a') as f:
            print(output_message, file=f)


class ConsoleFileLogger(Callback):
    def __init__(self, print_train_loss=True, as_json=True, logfile_path=None):
        self.print_train_loss = print_train_loss
        self.as_json = as_json
        self.logfile_path = logfile_path

    def aggregate_print(self, loss: float, logs: Interaction, mode: str, epoch: int):
        dump = dict(loss=loss) 
        aggregated_metrics = dict((k, v.mean().item()) for k, v in logs.aux.items())
        dump.update(aggregated_metrics)

        if self.as_json:
            dump.update(dict(mode=mode, epoch=epoch))
            output_message = json.dumps(dump)
        else:
            output_message = ', '.join(sorted([f'{k}={v}' for k, v in dump.items()]))
            output_message = f'{mode}: epoch {epoch}, loss {loss}, ' + output_message
        print(output_message, flush=True)
        if self.logfile_path is not None:
            create_dir_for_file(self.logfile_path)
            with open(self.logfile_path,'a') as f:
                print(output_message, file=f)

    def on_test_end(self, loss: float, logs: Interaction, epoch: int):
        self.aggregate_print(loss, logs, 'test', epoch)

    def on_epoch_end(self, loss: float, logs: Interaction, epoch: int):
        if self.print_train_loss:
            self.aggregate_print(loss, logs, 'train', epoch)


class View(nn.Module):
    def __init__(self, size):
        super(View, self).__init__()
        self.size = size

    def forward(self, x):
        return x.view(self.size)


def kaiming_init(m):
    if isinstance(m, (nn.Linear, nn.Conv2d)):
        init.kaiming_normal(m.weight)
        if m.bias is not None:
            m.bias.data.fill_(0)
    elif isinstance(m, (nn.BatchNorm1d, nn.BatchNorm2d)):
        m.weight.data.fill_(1)
        if m.bias is not None:
            m.bias.data.fill_(0)


def smooth(x,window_len=20,window='hanning'):
    """smooth the data using a window with requested size.
    
    This method is based on the convolution of a scaled window with the signal.
    The signal is prepared by introducing reflected copies of the signal 
    (with the window size) in both ends so that transient parts are minimized
    in the begining and end part of the output signal.
    
    input:
        x: the input signal 
        window_len: the dimension of the smoothing window; should be an odd integer
        window: the type of window from 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'
            flat window will produce a moving average smoothing.

    output:
        the smoothed signal
        
    example:

    t=linspace(-2,2,0.1)
    x=sin(t)+randn(len(t))*0.1
    y=smooth(x)
    
    see also: 
    
    numpy.hanning, numpy.hamming, numpy.bartlett, numpy.blackman, numpy.convolve
    scipy.signal.lfilter
 
    TODO: the window parameter could be the window itself if an array instead of a string
    NOTE: length(output) != length(input), to correct this: return y[(window_len/2-1):-(window_len/2)] instead of just y.
    """

    if x.ndim != 1:
        raise ValueError("smooth only accepts 1 dimension arrays.")

    if x.size < window_len:
        raise ValueError("Input vector needs to be bigger than window size.")


    if window_len<3:
        return x


    if not window in ['flat', 'hanning', 'hamming', 'bartlett', 'blackman']:
        raise ValueError("Window is on of 'flat', 'hanning', 'hamming', 'bartlett', 'blackman'")


    s=numpy.r_[x[window_len-1:0:-1],x,x[-2:-window_len-1:-1]]
    #print(len(s))
    if window == 'flat': #moving average
        w=numpy.ones(window_len,'d')
    else:
        w=eval('numpy.'+window+'(window_len)')

    y=numpy.convolve(w/w.sum(),s,mode='valid')
    return y

def create_dir_for_file(file_path:str):
    dir_path = os.path.dirname(file_path)
    if not os.path.exists(dir_path):
        Path(dir_path).mkdir(parents=True, exist_ok=True)
