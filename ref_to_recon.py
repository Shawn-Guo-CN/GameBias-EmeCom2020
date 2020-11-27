#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 28 11:49:27 2020

@author: joshua, Shawn
"""


# -*- coding: utf-8 -*-
"""
First play ref game, get messages, then use the messages to play reconstruction
game. Record the loss.
"""

from recon_game import SymbolicReconGame
from refer_game import SymbolicReferGame
import torch
import egg.core as core
from egg.core.util import move_to
import numpy as np
from utils import create_dir_for_file


# ================ First play the recon game, get the sender ==================
ref_game = SymbolicReferGame()
ref_game.train(10000) # the argument is the number of training epochs.

ref_game.game.eval() 
recon_game = SymbolicReconGame(training_log='~/GitWS/GameBias/log/recon_train_temp.txt')
optimizer = core.build_optimizer(recon_game.game.receiver.parameters())
train_loss = []
test_loss = []

for i in range(5000):
    acc_list = []
    for batch_idx, (target, label) in enumerate(recon_game.train_loader):
        optimizer.zero_grad()
        target = move_to(target, recon_game.trainer.device)
        label = move_to(label, recon_game.trainer.device)

        msg,_ = ref_game.sender(target)
        rec_out = recon_game.receiver(msg.detach())
        loss, _ = recon_game.loss(target, msg, msg, rec_out, label)
        acc_list.append(loss.mean().item())
        loss.sum().backward()
        optimizer.step()
    print('train loss:', np.mean(acc_list))    
    train_loss.append(np.mean(acc_list))

    acc_list = []
    for batch_idx, (target, label) in enumerate(recon_game.test_loader):
        recon_game.game.eval() 
        target = move_to(target, recon_game.trainer.device)
        label = move_to(label, recon_game.trainer.device)
        msg,_ = ref_game.sender(target)
        rec_out = recon_game.receiver(msg)
        loss, _ = recon_game.loss(target, msg, msg, rec_out, label)
        acc_list.append(loss.mean().item())
    print('test loss:', np.mean(acc_list))
    test_loss.append(np.mean(acc_list))
    
file_name = core.get_opts().generalisation_path
create_dir_for_file(file_name)

with open(file_name, 'w') as f:
    for i in range(len(train_loss)):
        print(str(train_loss[i]) + ',' + str(test_loss[i]), file=f)

