#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
First play recon game, get messages, then use the messages to play referential
game. Record the accuracy.
"""

from recon_game import SymbolicReconGame
from refer_game import SymbolicReferGame
import torch
import egg.core as core
from egg.core.util import move_to
import numpy as np
from utils import create_dir_for_file


# ================ First play the recon game, get the sender ==================
recon_game = SymbolicReconGame() # NOTE: training log needs to be specified by 
recon_game.train(10000) # the argument is the number of training epochs.

recon_game.game.eval() 
ref_game = SymbolicReferGame(training_log='~/GitWS/GameBias/log/refer_train_temp.txt')
optimizer = core.build_optimizer(ref_game.game.receiver.parameters())
train_acc = []
test_acc = []

for i in range(5000):
    # train model with train set
    acc_list = []
    for batch_idx, (target, label, candidate) in enumerate(ref_game.train_loader):
        optimizer.zero_grad()
        target = move_to(target, ref_game.trainer.device)
        label = move_to(label, ref_game.trainer.device)
        candidate = move_to(candidate, ref_game.trainer.device)
        
        msg,_ = recon_game.sender(target)
        #msg,_ = ref_game.sender(target)
        rec_out = ref_game.receiver(msg.detach(), candidate)
        loss, acc = ref_game.loss(target, msg, candidate, rec_out, label)
        loss.backward()
        optimizer.step()
        acc_list.append(acc['acc'].item())
    print('train acc:', np.mean(acc_list))    
    train_acc.append(np.mean(acc_list))

    # test generalisation with valid set
    acc_list = []
    for batch_idx, (target, label, candidate) in enumerate(ref_game.test_loader):
        ref_game.game.eval()
        target = move_to(target, ref_game.trainer.device)
        label = move_to(label, ref_game.trainer.device)
        candidate = move_to(candidate, ref_game.trainer.device)

        msg,_ = recon_game.sender(target)
        rec_out = ref_game.receiver(msg, candidate)
        loss, acc = ref_game.loss(target, msg, candidate, rec_out, label)
        acc_list.append(acc['acc'].item())
    print('test acc is ', np.mean(acc_list))
    test_acc.append(np.mean(acc_list))

file_name = core.get_opts().generalisation_path
create_dir_for_file(file_name)

with open(file_name, 'w') as f:
    for i in range(len(train_acc)):
        print(str(train_acc[i]) + ',' + str(test_acc[i]), file=f)
