import torch
import torch.nn as nn
import numpy as np
import random
import time

from third_party_code.training import *
from pruning_helper import *

class TRAIN_CONSTANTS:
    warm_up = 10
    learning_rate = 0.6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def __init__():
        return
    
def train_step(model,train_loader, criterion, optimizer, epoch,warm_up,validation=True,print_frequency=100):
    '''
    Train the model with simple backpropagation (NO  pruning)
    '''
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(train_loader):

        if epoch < warm_up:
            warmup_lr(epoch, i+1, optimizer, one_epoch_step=len(train_loader))

        image = image.to(TRAIN_CONSTANTS.device)
        target = target.to(TRAIN_CONSTANTS.device)

        # compute output
        output_clean = model(image)
        loss = criterion(output_clean, target)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output_clean.float()
        loss = loss.float()
        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        losses.update(loss.item(), image.size(0))
        top1.update(prec1.item(), image.size(0))

        if i % print_frequency == 0:
            end = time.time()
            print('Epoch: [{0}][{1}/{2}]\t'
                'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                'Accuracy {top1.val:.3f} ({top1.avg:.3f})\t'
                'Time {3:.2f}'.format(
                    epoch, i, len(train_loader), end-start, loss=losses, top1=top1))
            start = time.time()

    print('train_accuracy {top1.avg:.3f}'.format(top1=top1))

    return top1.avg

def train_model(model,train_loader,criterion,optimizer,epochs,warm_up,print_frequency=100,pruning_routine=applyDummyPruningRoutine,pruning_rate=0.6):
    '''
    Apply training step for "Epochs" times.
    -pruning_routine = custom function to apply pruning techniques, default is a dummy function that do nothing
    
    FOR EACH EPOCH:
        1. Train the model for 1 epoch
        2. Validation/Testing
        3. Pruning
    '''
    for epoch in range(epochs):
        #Step1: Train step
        train_step(model,train_loader,criterion,optimizer,epoch=epoch,print_frequency=print_frequency)
        
        #Step2: Validate the model
        #TODO Test and Validation 

        #Step3: Apply the pruning
        initialization = 0 #TODO (UNDERSTAND WHAT INITIALIZATION IS IN THE CHENG CODE)
        pruning_routine(model,initialization,pruning_rate)



