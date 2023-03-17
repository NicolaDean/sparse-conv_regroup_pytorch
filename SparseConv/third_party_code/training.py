import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import uuid

def accuracy(output, target, topk=(1,)):
    '''
    Calculate the accuracy of the model comparing outputs and label 
    -output: predictions of the model
    -target: labels
    -topk: top-k elements correctly predicted
    '''
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

def setup_seed(seed): 
    '''
    Setup the seeds for all relevant python environment (eg: Numpy, Pytorch..)
    '''
    torch.manual_seed(seed) 
    torch.cuda.manual_seed_all(seed) 
    np.random.seed(seed) 
    random.seed(seed) 
    torch.backends.cudnn.deterministic = True 

def warmup_lr(epoch, step, optimizer, one_epoch_step, warmup, learning_rate):
    '''
    Is an "Home made" Early stopping factor that change the learning rate 
    '''
    overall_steps = warmup*one_epoch_step
    current_steps = epoch*one_epoch_step + step 

    lr = learning_rate * current_steps/overall_steps
    lr = min(lr, learning_rate)

    for p in optimizer.param_groups:
        p['lr']=lr

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
