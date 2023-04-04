import torch
import torch.nn as nn
import numpy as np
import random
import time
from tqdm import tqdm
from third_party_code.training import *
from pruning_helper import *
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class TRAIN_CONSTANTS:
    warm_up = 10
    learning_rate = 0.6
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    def __init__():
        return
    
def train_step(model,train_loader, criterion, optimizer, epoch,warm_up,validation=True,print_frequency=100):
    '''
    Train the model with simple backpropagation for 1 Epoch (NO  pruning)
    '''
    losses = AverageMeter()
    top1 = AverageMeter()

    # switch to train mode
    model.train()

    start = time.time()
    for i, (image, target) in enumerate(tqdm(train_loader)):

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

def train_model(model,train_loader,criterion,optimizer,epochs,warm_up,print_frequency=100,pruning_routine=applyDummyPruningRoutine,pruning_rate=0.5):
    '''
    Apply training step for "Epochs" times.
    -pruning_routine = custom function to apply pruning techniques, default is a dummy function that do nothing
    
    FOR EACH EPOCH:
        1. Train the model for 1 epoch
        2. Validation/Testing
        3. Pruning
    '''

    model.to(TRAIN_CONSTANTS.device)
    initialization = copy.deepcopy(model.state_dict()) #TODO Understand better how rewind works

    for epoch in range(epochs):
        #Step1: Train step
        train_step(model,train_loader,criterion,optimizer,epoch=epoch,warm_up= warm_up,print_frequency=print_frequency)
        
        #Step2: Validate the model
        #TODO Test and Validation 

        #Step3: Apply the pruning
        pruning_routine(model,initialization,pruning_rate,train_loader)

def load_datasets_MNIST(BATCH_SIZE,transform=None):
    '''
    Here we load and prepare the data, just a simple resize should
    be enough
    '''
    if transform == None:
      transf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])
    else:
      transf = transform
    # download and create datasets
    train_dataset = datasets.MNIST(root='mnist_data',
                                   train=True,
                                   transform=transf,
                                   download=True)

    valid_dataset = datasets.MNIST(root='mnist_data',
                                   train=False,
                                   transform=transf)

    train_loader = DataLoader(dataset=train_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=True)

    valid_loader = DataLoader(dataset=valid_dataset,
                              batch_size=BATCH_SIZE,
                              shuffle=False)

    return train_dataset, valid_dataset, train_loader, valid_loader


def testing(model,loader,device="cpu"):
  correct = 0
  total = 0
  starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
  timings=np.zeros((469,1))
  i=0
  # since we're not training, we don't need to calculate the gradients for our outputs
  with torch.no_grad():
      for data in loader:
          images, labels = data[0].to(device), data[1].to(device)
          starter.record()
          # calculate outputs by running images through the network
          outputs = model(images)
          ender.record()
          # WAIT FOR GPU SYNC
          torch.cuda.synchronize()
          curr_time = starter.elapsed_time(ender)
          timings[i] = curr_time
          # the class with the highest energy is what we choose as prediction
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
          i = i+1
  mean_syn = np.sum(timings) / 469
  std_syn = np.std(timings)
  print(f'Accuracy of the network on the 10000 test images: {100 * correct // total} %')
  return mean_syn
