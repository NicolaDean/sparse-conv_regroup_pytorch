import sparse_conv_v2 as sp
import copy

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import copy


class VGG16(sp.SparseModel):
    """
    A standard VGG16 model
    """

    def __init__(self, n_classes,sparse_conv_flag=True):
        self._sparse_conv_flag=sparse_conv_flag
        super(VGG16, self).__init__(sparse_conv_flag)

        self.layer1 = nn.Sequential(
            self.conv(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            self.conv(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            self.conv(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            self.conv(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            self.conv(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            self.conv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            self.conv(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            self.conv(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            self.conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            self.conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            self.conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            self.conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            self.conv(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(7*7*512, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, n_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = self.layer6(out)
        out = self.layer7(out)
        out = self.layer8(out)
        out = self.layer9(out)
        out = self.layer10(out)
        out = self.layer11(out)
        out = self.layer12(out)
        out = self.layer13(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

class LeNet5(sp.SparseModel):
    """
    A standard LeNet5 model
    """

    def __init__(self, n_classes,sparse_conv_flag=True):
        self._sparse_conv_flag=sparse_conv_flag
        super(LeNet5, self).__init__(sparse_conv_flag)


        self.conv1 = self.conv(in_channels=1, out_channels=6, kernel_size=5, stride=1,padding=1)
        self.tanh1 = nn.Tanh()
        self.pool1 = nn.AvgPool2d(kernel_size=2)
        self.conv2 = self.conv(in_channels=6, out_channels=16, kernel_size=5, stride=1)
        self.tanh2 = nn.Tanh()
        self.pool2 = nn.AvgPool2d(kernel_size=2)
        self.conv3 = self.conv(in_channels=16, out_channels=120, kernel_size=5, stride=1)
        self.tanh3 = nn.Tanh()
        self.linear1 = nn.Linear(in_features=120, out_features=84)
        self.tanh4 = nn.Tanh()
        self.linear2 = nn.Linear(in_features=84, out_features=n_classes)

    def forward(self, x):
        print("Forwarddd")
        x = self.conv1(x)
        x = self.tanh1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.tanh2(x)
        x = self.pool2(x)
        x = self.conv3(x)
        x = self.tanh3(x)
        x = torch.flatten(x, 1)
        x = self.linear1(x)
        x = self.tanh4(x)
        logits = self.linear2(x)
        probs = F.softmax(logits, dim=1)
        return logits, probs


class AlexNet(sp.SparseModel):
    def __init__(self, n_classes,sparse_conv_flag=True):
        self._sparse_conv_flag=sparse_conv_flag
        super(AlexNet, self).__init__(sparse_conv_flag)

        self.layer1 = nn.Sequential(
            self.conv(3, 96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(96),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer2 = nn.Sequential(
            self.conv(96, 256, kernel_size=5, stride=1, padding=2),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.layer3 = nn.Sequential(
            self.conv(256, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            self.conv(384, 384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU())
        self.layer5 = nn.Sequential(
            self.conv(384, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 3, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(9216, 4096),
            nn.ReLU())
        self.fc1 = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(4096, 4096),
            nn.ReLU())
        self.fc2= nn.Sequential(
            nn.Linear(4096, n_classes))
        
    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)
        out = self.layer3(out)
        out = self.layer4(out)
        out = self.layer5(out)
        out = out.reshape(out.size(0), -1)
        out = self.fc(out)
        out = self.fc1(out)
        out = self.fc2(out)
        return out

def pruning_model_random(model, px):

    parameters_to_prune =[]
    for name,m in model.named_modules():
        if isinstance(m, sp.SparseConv2D):
            print(f"Pruning layer {name}")
            parameters_to_prune.append((m,'weight'))

    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.RandomUnstructured,
        amount=px,
    ) 

#ALEXNET
N_CLASSES       = 10
IMG_SIZE        = 227
BATCH_SIZE      = 64
INPUT_CHANNELS  = 3
PRUNING_PARAMETER = 0.90  
#VGG16
N_CLASSES       = 10
IMG_SIZE        = 32
BATCH_SIZE      = 64
INPUT_CHANNELS  = 3
PRUNING_PARAMETER = 0.90
#LENET
N_CLASSES       = 10
IMG_SIZE        = 32
BATCH_SIZE      = 64
INPUT_CHANNELS  = 1
PRUNING_PARAMETER = 0.90


INPUT_SHAPE = (BATCH_SIZE,INPUT_CHANNELS,IMG_SIZE,IMG_SIZE)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
# Assuming that we are on a CUDA machine, this should print a CUDA device:
print(device)

#LOAD THE MODEL
model = LeNet5(N_CLASSES,sparse_conv_flag=True)
model.to(device)

#PRUNE THE MODEL TO ADD SPARSITY
print("--------------------------------------")
print(f"-----Pruning the Network at [{PRUNING_PARAMETER}]-----")
print("--------------------------------------")
pruning_model_random(model,PRUNING_PARAMETER)

#SET MODEL IN TESTING MODE (For each SparseConv compare Conv2D with SparseConv2D)
print("----------------------------------")
print("-----Initialize the Network-------")
print("----------------------------------")
model._initialize_sparse_layers(input_shape=INPUT_SHAPE,use_vanilla_weights=True)
model._set_sparse_layers_mode(sp.Sparse_modes.Test)

#------------------------------------------
#------------------------------------------
#----------TESTING-------------------------
#------------------------------------------
#------------------------------------------

#Generate a dummy input to give the convolution

print("----------------------------------")
print("-----Example of Benchmark or Test-------")
print("----------------------------------")
batch_size = 1
dummy_input = torch.randn(INPUT_SHAPE, dtype=torch.float).to(device)
dummy_input = dummy_input.cuda()

print(f"INPUT SHAPE : {dummy_input.shape}")
input = copy.deepcopy(dummy_input)
input = input.cuda()

model._set_sparse_layers_mode(sp.Sparse_modes.Test)
model.forward(dummy_input)
model._set_sparse_layers_mode(sp.Sparse_modes.Test)
model.forward(dummy_input)

exit()