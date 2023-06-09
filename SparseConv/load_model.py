
import sparse_conv_v2 as sp
import copy

import torch.nn as nn
import torch
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import copy
from training_helper import *
import torch.optim as optim
import torch.optim as optim

class VGG16(sp.SparseModel):
    """
    A standard VGG16 model
    """

    def __init__(self, n_classes,sparse_conv_flag=True):
        self._sparse_conv_flag=sparse_conv_flag
        super(VGG16, self).__init__(sparse_conv_flag)

        self.layer1 = nn.Sequential(
            self.conv(1, 64, kernel_size=3, stride=1, padding=1),
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
            nn.Linear(512, 4096), #7*7*512 if input is 227
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
        return probs

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

N_CLASSES       = 10
IMG_SIZE        = 32
BATCH_SIZE      = 64
INPUT_CHANNELS  = 1
PRUNING_PARAMETER = 0.90

INPUT_SHAPE = (BATCH_SIZE,INPUT_CHANNELS,IMG_SIZE,IMG_SIZE)

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

#Declaring the Model
model = VGG16(N_CLASSES,sparse_conv_flag=True)
model = model.to(device)
#------------------------------------------
#------------------------------------------
#----------TRAINING-------------------------
#------------------------------------------
#------------------------------------------
model._set_sparse_layers_mode(sp.Sparse_modes.Training)
train_dataset, valid_dataset, train_loader, valid_loader = load_datasets_MNIST(BATCH_SIZE)
#DEFINE LOSS FUNCTION
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
train_model(model,train_loader,criterion,optimizer,epochs=2,warm_up=0,print_frequency=300,pruning_routine=applyDummyPruningRoutine)

#PRUNE THE MODEL TO ADD SPARSITY
print("--------------------------------------")
print(f"-----Pruning the Network at [{PRUNING_PARAMETER}]-----")
print("--------------------------------------")
#pruning_model_random(model,PRUNING_PARAMETER)

#Step1:
initialization = copy.deepcopy(model.state_dict()) #TODO Understand better how rewind works
pruning_model(model, PRUNING_PARAMETER, conv1=False)
remain_weight = check_sparsity(model, conv1=False)
#Step2:
current_mask = extract_mask(model.state_dict())
current_mask_copy = copy.deepcopy(current_mask)
#Step3:
for m in tqdm(current_mask_copy):
    mask = current_mask_copy[m]
    shape = mask.shape
    current_mask[m] = regroup(mask.view(mask.shape[0], -1)).view(*shape)
#Step4:
remove_prune(model, conv1=False)
#Step5:
model.load_state_dict(initialization)
#Step6:
prune_model_custom(model, current_mask)
#Step7:
check_sparsity(model, conv1=False)

print(f"W1 => {model.conv2.weight}")
print(f"W1 => {model.conv2.weight_orig}")
print(f"W1 => {model.conv2.weight_mask}")

train_model(model,train_loader,criterion,optimizer,epochs=1,warm_up=0,print_frequency=300,pruning_routine=applyDummyPruningRoutine)

model._set_sparse_layers_mode(sp.Sparse_modes.Inference_Vanilla)
print("Time Execution for golden PRE INIT = ",testing(model,valid_loader,device))

#SET MODEL IN TESTING MODE (For each SparseConv compare Conv2D with SparseConv2D)
print("----------------------------------")
print("-----Initialize the Network-------")
print("----------------------------------")
check_sparsity(model,conv1=False)
model._initialize_sparse_layers(input_shape=INPUT_SHAPE,use_vanilla_weights=True,force_load_code=False)
path = "./saved_models/lenet.par"

model._set_sparse_layers_mode(sp.Sparse_modes.Inference_Vanilla)
print("Time Execution for golden POST INIT = ",testing(model,valid_loader,device))
model._set_sparse_layers_mode(sp.Sparse_modes.Inference_Sparse)
print("Time Execution for sparse POST INIT= ",testing(model,valid_loader,device))

torch.save(model.state_dict(),path)

#------------------------------------------
#------------------------------------------
#----------LOADING BACK-------------------------
#------------------------------------------
#------------------------------------------
#Declaring the Model
model = LeNet5(N_CLASSES,sparse_conv_flag=True)
model = model.to(device)


#Loading Sparse Weights
path = "./saved_models/lenet.par"
sp.load_sparse_weights(model,path)

NUM_OF_TEST = 1000
loader = valid_loader
model._set_sparse_layers_mode(sp.Sparse_modes.Inference_Vanilla)
print("Time Execution for golden = ",testing(model,loader,device))
model._set_sparse_layers_mode(sp.Sparse_modes.Inference_Sparse)
print("Time Execution for sparse = ",testing(model,loader,device))
