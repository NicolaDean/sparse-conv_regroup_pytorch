from training_helper import *


import torch.nn.utils.prune as prune
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import torch.nn.init as init
from torchsummary import summary
import torch.optim as optim

class VGG16(nn.Module):
    """
    A standard VGG16 model
    """

    def __init__(self, n_classes,sparse_conv_flag=True):
        self._sparse_conv_flag=sparse_conv_flag
        super(VGG16, self).__init__()

        self.layer1 = nn.Sequential(
            nn.Conv2d(1, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU())
        self.layer2 = nn.Sequential(
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(), 
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer3 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU())
        self.layer4 = nn.Sequential(
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer5 = nn.Sequential(
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer6 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU())
        self.layer7 = nn.Sequential(
            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer8 = nn.Sequential(
            nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer9 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer10 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.layer11 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer12 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU())
        self.layer13 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2, stride = 2))
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(512, 4096),
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
    #Required only for benchmark Conv2d vs SparseConv
    def setSparseConvUsage(self,usage=True):
      self.layer1[0].use_sparse = usage
      self.layer2[0].use_sparse = usage
      self.layer3[0].use_sparse = usage
      self.layer4[0].use_sparse = usage
      self.layer5[0].use_sparse = usage
      self.layer6[0].use_sparse = usage
      self.layer7[0].use_sparse = usage
      self.layer8[0].use_sparse = usage
      self.layer9[0].use_sparse = usage
      self.layer10[0].use_sparse = usage
      self.layer11[0].use_sparse = usage
      self.layer12[0].use_sparse = usage
      self.layer13[0].use_sparse = usage

    def make_weights_sparse(self):
      '''
      Allow the convolution to compute the sparse representation of the weights
      '''
      self.layer1[0].load()
      self.layer2[0].load()
      self.layer3[0].load()
      self.layer4[0].load()
      self.layer5[0].load()
      self.layer6[0].load()
      self.layer7[0].load()
      self.layer8[0].load()
      self.layer9[0].load()
      self.layer10[0].load()
      self.layer11[0].load()
      self.layer12[0].load()
      self.layer13[0].load()
def load_datasets(BATCH_SIZE):
    '''
    Here we load and prepare the data, just a simple resize should
    be enough
    '''
    transf = transforms.Compose([transforms.Resize((32, 32)), transforms.ToTensor()])

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


train_dataset, valid_dataset, train_loader, valid_loader = load_datasets(BATCH_SIZE=128)

N_CLASSES = 10
LEARNING_RATE = 0.001
SPARSITY_LEVEL = 0.5  #TODO (Integrate into the train function)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
TRAIN_CONSTANTS.device = device
print(f"DEVICE IN USE{device}")

model = VGG16(N_CLASSES,sparse_conv_flag=False)

criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9)

#Step1. Train the model in the classic way:
train_model(model,train_loader,criterion,optimizer,1,warm_up=0,print_frequency=100,pruning_routine=applyDummyPruningRoutine)
golden_model = copy.deepcopy(model) #Copy the classic model

#Step2. FineTune the model using pruning at each epoch (USE FIP + regroup)
model = copy.deepcopy(golden_model)
train_model(model,train_loader,criterion,optimizer,2,warm_up=0,print_frequency=100,pruning_routine=applyPruningRegroup)
refill_model = copy.deepcopy(model) #Copy the regroup model


#Step3. FineTune the model using pruning at each epoch (USE FIP + refill)
model = copy.deepcopy(golden_model)
train_model(golden_model,train_loader,criterion,optimizer,2,warm_up=0,print_frequency=100,pruning_routine=applyPruningRefill)
refill_model = copy.deepcopy(model) #Copy the refill model

#Step4. Compare Golden vs Refill vs Regroup