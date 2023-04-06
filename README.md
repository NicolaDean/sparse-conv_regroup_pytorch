# sparse-conv_regroup_pytorch

This repo is a revisited version of the Sparse Conv implementation discussed in this [repo](https://github.com/VITA-Group/Structure-LTH).

Their work aim at combining a particular pruning technique (The IMP + regroup of weights) with a Sparse Convolution implementation that use a custom sparse compression data structure (weight regrouping). 

# INFO FROM THE ORIGINAL WORK:
Their repository: https://github.com/VITA-Group/Structure-LTH

Correlated Paper: https://arxiv.org/pdf/2202.04736.pdf

Inspiration Paper of their work (Prof. Peng Jiang) : https://pengjiang-hpc.github.io/papers/rumi2020.pdf

(original Sparse Convolution author is: Prof. Peng Jiang writer of the last paper)

# Working principle

### Pruning:
1. Prune a model using IMP (iterative pruning)
2. Extract a mask of 0 and 1 (unrelevant vs relevant weights)
3. Use the regrouping technique to reshape this mask by optimizing the positioning of sparse areas 
3. [Explanation => This will help having part of the matrix full sparse and part full dense]
4. Put back original weights
5. Prune again using the new custom mask with regrouping

### Sparse Convolution:
Their sparse Convolution optimization will do GEMM matrix multiplication only on the Dense submatrix extracted with the regrouping technique

# What we added?
We have not added any additional features to the algorithm but we have rewrite their custom layer to more user friendly and easy to integrate on existing models.
We added different helper functions to automatize various things:
### Rewritten the SparseConv2D layer to have more organized/Clean CUDA code generation and usage. (easy to riciclate old models)
```python
import sparse_conv_v2 as sp

class LeNet5(sp.SparseModel):
    """
    A standard LeNet5 model
    """
    
    #sparse_conv_flag = False => self.conv will be a nn.Conv2d
    #sparse_conv_flag = False => self.conv will be a sp.SparseConv2d
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
```
### Added various Helper functions to make it easy creating a model that use the SparseConv2D layer 
For example we can make all SparseConv2D layer behave in different modes, or work like a vanilla Convolution or test correctes of its output etc..

```python
model = LeNet5(num_classes=10)
#Initialize regrouping of all layers by simply calling a method
model._initialize_sparse_layers(input_shape=INPUT_SHAPE,use_vanilla_weights=True)

#Can set all sparse layer modes as easy as calling a single methon
model._set_sparse_layers_mode(sp.Sparse_modes.Test)             #Test behaviour of each individual layer 
model._set_sparse_layers_mode(sp.Sparse_modes.Inference_Sparse) #Do inference Sparse
model._set_sparse_layers_mode(sp.Sparse_modes.Inference_Vanilla) #Behave like a normal nn.Conv2D
model._set_sparse_layers_mode(sp.Sparse_modes.Benchmark)        #Test benchmark of each individual layer

[... SOME CODE ...]
model.forward(input)
```
### Sparse model modes available:
```python
#Define the possible layer modes
class Sparse_modes(Enum):
        Training                = 1 #Execute conv by using Vanilla implementation
        Inference_Vanilla       = 2 #Execute conv by using Vanilla implementation
        Inference_Sparse        = 3 #Execute conv by using Sparse implementation
        Test                    = 4 #Check correctness of the Sparse output
        Benchmark               = 5 #Print execution time of the Vanilla vs Sparse implementation
        Calibration             = 6 #Execute benchmark and chose best mode to use on this layer with a specific batch size

```
