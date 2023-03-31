
#System Modules
import math
from enum import Enum

#TORCH STUFF
import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.common_types import _size_1_t, _size_2_t, _size_3_t
from torch.nn.modules.utils import _single, _pair, _triple, _reverse_repeat_tuple

#Custom Modules
import sparse_conv_wrapper as sp 
import sparse_conv_helper as sp_helper

#Define the possible layer modes
class Sparse_modes(Enum):
        Training                = 1 #Execute conv by using Vanilla implementation
        Inference_Vanilla       = 2 #Execute conv by using Vanilla implementation
        Inference_Sparse        = 3 #Execute conv by using Sparse implementation
        Test                    = 4 #Check correctness of the Sparse output
        Benchmark               = 5 #Print execution time of the Vanilla vs Sparse implementation
        Calibration             = 6 #Execute benchmark and chose best mode to use on this layer with a specific batch size


class SparseConv2D(torch.nn.Conv2d):
  nn = 32
  B2 = 16

  def __init__(self, in_channels: int, out_channels: int, kernel_size: _size_2_t, stride: _size_2_t  = 1, padding: _size_2_t = 0, dilation: _size_2_t = 1, bias: bool = None):
    self.use_sparse = False
    self.filename = ""
    groups = 1
    self.mode = Sparse_modes.Training
    super(SparseConv2D, self).__init__(in_channels,out_channels,kernel_size,stride,padding,dilation,groups,None)#Conv2D init function

    self._lib = None
    self.sparse_weight = None

    self.in_channels = in_channels
    self.out_channels = out_channels
    self.kernel_size = kernel_size
    self.stride = stride
    self.padding = padding
    self.dilation = dilation
    self.bias = bias
    print(f"OUT CHANNELS: {self.out_channels}")

  def initialize_layer(self,name,use_vanilla_weights=False):
    print(f"Initialize weights of layer: {name}")
    self.name = name
    if use_vanilla_weights:
         self.sparse_weight = sp_helper.Weight_Regroup_Config(self.weight)
    else:
      self.sparse_weight = sp_helper.Weight_Regroup_Config(self.weight_origin,self.weight_mask)
    #TODO
    #INITIALIZE CUSTOM KERNEL
    #INITIALIZE REGROUPING
  
  def set_mode(self,mode=Sparse_modes.Training):
    '''
    This method change the configuration of inference of this layer
    1.Training                => Like nn.Conv2d
    2.Inference_Vanilla       => Like nn.Conv2d
    3.Inference_Sparse        => Sparse Convolution
    ... (Se Sparse_modes documentation)
    '''
    self.mode =  mode

  def benchmark(self, input:Tensor,print_flag=True) ->Tensor:
    starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)
    #Set mode to sparse
    self.set_mode(Sparse_modes.Inference_Sparse)
    
    #--------------------------------------------------
    #Compute nn.Conv2D output
    #--------------------------------------------------
    starter.record()
    vanilla_out = super().forward(input)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    vanilla_time = starter.elapsed_time(ender)
    if print_flag:
            print("\033[93m")
            print(f"TIME-Vanilla: {vanilla_time} ms")

    #--------------------------------------------------
    #Compute the SparseConv2D output
    #--------------------------------------------------
    starter.record()
    sparse_out  = self.forward(input)
    ender.record()
    # WAIT FOR GPU SYNC
    torch.cuda.synchronize()
    sparse_time = starter.elapsed_time(ender)
    if print_flag:
            print(f"TIME-Sparse : {sparse_time} ms")
            print("\033[0m")
    if print_flag:
            print("-----------------------------")
    self.set_mode(Sparse_modes.Benchmark)
    return sparse_out,vanilla_time,sparse_time
  
  def test_behaviour(self, input:Tensor,print_flag=True) ->Tensor:
                '''
                Check if the Sparse Layer and Conv2D layer has same behaviour
                '''
                #Set mode to sparse
                self.set_mode(Sparse_modes.Inference_Sparse)
                #Compute nn.Conv2D output
                vanilla_out = super().forward(input)
                #Compute the SparseConv2D output
                sparse_out  = self.forward(input)

                #Comparing the Output
                print("Vanilla vs SparseConv:")
                comparison = sparse_out.eq(vanilla_out)
                comparison = comparison.to("cpu")
                if torch.all(comparison):
                        self.set_mode(Sparse_modes.Test)
                        if print_flag:
                                print(f"\033[92mSUCCESS [{self.name}] => Same Outputs\033[0m")
                        #print(comparison)
                        return sparse_out
                else:
                        if print_flag:
                                print(f"\033[91mFAIL [{self.name}] => Divergent Outputs\033[0m")
                        print(f"Vanilla:{vanilla_out}")
                        print(f"Sparse:{sparse_out}")
                        #plt.imshow(comparison.numpy())
                        #plt.colorbar()
                        #plt.show()
                        raise Exception(f"\033[91mFAILED TEST SPARSE BEHAVIOUR ON LAYER [{self.name}]=> Divergent Outputs\033[0m") 

                        return False
  
  def forward(self, input: Tensor) -> Tensor:  # input: HWCN
    
    if self.mode == Sparse_modes.Training or self.mode == Sparse_modes.Inference_Vanilla:
      return super().forward(input) #IF WE ARE IN TRAINING USE THE CLASSIC CONV2D forward to build the weights
    elif self.mode == Sparse_modes.Test:
      return self.test_behaviour(input,print_flag=True)
    elif self.mode == Sparse_modes.Benchmark:
      return self.benchmark(input,print_flag=True)[0]
    '''
    elif self.mode == Sparse_modes.Calibration:
        return self.layer_mode_calibration(input,print_flag=True)
    '''

    #-------------_SPARSE CONV CODE--------------------------------------

    #Convert Input from NCHW to HWCN format  ( N => None batch size)
    input = input.transpose(0, 3).transpose(1, 2).transpose(0, 1)
    
    #TODO WRITE THIS IN A CLEANER MORE READABLE WAY

    if self.name == None:
      raise Exception("\033[91mSHOULD RUN THE sp.SparseConv2D.initialize_layer(...) function before any inference can be done\n Can also use the sp.SparseModel.init_sparse(..) method that will do all automatically for each sparse layer\033[0m") 
    
    if self.sparse_weight == None:
      print(f"Regrouping weights for layer: {self.name}")
      self.sparse_weight = sp_helper.Weight_Regroup_Config(self.weight)

    #Kernel Shape
    kernel_h = self.weight.shape[2]
    kernel_w = self.weight.shape[3]
      
    #Input size
    in_height  = input.shape[0]
    in_width   = input.shape[1]
    batch_size = input.shape[3]
    
    #Output size
    output_h = math.floor((in_height + 2 * self.padding - (self.dilation * (kernel_h - 1) + 1)) / self.stride + 1)
    output_w = math.floor((in_width  + 2 * self.padding - (self.dilation * (kernel_w - 1) + 1)) / self.stride + 1)
    
    if self._lib == None:
      print("COMPILE CODE")
      self._lib = sp.gen_custom_sparse_conv_kernel(self.name,self.sparse_weight.block_ptr,self.sparse_weight.kernel_ptr_sparse,in_height,in_width,self.in_channels,output_h,output_h,self.out_channels,batch_size,kernel_h,kernel_w,self.padding,self.padding,self.stride,self.stride,self.dilation,self.dilation,SparseConv2D.nn,SparseConv2D.B2)

    #Malloc the output vector
    
    output = torch.zeros(output_h, output_w, self.out_channels, batch_size).cuda()
    print(f"Output shape: {output.shape}")
    input = input.cuda()
    #Execute sparse Convolution
    sp.launch_wrapper(self._lib,input,output,self.sparse_weight)

    #Put output back to NCHW format
    return output.transpose(0, 1).transpose(0, 3).transpose(1, 2)
  

class SparseModel(nn.Module):
        '''
        This custom model simply help with the usage of Models that integrate our custom SparseConv module
        Contain helper method to initialize the sparseConv layers and change inference mode from vanilla to sparse (and opposite)
        '''
        def __init__(self, sparse_conv_flag=True):
                self._sparse_conv_flag=sparse_conv_flag
                super(SparseModel, self).__init__()
                if sparse_conv_flag:
                        self.conv = SparseConv2D
                else:
                        self.conv = nn.Conv2d

        def _set_sparse_layers_mode(self,mode=Sparse_modes.Training):
                '''
                Set the mode of all SparseConv2D layer in the net to "mode"
                '''
                for name, m in self.named_modules():
                        if isinstance(m, SparseConv2D):
                                m.set_mode(mode)

        def _initialize_sparse_layers(self,input_shape,use_vanilla_weights=False):
                #Give each layer a unique name (to compile the custom kernel file or load it back from memory)
                for name, m in self.named_modules():
                  if isinstance(m, SparseConv2D):
                    m.initialize_layer(name,use_vanilla_weights)
                
                #Change the Network mode to sparse
                self._set_sparse_layers_mode(mode = Sparse_modes.Inference_Sparse)
                #Generate a dummy input
                dummy_input = torch.randn(input_shape[0], input_shape[1],input_shape[2],input_shape[3], dtype=torch.float).cuda()
                dummy_input = dummy_input.cuda()
                #Do a forword so that all sparseConv layer initialize the CSR kernel and stuff
                self.forward(dummy_input)

        def forward(self,input: Tensor) -> Tensor:
                #IMPLEMENT IT AS YOU LIKE
                return
