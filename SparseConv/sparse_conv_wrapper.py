import os
import ctypes
import sparse_conv_helper as sp_helper

PATH_TO_LIB = "./lib"
PATH_TO_SRC = "./src"

CUDA_COMPILE_COMMAND = 'nvcc -arch=sm_52  -gencode=arch=compute_50,code=sm_50 -gencode=arch=compute_52,code=sm_52  -gencode=arch=compute_60,code=sm_60  -gencode=arch=compute_61,code=sm_61 -gencode=arch=compute_70,code=sm_70  -gencode=arch=compute_75,code=sm_75 -gencode=arch=compute_80,code=sm_80 -gencode=arch=compute_80,code=compute_80 -Xptxas "-v -dlcm=ca" -shared -Xcompiler=\"-fPIC\"'
BLOCK_TILING = 64

#--------------------------------------
#--------------------------------------
#-------------HELPER FUNCTIONS---------
#--------------------------------------
#--------------------------------------

def load_src_files():
    '''
    Load the Template kernels code
    '''
    print("Loading the SparseConv CUDA library")
    f = open(os.path.join(PATH_TO_SRC,'spmm_conv_n.cu'), 'r')
    code_n = f.read()
    f.close()

    f = open(os.path.join(PATH_TO_SRC,'spmm_conv_sparse.cu'), 'r')
    code_s = f.read()
    f.close()

    f = open(os.path.join(PATH_TO_SRC,'test_template.cu'), 'r')
    code_template = f.read()
    f.close()

    return code_n,code_s,code_template

def generate_actual_code(code_n,code_s,code_template,block_ptr,kernel_ptr_sparse,   #Sparse config
                         input_height,input_width,in_channels,                      #Input  info
                         output_width,output_height,output_channels,batch_size,     #Output info
                         kernel_height,kernel_width,                                #Convolution kernel     sizes
                         vertical_padding,horizontal_padding,                       #Convolution padding    sizes
                         vertical_stride,horizontal_stride,                         #Convolution stride     sizes
                         vertical_dilation,horizontal_dilation,                     #Convolution dilation   sizes
                         nn=32,B2=16                                                #Sparse algorithm parameters (weight regrouping)
                         ):
    '''
    Take as input all the Sparse Config and the Convolution config 
    Output the custom code to accelerate the sparse conv
    '''
    print("-------------------------------------")
    #INITIALIZE THE CODE STRINGS
    code_kernel = ''
    call_kernel = ''
    code_stream_decl = ''
    
    

    #CASE A: (MULTIPLE blockptr)
    size = len(block_ptr)-1
    for i in range(size):
        print(f"Cycle {i}")
        block_kernel_size = block_ptr[i+1] - block_ptr[i] - 1
        block_kernel_size = block_kernel_size.item()
        if block_kernel_size  < 1:
            continue
        code_stream_decl += f'cudaStream_t stream_{i};\n'

        #Block.y dimension
        BLOCK_Y = batch_size // BLOCK_TILING
        if BLOCK_Y == 0:
            BLOCK_Y = 1

        if block_kernel_size % nn == 0:
            print(f"{i} => Multiple of nn => {block_kernel_size}")
            BLOCK_X = output_width*output_height*block_kernel_size//(4*nn)
            if BLOCK_X == 0: 
                BLOCK_X = 1
            code_kernel += code_n.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(kernel_height)).replace('_KERNEL_WIDTH', str(kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NN', str(nn)).replace('_NKERNEL', str(block_kernel_size)).replace('_TOT_KERNEL', str(output_channels)).replace('_spmm_conv_n', f'_spmm_conv_{i}')
            call_kernel += f'cudaStreamCreate(&stream_{i});'
            call_kernel += f'\ndim3 nblocks_{i}({BLOCK_X}, {(BLOCK_Y)});\ndim3 nthreads_{i}(32, 4);\n_spmm_conv_{i}<<<nblocks_{i}, nthreads_{i}, 0, stream_{i}>>>(input_data, output_data, {block_ptr[i]}, {block_ptr[i+1]}, kernel_ptr, kernel_map, kernel_offset, kernel_data);\n'
        else:
            print(f"{i} => NOT Multiple of nn => {block_kernel_size}")
            BLOCK_X = output_width*output_height//4
            if BLOCK_X == 0: 
                BLOCK_X = 1
            block_x = output_width*output_height*block_kernel_size//(4*nn)
            if block_x == 0: 
                block_x = 1
            assert (block_kernel_size < nn)
            code_kernel += code_n.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(kernel_height)).replace('_KERNEL_WIDTH', str(kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NN', str(block_kernel_size)).replace('_NKERNEL', str(block_kernel_size)).replace('_TOT_KERNEL', str(output_channels)).replace('_spmm_conv_n', f'_spmm_conv_{i}')
            call_kernel += f'cudaStreamCreate(&stream_{i});'
            call_kernel += f'\ndim3 nblocks_{i}({BLOCK_X}, {BLOCK_Y});\ndim3 nthreads_{i}(32, 4);\n_spmm_conv_{i}<<<nblocks_{i}, nthreads_{i}, 0, stream_{i}>>>(input_data, output_data, {block_ptr[i]}, {block_ptr[i+1]}, kernel_ptr, kernel_map, kernel_offset, kernel_data);\n'
    #CASE B:  ONLY ONE BLOCK PTR IS AVAILABLE
    if len(kernel_ptr_sparse) > 1 and len(block_ptr) == 1:
        print("INSIDE!!!!!")
        BLOCK_X = output_width*output_height*sparse_kernel_size//2
        if BLOCK_X == 0: 
            BLOCK_X = 1
        code_stream_decl += 'cudaStream_t stream_sparse;\n'
        sparse_kernel_size = len(kernel_ptr_sparse) - 1
        code_kernel += code_s.replace('_OWIDTH', str(output_width)).replace('_OHEIGHT', str(output_height)).replace('_OCHANNEL', str(output_channels)).replace('_STRIDE_HEIGHT', str(vertical_stride)).replace('_STRIDE_WIDTH', str(horizontal_stride)).replace('_PADDING_HEIGHT', str(vertical_padding)).replace('_PADDING_WIDTH', str(horizontal_padding)).replace('_KERNEL_HEIGHT', str(kernel_height)).replace('_KERNEL_WIDTH', str(kernel_width)).replace('_INPUT_HEIGHT', str(input_height)).replace('_INPUT_WIDTH', str(input_width)).replace('_DIALATION_HEIGHT', str(vertical_dilation)).replace('_DIALATION_WIDTH', str(horizontal_dilation)).replace('_INPUT_CHANNEL', str(in_channels)).replace('_BATCH_SIZE', str(batch_size)).replace('_NKERNEL', str(sparse_kernel_size)).replace('_TOT_KERNEL', str(output_channels))
        call_kernel += f'cudaStreamCreate(&stream_sparse);\ndim3 nblocks_sparse({BLOCK_X}, {BLOCK_Y});\ndim3 nthreads_sparse(32, 2);\n_spmm_conv_sparse<<<nblocks_sparse, nthreads_sparse, 0, stream_sparse>>>(input_data, output_data, kernel_ptr_sparse, kernel_map_sparse, kernel_offset, kernel_data);\n'
    
    #COMPOSE THE CODE:
    code = code_template.replace('_CODE_KERNEL', code_kernel).replace('_CODE_N', code_kernel).replace('_CALL_KERNEL', call_kernel).replace('_DECL_STREAM', code_stream_decl)
    
    #ADD THE CLEAN UP CODE SEQUENCE (eg: free resources, destroy streams...)
    cleanup = ""
    for i in range(len(block_ptr)-1):
        block_kernel_size = block_ptr[i+1] - block_ptr[i] - 1
        block_kernel_size = block_kernel_size.item()
        if block_kernel_size  < 1:
            continue

        cleanup += f"cudaStreamDestroy(stream_{i});\n"
    if len(kernel_ptr_sparse) > 1 and len(block_ptr) == 1:
        cleanup += "cudaStreamDestroy(stream_sparse);\n"

    code = code.replace("_CLEAN_UP", cleanup)

    print("END of Generation")
    print("-------------------------------------")
    return code
    
def compile_custom_code(code,lib_name):
    '''
    Write the Custom kernel on a cuda file and compile it to a shared library file
    '''
    #Create a file and write the custom code on it
    path = os.path.join(PATH_TO_SRC,lib_name)
    path_lib = os.path.join(PATH_TO_LIB,lib_name)

    file_name = path + ".cu"
    with open(file_name, 'w') as fw:
        fw.write(code)

    #Compile custom kernel with CUDA compiler
    compile_line = f'{CUDA_COMPILE_COMMAND} -o {path_lib}.so {path}.cu -O2'
    print(f"COMPILE LINE: {compile_line}")
    os.system(compile_line)
    
    #Remove the file from memory
    #os.system(f"rm {file_name}")
    
def load_custom_code_wrapper(lib_name):
    '''
    return a wrapper function with the custom code for this specific convolution
    '''
    path = os.path.join(PATH_TO_LIB,lib_name)
    file_name = path + ".so"

    _lib = ctypes.cdll.LoadLibrary(file_name)
    _lib.spmm_conv.restype = None
    _lib.spmm_conv.argtypes = [ctypes.c_void_p, #input_data
                                ctypes.c_void_p,#output_data
                                ctypes.c_void_p,#kernel_ptr
                                ctypes.c_void_p,#kernel_map
                                ctypes.c_void_p,#kernel_offset
                                ctypes.c_void_p,#kernel_data
                                ctypes.c_void_p,#kernel_ptr_sparse
                                ctypes.c_void_p #kernel_map_sparse
                                ]
    return _lib


#--------------------------------------
#--------------------------------------
#-------------WRAPPER GENERATOR--------
#--------------------------------------
#--------------------------------------

def launch_wrapper(_lib,input_data,output_data,sparse_weight:sp_helper.Weight_Regroup_Config):
    _lib.spmm_conv(  ctypes.c_void_p(input_data.data_ptr()),
                                ctypes.c_void_p(output_data.data_ptr()),
                                ctypes.c_void_p(sparse_weight.kernel_ptr.data_ptr()),
                                ctypes.c_void_p(sparse_weight.kernel_map.data_ptr()),
                                ctypes.c_void_p(sparse_weight.kernel_offset.data_ptr()),
                                ctypes.c_void_p(sparse_weight.kernel_value.data_ptr()),
                                ctypes.c_void_p(sparse_weight.kernel_ptr_sparse.data_ptr()),
                                ctypes.c_void_p(sparse_weight.kernel_map_sparse.data_ptr()))

def gen_custom_sparse_conv_kernel(layer_name,
                                block_ptr,kernel_ptr_sparse,
                                input_height,input_width,in_channels,                      #Input  info
                                output_width,output_height,output_channels,batch_size,     #Output info
                                kernel_height,kernel_width,                                #Convolution kernel     sizes
                                vertical_padding,horizontal_padding,                       #Convolution padding    sizes
                                vertical_stride,horizontal_stride,                         #Convolution stride     sizes
                                vertical_dilation,horizontal_dilation,                     #Convolution dilation   sizes
                                nn=32,B2=16                                                #Sparse algorithm parameters (weight regrouping)
                                ):
    '''
    Generate the Custom code for this kernel and compile it then return a wrapper to call it from python
    '''
    #TODO Check if code already available in "lib" folder

    #Load the template code for sparse conv kernels
    code_n,code_s,code_template = load_src_files() 

    #Execute the transformation to customize it
    code = generate_actual_code(code_n,code_s,code_template,
                                block_ptr,kernel_ptr_sparse,
                                input_height,input_width,in_channels,                      #Input  info
                                output_width,output_height,output_channels,batch_size,     #Output info
                                kernel_height,kernel_width,                                #Convolution kernel     sizes
                                vertical_padding,horizontal_padding,                       #Convolution padding    sizes
                                vertical_stride,horizontal_stride,                         #Convolution stride     sizes
                                vertical_dilation,horizontal_dilation,                     #Convolution dilation   sizes
                                nn=32,B2=16                                                #Sparse algorithm parameters (weight regrouping)
                                )

    #Compile the code
    compile_custom_code(code,layer_name)

    #Load the wrapper
    wrapper = load_custom_code_wrapper(layer_name)

    return wrapper