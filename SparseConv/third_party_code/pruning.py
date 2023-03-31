import torch
import torch.nn as nn
import torch.nn.utils.prune as prune
import os
import uuid
from tqdm import tqdm

'''
    Those functions are extracted from code in github repo:
    https://github.com/VITA-Group/Structure-LTH
'''

#------------------------------------------------------------------------
#------------------------------------------------------------------------
#---------------------Chen groups METHODS--------------------------------
#------------------------------------------------------------------------
#------------------------------------------------------------------------

def pruning_model(model, px, conv1=False):
    '''
    Add a Prune layer around the Convolutions inside the
    '''
    print('start unstructured pruning for all conv layers')
    parameters_to_prune =[]
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                parameters_to_prune.append((m,'weight'))


    parameters_to_prune = tuple(parameters_to_prune)

    prune.global_unstructured(
        parameters_to_prune,
        pruning_method=prune.L1Unstructured,
        amount=px,
    )

def remove_prune(model, conv1=True):
    print('remove pruning')
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                prune.remove(m,'weight')

def check_sparsity(model, conv1=True):
    
    sum_list = 0
    zero_sum = 0

    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if name == 'conv1':
                if conv1:
                    sum_list = sum_list+float(m.weight.nelement())
                    zero_sum = zero_sum+float(torch.sum(m.weight == 0))    
                else:
                    print('skip conv1 for sparsity checking')
            else:
                sum_list = sum_list+float(m.weight.nelement())
                zero_sum = zero_sum+float(torch.sum(m.weight == 0))  

    print('* remain weight = ', 100*(1-zero_sum/sum_list),'%')
    
    return 100*(1-zero_sum/sum_list)

def extract_mask(model_dict):
    '''
    Extract the mask we need to apply to the real weights to improve the regrouping
    '''
    print("Extracting mask from layers")
    new_dict = {}
    for key in model_dict.keys():
        if 'mask' in key:
            new_dict[key] = model_dict[key]

    return new_dict

def regroup_multithread(sparse_kernel, t1 = 1.5, nn = 32, B2 = 16, cn = 8):
    '''
    Reorganize the weights to make sparse convolution optimizations
    '''
    nrows = sparse_kernel.shape[0]
    ncols = sparse_kernel.shape[1]

    nonempty_rows = []

    for i in range(nrows):
        nz = 0
        for j in range(ncols):
            if sparse_kernel[i, j] != 0:
                nonempty_rows.append(i)
                break
def regroup(sparse_kernel, t1 = 1.5, nn = 32, B2 = 16, cn = 8):
    '''
    Reorganize the weights to make sparse convolution optimizations
    '''
    nrows = sparse_kernel.shape[0]
    ncols = sparse_kernel.shape[1]

    nonempty_rows = []
    for i in range(nrows):
        nz = 0
        for j in range(ncols):
            if sparse_kernel[i, j] != 0:
                nonempty_rows.append(i)
                break
    #print (nrows, len(nonempty_rows))

    nonempty_cols = []
    for j in range(ncols):
        nz = 0
        for i in nonempty_rows:
            if sparse_kernel[i, j] != 0:
                nonempty_cols.append(j)
                break
    #print (ncols, len(nonempty_cols))
    tempname = str(uuid.uuid1())
    tmp = open(tempname, "w")
    tmp.write(str(len(nonempty_cols))+' '+str(len(nonempty_rows))+'\n')
    for j in range(len(nonempty_cols)):
        for i in range(len(nonempty_rows)):
            if sparse_kernel[nonempty_rows[i], nonempty_cols[j]] != 0:
                tmp.write(str(i+1)+' ')
        tmp.write('\n')

    tmp.close()
    
    os.system(f'./shmetis {tempname} {cn} 10')
    from glob import glob
    file_to_find = glob(f'{tempname}.part.*')
    try:
        f = open(file_to_find[0], 'r')
        clusters = {}
        s = f.readlines()
    except:
        return sparse_kernel
    #print (len(s))
    #Remove all temporary files
    os.system(f'rm {tempname}*')

    assert (len(s) == len(nonempty_rows))
    

    for i in range(len(s)):
        t = int(s[i].strip())
        if t not in clusters:
            clusters[t] = []
        clusters[t].append(i)
    f.close()

    os.system(f'rm {tmp.name}')

    clusters = [clusters[c] for c in clusters]
    clusters.sort(key=lambda x:len(x), reverse=True)
        
    blocks = []

    for r in clusters:
        nnz_cols = [0] * ncols
        for i in range(ncols):
            s = 0
            for rr in r:
                if sparse_kernel[nonempty_rows[rr],i] != 0:
                    s += 1
            nnz_cols[i] = s
        cc = sorted(list(range(ncols)), key=lambda x:nnz_cols[x], reverse=True)
        nnz_rows = [0] * len(r)

        for i in range(len(r)):
            for j in range(ncols):
                if sparse_kernel[nonempty_rows[r[i]], j] != 0:
                    nnz_rows[i] += 1


        for i in range(1, ncols):
            dense_cols = cc[:i]
            flag = False
            for j in range(len(r)):
                #print(i, j)
                #print(sparse_kernel[nonempty_rows[r[j]], i])
                #print(nnz_rows[j])
                if sparse_kernel[nonempty_rows[r[j]], i] != 0:
                    nnz_rows[j] -= 1
                if i <= t1*nnz_rows[j]:
                    flag = True
                    break
            
            if flag == False:
                dense_rows = [nonempty_rows[i] for i in r]
                #print (len(dense_rows), len(dense_cols))
                if len(dense_rows) > nn:
                    dense_rows_1 = dense_rows[:len(dense_rows)//nn*nn]
                    dense_rows_2 = dense_rows[len(dense_rows)//nn*nn:]
                    blocks.append((dense_rows_1, dense_cols))
                    blocks.append((dense_rows_2, dense_cols))
                elif len(dense_rows) > B2:
                    blocks.append((dense_rows, dense_cols))
                break

    new_mask = torch.zeros_like(sparse_kernel)
    if len(blocks) > 0:
        for b in blocks:
            for r in b[0]:
                for c in b[1]:
                    new_mask[r,c] = 1
        return new_mask
    else:
        return sparse_kernel
    
def prune_model_custom(model, mask_dict, conv1=False):
    for name,m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                print('pruning layer with custom mask:', name)
                prune.CustomFromMask.apply(m, 'weight', mask=mask_dict[name+'.weight_mask'].to(m.weight.device))

def prune_model_custom_fillback(model, mask_dict, conv1=False, criteria="remain", train_loader=None, init_weight=None, trained_weight=None, return_mask_only=False, strict=True, fillback_rate=0.0):

    feature_maps = []
    '''
    try:
        model.load_state_dict(trained_weight, strict=strict)
    except:
        for key in list(trained_weight.keys()):
            if ('mask' in key):
                trained_weight[key[:-5]] = trained_weight[key[:-5] + "_orig"] * trained_weight[key]
                del trained_weight[key[:-5] + "_orig"]
                del trained_weight[key]
        model.load_state_dict(trained_weight, strict=strict)
    '''
    def hook(module, input, output):
        feature_maps.append(output)
    
    image, label = next(iter(train_loader))
    handles = []
    masks = {}
    for name, m in model.named_modules():
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                handles.append(m.register_forward_hook(hook))
                device = m.weight.data.device
    output = model(image.to(device))
    loss = torch.nn.CrossEntropyLoss()(output, label.to(output.device))
    counter = 0

    for i, (name,m) in enumerate(model.named_modules()):
        if isinstance(m, nn.Conv2d):
            if (name == 'conv1' and conv1) or (name != 'conv1'):
                mask = mask_dict[name+'.weight_mask']
                mask = mask.view(mask.shape[0], -1)
                count = torch.sum(mask, 1) # [C]
                #sparsity = torch.sum(mask) / mask.numel()
                num_channel = (count.sum().float() / mask.shape[1]).item()
                print(num_channel)
                print(mask.shape[0])
                print(fillback_rate)
                print(mask.shape[0] - num_channel)
                print((mask.shape[0] - num_channel) * fillback_rate)
                int_channel = int(num_channel + (mask.shape[0] - num_channel) * fillback_rate)
                frac_channel = num_channel - int_channel
                print(mask.shape)
                print(int_channel)
                if criteria == 'remain':
                    print(mask.shape[0] - int_channel)
                    threshold, _ = torch.kthvalue(count, max(mask.shape[0] - int_channel, 1))
                
                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    tensor = torch.where(count == threshold)[0]
                    perm = torch.randperm(tensor.size(0))
                    idx = perm[0]
                    samples = tensor[idx]
                    mask[samples] = 1

                elif criteria == 'magnitude':
                    mask = mask_dict[name+'.weight_mask']
                    count = trained_weight[name + '.weight'].view(mask.shape[0], -1).abs().sum(1)
                    if (mask.shape[0] - int_channel) > 0:
                        threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)
                        mask[torch.where(count > threshold)[0]] = 1
                        mask[torch.where(count < threshold)[0]] = 0
                        tensor = torch.where(count == threshold)[0]
                        perm = torch.randperm(tensor.size(0))
                        idx = perm[0]
                        samples = tensor[idx]
                        mask[samples] = 1
                    else:
                        mask[:,:] = 1
                
                elif criteria == 'l1':
                    mask = mask_dict[name+'.weight_mask']
                    count = feature_maps[counter].view(mask.shape[0], -1).abs().sum(1)
                    threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)
                
                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    tensor = torch.where(count == threshold)[0]
                    perm = torch.randperm(tensor.size(0))
                    idx = perm[0]
                    samples = tensor[idx]
                    mask[samples] = 1

                elif criteria == 'l2':
                    mask = mask_dict[name+'.weight_mask']
                    count = (feature_maps[counter].view(mask.shape[0], -1).abs() ** 2).sum(1)
                    threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)
                
                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    tensor = torch.where(count == threshold)[0]
                    perm = torch.randperm(tensor.size(0))
                    idx = perm[0]
                    samples = tensor[idx]
                    mask[samples] = 1
                elif criteria == 'saliency':
                    mask = mask_dict[name+'.weight_mask']
                    count = (feature_maps[counter] * torch.autograd.grad(loss, feature_maps[counter], retain_graph=True, only_inputs=True)[0]).view(mask.shape[0], -1).abs().sum(1)
                    threshold, _ = torch.kthvalue(count, mask.shape[0] - int_channel)
                
                    mask[torch.where(count > threshold)[0]] = 1
                    mask[torch.where(count < threshold)[0]] = 0
                    tensor = torch.where(count == threshold)[0]
                    perm = torch.randperm(tensor.size(0))
                    idx = perm[0]
                    samples = tensor[idx]
                    mask[samples] = 1
                
                if not return_mask_only:
                    #m.weight.data = init_weight[name + ".weight"]
                    mask = mask.view(*mask_dict[name+'.weight_mask'].shape)
                    print('pruning layer with custom mask:', name)
                    prune.CustomFromMask.apply(m, 'weight', mask=mask.to(m.weight.device))
                else:
                    masks[name] = mask

    for h in handles:
        h.remove()
    
    if return_mask_only:
        return masks