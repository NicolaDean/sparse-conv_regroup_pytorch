
import copy
import math
import os
import uuid
from tqdm import tqdm
import numpy as np
import torch

class Weight_Regroup_Config(torch.nn.Module):

    def __init__(self,weight_origin=None,weight_mask= None,create_param = False):
        super(Weight_Regroup_Config, self).__init__()
        if create_param:
            self.block_ptr = [0]
            self.kernel_ptr = []
            self.kernel_map = []
            self.kernel_offset = []
            self.kernel_value = []

            self.kernel_ptr_sparse = []
            self.kernel_map_sparse = []
            
        #No Weights availables
        if weight_origin == None and weight_mask == None:
            self.force_vanilla_cnn = True
            return
        #Only original weights
        with  torch.no_grad():
            if weight_mask == None:
                x = weight_origin.clone()
                self.regroup_weight(x)
                return
        #Pruning Mask + original
            w = weight_origin.clone()
            w = w * weight_mask
            self.regroup_weight(w)
        
    def extract_dense(self, sparse_kernel,nn=32,B2=16):
        self.force_vanilla_cnn = False
        #return self.extract_dense_old(sparse_kernel)
        t1 = 1.5
        cn = 8

        nrows = sparse_kernel.shape[0]
        ncols = sparse_kernel.shape[1]

        print(f"Sparsity: {((sparse_kernel.abs() > 0).sum() / sparse_kernel.numel())}")
        #Find Non Empty Rows
        nonempty_rows = []
        for i in range(nrows):
            for j in range(ncols):
                if sparse_kernel[i, j] != 0:
                    nonempty_rows.append(i)
                    break
        #Find Non Empty Columns in each non empty row
        nonempty_cols = []
        for j in range(ncols):
            for i in nonempty_rows:
                if sparse_kernel[i, j] != 0:
                    nonempty_cols.append(j)
                    break
        
        #FILE FORMAT:
        #(Non empty cols)(Non empty Rows)

        #print (ncols, len(nonempty_cols))
        graph_file = f'{uuid.uuid1()}.txt'
        f = open(graph_file, 'w')
        f.write(str(len(nonempty_cols))+' '+str(len(nonempty_rows))+'\n') #Print on file num of rows and cols not empty
        #For each Non empty Column
        for j in range(len(nonempty_cols)):
            #For each non empty rows
            for i in range(len(nonempty_rows)):
                if sparse_kernel[nonempty_rows[i], nonempty_cols[j]] != 0:
                    f.write(str(i+1)+' ')
            f.write('\n')
        f.close()

        #APPLY hmetis
        res = os.system(f'./shmetis {graph_file} {cn} 10')
        print(f"RESULT OF SYS IS {res}")

        try:
            f = open(f'{graph_file}.part.{cn}', 'r')
            clusters = {}
            s = f.readlines()
        except:
            self.force_vanilla_cnn = True
            print("Something gone wrong")
            return [(list(range(nrows)), list(range(ncols)))]
        if len(s) != len(nonempty_rows):
            self.force_vanilla_cnn = True
            print("Something gone wrong case Len(s)!=len(nonemptyrows)")
            return [(list(range(nrows)), list(range(ncols)))]

        for i in range(len(s)):
            t = int(s[i].strip())
            if t not in clusters:
                clusters[t] = []
            clusters[t].append(i)
        f.close()
        #os.system(f'cat {graph_file}')
        try:
            print("Removing file")
            os.system(f'rm {graph_file}*')
        except:
            print("Failed removing files")


        clusters = [clusters[c] for c in clusters]
        clusters.sort(key=lambda x:len(x), reverse=True)
        
        #print(f"Clusters: {clusters}")
        blocks = []

        #For each partitioned ROW group
        for r in clusters:
            #FIND NON ZERO COLUMNS
            nnz_cols = [0] * ncols
            for i in range(ncols):
                s = 0
                #For each row in the rows partition count non zero elements
                for rr in r:
                    if sparse_kernel[nonempty_rows[rr],i] != 0:
                        s += 1
                nnz_cols[i] = s #For each row find the non zero elements 

            #sort the non_zero_Count of the groups by index (so cc contain indexes in order of num of non zero element in that column)
            cc = sorted(list(range(ncols)), key=lambda x:nnz_cols[x], reverse=True) 
            nnz_rows = [0] * len(r)

            for i in range(len(r)):
                for j in range(ncols):
                    if sparse_kernel[nonempty_rows[r[i]], j] != 0:
                        nnz_rows[i] += 1
            '''
            for i in range(1, ncols):
                dense_cols = cc[:i]
                flag = False
                for j in range(len(r)):
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
            '''
            for i in range(ncols):
                dense_cols = cc[:(i+1)]
                flag = False
                for j in range(len(r)):
                    if sparse_kernel[nonempty_rows[r[j]], cc[i]] != 0:
                        nnz_rows[j] -= 1

                    if i <= t1*nnz_rows[j]:
                        flag = True
                        break
                if flag == False:
                    dense_rows = [nonempty_rows[i] for i in r]
                    #print (len(dense_rows), len(dense_cols))
                    if len(dense_rows) > nn:
                        dense_rows_1 = dense_rows[:math.floor(len(dense_rows)/nn*nn)]
                        dense_rows_2 = dense_rows[math.floor(len(dense_rows)/nn*nn):]
                        blocks.append((dense_rows_1, dense_cols))
                        blocks.append((dense_rows_2, dense_cols))
                    elif len(dense_rows) >  B2:#B2 :
                        blocks.append((dense_rows, dense_cols))
                    break
            
        #print(f"Blocks = {blocks}")
        if len(blocks) > 0:
            return blocks
        else:
            self.force_vanilla_cnn = True
            print("Something goes wrong.... the num of block is 0")
            return [(list(range(nrows)), list(range(ncols)))]

    def extract_new_mask(self, sparse_kernel,nn=32,B2=16):
        blocks = self.extract_dense(sparse_kernel,nn,B2)

        new_mask = torch.zeros_like(sparse_kernel)
        if len(blocks) > 0:
            for b in blocks:
                for r in b[0]:
                    for c in b[1]:
                        new_mask[r,c] = 1
            return new_mask
        else:
            return sparse_kernel
    
    def regroup_weight(self,w,nn=32,B2=16):
        '''
        Given pruned weight and a mask it will give the Regroup Configuration
        '''
        kernel_shape = w.shape

        block_ptr = [0]
        kernel_ptr = []
        kernel_map = []
        kernel_offset = []
        kernel_value = []

        kernel_ptr_sparse = []
        kernel_map_sparse = []
        #w = copy.deepcopy(self.weight)


        sparse_weight = w.view(kernel_shape[0], kernel_shape[1] * kernel_shape[2] * kernel_shape[3])
        new_kernel = sparse_weight.clone()

        nnz = 0
        for a in sparse_weight:
            for b in a:
                if b != 0:
                    nnz += 1
            
        sparse_nnz = nnz
        with torch.no_grad():
            blocks = self.extract_dense(sparse_weight,nn,B2)

            for b in blocks:
                kernel_ptr.append(len(kernel_offset))
                for r in b[0]:
                    kernel_offset.extend(b[1])
                    kernel_value.extend(sparse_weight[r,b[1]].tolist())
                    kernel_ptr.append(len(kernel_offset))
                    kernel_map.append(r)
                    for c in b[1]:
                        if (sparse_weight[r,c] != 0):
                            sparse_weight[r, c] = 0
                            sparse_nnz -= 1
                        else:
                            new_kernel[r, c] = np.random.rand() 
                kernel_map.append(-1)
                assert (len(kernel_ptr) == len(kernel_map))
                block_ptr.append(len(kernel_ptr))
            
        kernel_ptr_sparse = []
        kernel_map_sparse = []
        nrows = sparse_weight.shape[0]
        ncols = sparse_weight.shape[1]
        kernel_ptr_sparse.append(len(kernel_offset))
        for i in range(nrows):
            empty = True
            for j in range(ncols):
                if sparse_weight[i,j]	!= 0:
                    kernel_offset.append(j)
                    kernel_value.append(sparse_weight[i,j])
                    empty = False
            if not empty:
                kernel_ptr_sparse.append(len(kernel_offset))
                kernel_map_sparse.append(i)


        #print(kernel_ptr_sparse)
        
        self.block_ptr = torch.IntTensor(block_ptr).cuda()
        self.kernel_ptr = torch.IntTensor(kernel_ptr).cuda()
        self.kernel_map = torch.IntTensor(kernel_map).cuda()
        self.kernel_offset = torch.IntTensor(kernel_offset).cuda()
        self.kernel_value = torch.FloatTensor(kernel_value).cuda()
        self.kernel_ptr_sparse = torch.IntTensor(kernel_ptr_sparse).cuda()
        self.kernel_map_sparse = torch.IntTensor(kernel_map_sparse).cuda() 

        if not self.force_vanilla_cnn:
            self.block_ptr = torch.nn.Parameter(self.block_ptr,requires_grad=False)
            self.kernel_ptr = torch.nn.Parameter(self.kernel_ptr,requires_grad=False)
            self.kernel_map = torch.nn.Parameter(self.kernel_map,requires_grad=False)
            self.kernel_offset = torch.nn.Parameter(self.kernel_offset,requires_grad=False)
            self.kernel_value = torch.nn.Parameter(self.kernel_value,requires_grad=False)
            self.kernel_ptr_sparse = torch.nn.Parameter(self.kernel_ptr_sparse,requires_grad=False)
            self.kernel_map_sparse = torch.nn.Parameter(self.kernel_map_sparse,requires_grad=False)

        self.force_vanilla_cnn = torch.BoolTensor([self.force_vanilla_cnn]).cuda()
        self.force_vanilla_cnn = torch.nn.Parameter(self.force_vanilla_cnn,requires_grad=False)
