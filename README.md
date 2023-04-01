# sparse-conv_regroup_pytorch

This repo is a "more user friendly" version of the Sparse Conv implementation discussed in this [repo](https://github.com/VITA-Group/Structure-LTH).

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
