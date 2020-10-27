
# This feature can be used to reduce the memory size consumed by the feature layer of the top MLP.
# Suppose we have n sparse features, each sparse features is represented by an embedding of size d,
# then, we can represent the sparse embeddings by a matrix X = (n, d). The dot product between sparse
# features is X(X^T), which is a symmetric matrix of (n, n) and will be fed into the top MLP. 
# Actually We only need the upper or lower traingles to eliminate duplication. If n is large,
# such as, n = 1000, then the number of dot features fed into the MLP will be n^2/2 = 50,000.
# Considering the layer size 4096, the weight parameters will be a matrix (n^2/2, 4096), which
# may consume a large amount of precious memory resources.

# To reduce the number of dot features, we introduce a parameter called arch-projec-size (k) to compress
# the embeddings. We introduce a parameter matrix Y = (n, k) to compute the weighted sum of the
# dot features. The compressed embeddings is represented by (X^T)Y. Then, we compute the compressed dot 
# features by X(X^T)Y = (n, k). Therefore, we can reduce the dot features fed into MLP from n*n/2
# to n*k.

import sys
import torch
import torch.nn as nn
import numpy as np

"""
Compute the projected dot features
T: (batch_size, n, d), batched raw embeddings
x: dense features
proj_layer: the projection layer created by create_proj
"""
def project(T, x, proj_layer):

  TT = torch.transpose(T, 1, 2)
  # TS = torch.reshape(TT, (-1, TT.size(2)))
  # TC = proj_layer(TS)
  # TR = torch.reshape(TC, (-1, T.shape[2], k))
  TR = proj_layer(TT)
  Z  = torch.bmm(T, TR)
  Zflat = Z.view((T.shape[0], -1))
  R = torch.cat([x] + [Zflat], dim=1)

  return R

"""
Create the project layer
n: number of sparse features
m: projection size
"""
def create_proj(n, m):
  # build MLP layer by layer
  layers = nn.ModuleList()
  # construct fully connected operator
  LL = nn.Linear(int(n), int(m), bias=True)

  # initialize the weights
  # with torch.no_grad():
  # custom Xavier input, output or two-sided fill
  mean = 0.0  # std_dev = np.sqrt(variance)
  std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
  W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
  std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
  bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
  # approach 1
  LL.weight.data = torch.tensor(W, requires_grad=True)
  LL.bias.data = torch.tensor(bt, requires_grad=True)
  # approach 2: constant value ?
  layers.append(LL)

  return torch.nn.Sequential(*layers)

