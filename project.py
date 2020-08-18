
import sys
import torch
import torch.nn as nn
import numpy as np

def project(T, project_size, batch_size, d, x, layer):

  TT = torch.transpose(T, 1, 2)
  TS = torch.reshape(TT, (-1, TT.size(2)))
  TC = layer(TS)
  TR = torch.reshape(TC, (-1, d, project_size))
  Z  = torch.bmm(T, TR)
  Zflat = Z.view((batch_size, -1))
  R = torch.cat([x] + [Zflat], dim=1)

  return R

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

