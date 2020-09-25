
# DLRM Distributed Branch

Extend the PyTorch implementation to run DLRM on multi nodes on distributed platforms.
The distributed version will be needed when data model becomes large.

It inherents all the parameters from master DLRM implementation. 
The distributed version add one more parameter:

**--dist-backend**:
   The backend support for the distributed version. As in torch.distributed package,
   it can be "nccl", "mpi", and "gloo".

In addition, it introduces the following new parameter::
**--arch-project-size** : 
   Reducing the number of interaction features for the dot operation. 
   A project operation is applied to the dotted features to reduce its dimension size.
   This is mainly due to the memory concern. It reduces the memory size needed for top MLP. 
   A side effect is that it may also imrpove the model accuracy.

## Usage

Currently, it is launched with mpirun on multi-nodes. The hostfile need to be created or 
a host list should be given. The DLRM parameters should be given in the same way as single
node master branch.
```bash
mpirun -np 128 -hostfile hostfile python dlrm_s_pytorch.py ...
```

## Example
```bash
python dlrm_s_pytorch.py 
   --arch-sparse-feature-size=128 
   --arch-mlp-bot="2000-1024-1024-128" 
   --arch-mlp-top="4096-4096-4096-1" 
   --arch-embedding-size=$large_arch_emb 
   --data-generation=random 
   --loss-function=bce 
   --round-targets=True 
   --learning-rate=0.1 
   --mini-batch-size=2048 
   --print-freq=10240 
   --print-time 
   --test-mini-batch-size=16384 
   --test-num-workers=16
   --num-indices-per-lookup-fixed=1 
   --num-indices-per-lookup=100
   --arch-projection-size 30
   --use-gpu
```

