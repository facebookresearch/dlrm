#!/bin/bash

#SBATCH --job-name=testdlrm   #The name you want the job to have
#SBATCH --output=/private/home/hongzhang/tmp/dlrm/output-%j
#SBATCH --error=/private/home/hongzhang/tmp/dlrm/error-%j
#SBATCH --nodes=1 # -C volta32gb    #The number of compute nodes to use
#SBATCH --ntasks=8     #The total number of cpu tasks to run
#SBATCH --time=00:40:00  # max time
#SBATCH --exclusive       # exclusive nodes
#SBATCH --gres=gpu:volta:8 -C volta32gb
#SBATCH --mem-per-cpu=60GB

# for mpirun host file
echo $SLURM_NODELIST
echo $SLURM_NODELIST > hostfile1

source /private/home/hongzhang/.zshrc
#module purge
#module load anaconda3/2019.07
#module load cuda/10.1
#module load cudnn/v7.6.5.32-cuda.10.1
#module load openmpi/4.0.2/gcc.7.4.0-cuda.10.1

#export NCCL_ROOT_DIR=/private/home/hongzhang/codes/nccl/build
#export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:$NCCL_ROOT_DIR/lib
#export CUDA_PATH=$CUDA_HOME
#export CUDNN_PATH=$CUDNN_ROOT_DIR
#export MPI_PATH=$MPI_HOME
#export NCCL_PATH=$NCCL_ROOT_DIR

conda activate mytorch

which python3

# large_arch_emb="2600-2600-2600-2600-2600-2600-2600-2600"
# large_arch_emb="26000000-26000000-26000000-26000000-26000000-26000000-26000000-26000000"
large_arch_emb_usr=$(printf '260%.0s' {1..815})
large_arch_emb_usr=${large_arch_emb_usr//"02"/"0-2"} 
large_arch_emb_ads=$(printf '140%.0s' {1..544}) 
large_arch_emb_ads=${large_arch_emb_ads//"01"/"0-1"}
large_arch_emb="$large_arch_emb_usr-$large_arch_emb_ads"

# --hostfile hostfile1
# random
# /public/apps/openmpi/4.0.2/gcc.7.4.0/bin/mpirun -prefix /public/apps/openmpi/4.0.2/gcc.7.4.0/ -v -np 8 python3 dlrm_s_pytorch.py --arch-sparse-feature-size=64 --arch-mlp-bot="2000-1024-1024-1024-1024-1024-1024-1024-1024-1024-1024-512-64" --arch-mlp-top="4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-1" --arch-embedding-size=$large_arch_emb --data-generation=random --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=2048 --print-freq=1 --print-time --test-mini-batch-size=10240 --test-num-workers=16 --use-gpu --dist-backend='nccl' --num-indices-per-lookup-fixed=1 --num-indices-per-lookup=30 --num-batches=4 --arch-project-size=30

# fb_synthetic
/public/apps/openmpi/4.0.2/gcc.7.4.0/bin/mpirun -prefix /public/apps/openmpi/4.0.2/gcc.7.4.0/ -v -np 8 python3 dlrm_s_pytorch.py --arch-sparse-feature-size=64 --arch-mlp-bot="2000-1024-1024-1024-1024-1024-1024-1024-1024-1024-1024-512-64" --arch-mlp-top="4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-1" --arch-embedding-size=$large_arch_emb --data-generation=synthetic --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=2048 --print-freq=1 --print-time --test-mini-batch-size=10240 --test-num-workers=16 --use-gpu --dist-backend='nccl' --num-indices-per-lookup-fixed=1 --num-indices-per-lookup=28 --num-batches=4 --arch-project-size=30

# srun --label /private/home/hongzhang/.conda/envs/mytorch/bin/python3 dlrm_s_pytorch.py --arch-sparse-feature-size=64 --arch-mlp-bot="2000-1024-1024-1024-1024-1024-1024-1024-1024-1024-1024-512-64" --arch-mlp-top="4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-4096-1" --arch-embedding-size=$large_arch_emb --data-generation=random --loss-function=bce --round-targets=True --learning-rate=0.1 --mini-batch-size=128 --print-freq=1 --print-time --test-mini-batch-size=10240 --test-num-workers=16 --use-gpu --dist-backend='nccl' --num-indices-per-lookup-fixed=1 --num-indices-per-lookup=30 --num-batches=4
