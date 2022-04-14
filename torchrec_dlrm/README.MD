# TorchRec DLRM Example

`dlrm_main.py` trains, validates, and tests a [Deep Learning Recommendation Model](https://arxiv.org/abs/1906.00091) (DLRM) with TorchRec. The DLRM model contains both data parallel components (e.g. multi-layer perceptrons & interaction arch) and model parallel components (e.g. embedding tables). The DLRM model is pipelined so that dataloading, data-parallel to model-parallel comms, and forward/backward are overlapped. Can be run with either a random dataloader or [Criteo 1 TB click logs dataset](https://ailab.criteo.com/download-criteo-1tb-click-logs-dataset/).

It has been tested on the following cloud instance types:
| Cloud  | Instance Type       | GPUs             | vCPUs | Memory (GB) |
| ------ | ------------------- | ---------------- | ----- | ----------- | 
| AWS    | p4d.24xlarge        | 8 x A100 (40GB)  | 96    | 1152        |
| Azure  | Standard_ND96asr_v4 | 8 x A100 (40GB)  | 96    | 900         |
| GCP    | a2-megagpu-16g      | 16 x A100 (40GB) | 96    | 1300        |

# Running

## Install dependencies
`pip install tqdm torchmetrics`

## Torchx
We recommend using [torchx](https://pytorch.org/torchx/main/quickstart.html) to run. Here we use the [DDP builtin](https://pytorch.org/torchx/main/components/distributed.html)

1. pip install torchx
2. (optional) setup a slurm or kubernetes cluster
3.
    a. locally: `torchx run -s local_cwd dist.ddp -j 1x2 --script dlrm_main.py`
    b. remotely: `torchx run -s slurm dist.ddp -j 1x8 --script dlrm_main.py`

## TorchRun
You can also use [torchrun](https://pytorch.org/docs/stable/elastic/run.html).
* e.g. `torchrun --nnodes 1 --nproc_per_node 2 --rdzv_backend c10d --rdzv_endpoint localhost --rdzv_id 54321 --role trainer dlrm_main.py`


## Preliminary Training Results

**Setup:**
* Dataset: Criteo 1TB Click Logs dataset
* CUDA 11.0, NCCL 2.10.3.
* AWS p4d24xlarge instances, each with 8 40GB NVIDIA A100s.
   *  Elastic Fabric Adapter enabled.
* Shared flags for all runs include: `--pin_memory --batch_size 4096 --epochs 1 --embedding_dim 128 --dense_arch_layer_sizes "512,256,128" --over_arch_layer_sizes "1024,1024,512,256,1" --in_memory_binary_criteo_path "..."`


**Results**

|Configuration|Collective Size of Embedding Tables (GiB)|Number of GPUs|Local Batch Size|Global Batch Size|AUROC over Val Set After 1 Epoch|AUROC Over Test Set After 1 Epoch|Train Records/Second|Time to Train 1 Epoch | Unique Flags |
--- | --- | --- | --- | --- | --- | --- | --- | --- | ---
|MLPerf|97.36|8|8192|65536|0.7736921906471252|0.7684569954872131|5497159.68 rec/s| 12m43s | `--batch_size 8192 --num_embeddings_per_feature "40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36"` |
|MLPerfx2|174.03|16|4096|65536|0.7760998606681824|0.7706344723701477|2028339.2 rec/s| 34m:27s | `--batch_size 4096 --num_embeddings_per_feature "80000000,39060,17295,7424,20265,3,7122,1543,63,80000000,3067956,405282,10,2209,11938,155,4,976,14,80000000,40790948,80000000,590152,12973,108,36"` |
|Full Size|420.94|80|819|65520|0.7841252684593201|0.778694748878479| 459950.4 rec/s | 02h32m03s | `--batch_size 819 --num_embeddings_per_feature "227605432,39060,17295,7424,20265,3,7122,1543,63,130229467,3067956,405282,10,2209,11938,155,4,976,14,292775614,40790948,187188510,590152,12973,108,36"` |

**Reproduce**

Run the following command to reproduce the results for a single node (8 GPUs) on AWS. This command makes use of the `aws_component.py` script.

Ensure to:
- set $PATH_TO_1TB_NUMPY_FILES to the path with the pre-processed .npy files of the Criteo 1TB dataset.
- set $TRAIN_QUEUE to the partition that handles training jobs

```
torchx run --scheduler slurm --scheduler_args partition=$TRAIN_QUEUE,time=5:00:00 aws_component.py:run_dlrm_main --num_trainers=8 -- --pin_memory --batch_size 8192 --epochs 1 --num_embeddings_per_feature "45833188,36746,17245,7413,20243,3,7114,1441,62,29275261,1572176,345138,10,2209,11267,128,4,974,14,48937457,11316796,40094537,452104,12606,104,35" --embedding_dim 128 --dense_arch_layer_sizes "512,256,128" --over_arch_layer_sizes "1024,1024,512,256,1" --in_memory_binary_criteo_path $PATH_TO_1TB_NUMPY_FILES --learning_rate 5.0
```
Upon scheduling the job, there should be an output that looks like this:

```
warnings.warn(
slurm://torchx/14731
torchx 2022-01-07 21:06:59 INFO     Launched app: slurm://torchx/14731
torchx 2022-01-07 21:06:59 INFO     AppStatus:
  msg: ''
  num_restarts: -1
  roles: []
  state: UNKNOWN (7)
  structured_error_msg: <NONE>
  ui_url: null

torchx 2022-01-07 21:06:59 INFO     Job URL: None
```

In this example, the job was launched to: `slurm://torchx/14731`.

Run the following commands to check the status of your job and read the logs:

```
# Status should be "RUNNING" if properly scheduled
torchx status slurm://torchx/14731

# Log file was automatically created in the directory where you launched the job from
cat slurm-14731.out

```

The results from the training can be found in the log file (e.g. `slurm-14731.out`).


**TODO/Work In Progress**
* Confirm DLRM example converges to MLPerf training benchmark threshold of val AUROC 0.8025 in reasonable number of iterations.
* Write section on how to pre-process the dataset.
