# Pipeline Parallelism with Deep Learning Recommendation Models (PipeDLRM)
Deep learning recommendation models are widely used at Facebook. In these models, there are hundreds of sparse features, and there is an embedding table corresponding to each sparse feature (see DLRM (https://ai.facebook.com/blog/dlrm-an-advanced-open-source-deep-learning-recommendation-model/) for details). The rows of an embedding table correspond to different values for the corresponding sparse feature, and the columns correspond to different sparse dimensions. Typically, there are 10s of millions of rows and 64-128 columns for each table. Additionally these model consume petabytes of data, hence resulting in significant time to train. Given the size and complexity of these models, they are almost always trained at scale. Furthermore these models are unlike typical DL models, with low arithmetic intensity and overall compute. So to ensure training throughput it is essential to ensure the different components such as data-reading, sparse feature look-ups don’t become the bottleneck. A promising solution for this is to formulate training as a multi-stage pipeline of these different components including I/O.

# Install

## Install Pytorch

Appy the patches under `pytorch_pathes` and then compile Pytorch from source.
See https://github.com/pytorch/pytorch.

## Install the dependencies
    pip install -r requirements.txt

## Configure the environmental variables
    source env.sh

## Run the PipeDLRM on DAC dataset
    # prepare the dataset according to the tutorial of DLRM.
    cd runtime/recommendation
    ../../exp/pipeline/dlrm_dac_pytorch.sh