# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#
# This script performs the visualization of the embedding tables created in 
# DLRM during the training procedure. We use two popular techniques for 
# visualization: umap (https://umap-learn.readthedocs.io/en/latest/) 
# and tsne (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html). 
# These links also provide instructions on how to install these packages 
# in different environments.
#
# Warning: the size of the data to be visualized depends on the RAM on your machine.
#
#
# A sample run of the code, with a kaggle model is shown below 
# $ python ./tools/visualize.py –dataset=kaggle --load-model=./input/dlrm_kaggle.pytorch 
#
#
# The following command line arguments are available to the user:
#
#    --load-model      - DLRM model file
#    --dataset         - one of ["kaggle", "terabyte"]
#    --max-ind-range   - max index range used during the traning 
#    --output-dir      - output directory where output plots will be written, default will be on of these: ["kaggle_vis", "terabyte_vis"] 
#    --max-umap-size   - max number of points to visualize using UMAP, default=50000
#    --use-tsne        - use T-SNE
#    --max-tsne-size   - max number of points to visualize using T-SNE, default=1000)    
#

import sys, os
import argparse
import numpy as np
import umap
import json
import torch
import matplotlib.pyplot as plt

from sklearn import manifold

import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net


def visualize_embeddings_umap(emb_l, 
                              output_dir    = "",
                              max_size = 500000):

    for k in range(0, len(emb_l)):

        E = dlrm.emb_l[k].weight.detach().cpu()    
        print("umap", E.shape)

        if E.shape[0] < 20:
            print("Skipping small embedding")
            continue
        
#        reducer = umap.UMAP(random_state=42, n_neighbors=25, min_dist=0.1)
        reducer = umap.UMAP(random_state=42)
        Y = reducer.fit_transform(E[:max_size,:])

        plt.figure(figsize=(8,8))
        if Y.shape[0] > 2000:
            size = 1 
        else:
            size = 5
        plt.scatter(-Y[:,0], -Y[:,1], s=size)

        n_vis = min(max_size, E.shape[0])
        plt.title("UMAP: categorical var. "+str(k)+"  ("+str(n_vis)+" of "+str(E.shape[0])+")")
        plt.savefig(output_dir+"/cat-"+str(k)+"-"+str(n_vis)+"-of-"+str(E.shape[0])+"-umap.png")
        plt.close()

def visualize_embeddings_tsne(emb_l, 
                              output_dir = "",
                              max_size   = 10000):

    for k in range(0, len(emb_l)):

        E = dlrm.emb_l[k].weight.detach().cpu()    
        print("tsne", E.shape)

        if E.shape[0] < 20:
            print("Skipping small embedding")
            continue
        
        tsne = manifold.TSNE(init='pca', random_state=0, method='exact')
    
        Y = tsne.fit_transform(E[:max_size,:])
    
        plt.figure(figsize=(8,8))
        plt.scatter(-Y[:,0], -Y[:,1])
        
        n_vis = min(max_size, E.shape[0])
        plt.title("TSNE: categorical var. "+str(k)+"  ("+str(n_vis)+" of "+str(E.shape[0])+")")
        plt.savefig(output_dir+"/cat-"+str(k)+"-"+str(n_vis)+"-of-"+str(E.shape[0])+"-tsne.png")
        plt.close()


if __name__ == "__main__":

    output_dir = ""
    
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Exploratory DLRM analysis"
    )

    parser.add_argument("--load-model", type=str, default="")
    parser.add_argument("--dataset", choices=["kaggle", "terabyte"], help="dataset")
#    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--max-ind-range", type=int, default=-1)
#    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--output-dir", type=str, default="")
    # umap related
    parser.add_argument("--max-umap-size", type=int, default=50000)
    # tsne related
    parser.add_argument("--use-tsne", action='store_true', default=False)
    parser.add_argument("--max-tsne-size", type=int, default=1000)

    args = parser.parse_args()

    print('command line args: ', json.dumps(vars(args)))

    if output_dir == "":
        output_dir = args.dataset+"_vis"
    print('output_dir:', output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.dataset == "kaggle":
        # 1. Criteo Kaggle Display Advertisement Challenge Dataset (see ./bench/dlrm_s_criteo_kaggle.sh)
        m_spa=16
        ln_emb=np.array([1460,583,10131227,2202608,305,24,12517,633,3,93145,5683,8351593,3194,27,14992,5461306,10,5652,2173,4,7046547,18,15,286181,105,142572])
        ln_bot=np.array([13,512,256,64,16])
        ln_top=np.array([367,512,256,1])
        
    elif args.dataset == "terabyte":

        if args.max_ind_range == 10000000:
            # 2. Criteo Terabyte (see ./bench/dlrm_s_criteo_terabyte.sh [--sub-sample=0.875] --max-in-range=10000000)
            m_spa=64
            ln_emb=np.array([9980333,36084,17217,7378,20134,3,7112,1442,61, 9758201,1333352,313829,10,2208,11156,122,4,970,14, 9994222, 7267859, 9946608,415421,12420,101, 36])
            ln_bot=np.array([13,512,256,64])
            ln_top=np.array([415,512,512,256,1])
        elif args.max_ind_range == 40000000:
            # 3. Criteo Terabyte MLPerf training (see ./bench/run_and_time.sh --max-in-range=40000000)
            m_spa=128
            ln_emb=np.array([39884406,39043,17289,7420,20263,3,7120,1543,63,38532951,2953546,403346,10,2208,11938,155,4,976,14,39979771,25641295,39664984,585935,12972,108,36])
            ln_bot=np.array([13,512,256,128])
            ln_top=np.array([479,1024,1024,512,256,1])
        else:
            raise ValueError("only --max-in-range 10M or 40M is supported")
    else:
        raise ValueError("only kaggle|terabyte dataset options are supported")

    dlrm = DLRM_Net(
            m_spa,
            ln_emb,
            ln_bot,
            ln_top,
            arch_interaction_op="dot",
            arch_interaction_itself=False,
            sigmoid_bot=-1,
            sigmoid_top=ln_top.size - 2,
            sync_dense_params=True,
            loss_threshold=0.0,
            ndevices=-1,
            qr_flag=False,
            qr_operation=None,
            qr_collisions=None,
            qr_threshold=None,
            md_flag=False,
            md_threshold=None,
        )

    # Load model is specified
    if not (args.load_model == ""):
        print("Loading saved model {}".format(args.load_model))

        ld_model = torch.load(args.load_model, map_location=torch.device('cpu'))
        dlrm.load_state_dict(ld_model["state_dict"])

        print("Model loaded", args.load_model)
        #print(dlrm)
    
    visualize_embeddings_umap(emb_l      = dlrm.emb_l,
                              output_dir = output_dir,
                              max_size   = args.max_umap_size)

    if args.use_tsne == True:
        visualize_embeddings_tsne(emb_l      = dlrm.emb_l,
                                  output_dir = output_dir,
                                  max_size   = args.max_tsne_size)

