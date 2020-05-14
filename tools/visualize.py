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
#    --data-set        - one of ["kaggle", "terabyte"]
#    --max-ind-range   - max index range used during the traning 
#    --output-dir      - output directory where output plots will be written, default will be on of these: ["kaggle_vis", "terabyte_vis"] 
#    --max-umap-size   - max number of points to visualize using UMAP, default=50000
#    --use-tsne        - use T-SNE
#    --max-tsne-size   - max number of points to visualize using T-SNE, default=1000)    
#

import os
import argparse
import numpy as np
import umap
import json
import torch
import random
import matplotlib
import matplotlib.pyplot as plt

from sklearn.metrics import accuracy_score
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

        n_vis = min(max_size, E.shape[0])

#        reducer = umap.UMAP(random_state=42, n_neighbors=25, min_dist=0.1)
        reducer = umap.UMAP(random_state=42)
        Y = reducer.fit_transform(E[:n_vis,:])

        plt.figure(figsize=(8,8))
        if Y.shape[0] > 2000:
            size = 1 
        else:
            size = 5
        plt.scatter(-Y[:,0], -Y[:,1], s=size)

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

        n_vis = min(max_size, E.shape[0])
        
        tsne = manifold.TSNE(init='pca', random_state=0, method='exact')
    
        Y = tsne.fit_transform(E[:n_vis,:])
    
        plt.figure(figsize=(8,8))
        plt.scatter(-Y[:,0], -Y[:,1])
        
        plt.title("TSNE: categorical var. "+str(k)+"  ("+str(n_vis)+" of "+str(E.shape[0])+")")
        plt.savefig(output_dir+"/cat-"+str(k)+"-"+str(n_vis)+"-of-"+str(E.shape[0])+"-tsne.png")
        plt.close()


def visualize_data_umap(data, data_ld, info=""):

    all_features = []

    for j, (X, lS_o, lS_i, T) in enumerate(data_ld):

        if j >= args.max_umap_size:
            break
            
#        print(X)
#        print(lS_o)
#        print(lS_i)
#        print(T)
        features = []

        x = dlrm.apply_mlp(X, dlrm.bot_l)
        # debug prints
        #print("intermediate")
        #print(x[0].detach().cpu().numpy())
        features.append(x[0].detach().cpu().numpy())

        # process sparse features(using embeddings), resulting in a list of row vectors
        ly = dlrm.apply_emb(lS_o, lS_i, dlrm.emb_l)

        for e in ly:
            #print(e.detach().cpu().numpy())
            features.append(e[0].detach().cpu().numpy())

        features= np.concatenate(features, axis=0)
        #print('features')
        #print(features)
        all_features.append(features)


#    reducer = umap.UMAP(random_state=42, n_neighbors=25, min_dist=0.1)
    reducer = umap.UMAP(random_state=42)
    Y = reducer.fit_transform(all_features)

    plt.figure(figsize=(8,8))
    if Y.shape[0] > 2000:
        size = 1 
    else:
        size = 5

    colors = ['red','green']
    plt.scatter(-Y[:,0], -Y[:,1], s=size, c=data.y[:len(all_features)], cmap=matplotlib.colors.ListedColormap(colors))

    plt.title("UMAP: "+info+" data. "+"("+str(len(all_features))+" of "+str(len(data))+")")
    plt.savefig(output_dir+"/"+info+"-data-"+str(len(all_features))+"-of-"+str(len(data))+"-umap.png")
    plt.close()


def analyse_categorical_data(X_cat, n_days=10, output_dir=""):

    # analyse categorical variables
    n_vec = len(X_cat)
    n_cat = len(X_cat[0])
    n_days = n_days
    
    print('n_vec', n_vec, 'n_cat', n_cat)
#    for c in train_data.X_cat:
#        print(n_cat, c)

    all_cat = np.array(X_cat)
    print('all_cat.shape', all_cat.shape)
    day_size = all_cat.shape[0]/n_days

    for i in range(0,n_cat):
        l_d   = []
        l_s1  = []
        l_s2  = []
        l_int = []
        l_rem = []

        cat = all_cat[:,i]
        print('cat', i, cat.shape)
        for d in range(1,n_days):
            offset = int(d*day_size)
            #print(offset)
            cat1 = cat[:offset]
            cat2 = cat[offset:]

            s1 = set(cat1)
            s2 = set(cat2)

            intersect = list(s1 & s2) 
            #print(intersect)
            l_d.append(d)
            l_s1.append(len(s1))
            l_s2.append(len(s2))
            l_int.append(len(intersect))
            l_rem.append((len(s1)-len(intersect)))

            print(d, ',', len(s1), ',', len(s2), ',', len(intersect), ',', (len(s1)-len(intersect)))

        print("spit",    l_d)
        print("before",  l_s1)
        print("after",   l_s2)
        print("inters.", l_int)
        print("removed", l_rem)

        plt.figure(figsize=(8,8))
        plt.plot(l_d, l_s1,  'g', label='before')
        plt.plot(l_d, l_s2,  'r', label='after')
        plt.plot(l_d, l_int, 'b', label='intersect')
        plt.plot(l_d, l_rem, 'y', label='removed')
        plt.title("categorical var. "+str(i))
        plt.legend()
        plt.savefig(output_dir+"/cat-"+str(i).zfill(3)+".png")
        plt.close()



if __name__ == "__main__":

    output_dir = ""
    
    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Exploratory DLRM analysis"
    )

    parser.add_argument("--load-model", type=str, default="")
    parser.add_argument("--data-set", choices=["kaggle", "terabyte"], help="dataset")
#    parser.add_argument("--dataset-path", required=True, help="path to the dataset")
    parser.add_argument("--max-ind-range", type=int, default=-1)
#    parser.add_argument("--mlperf-bin-loader", action='store_true', default=False)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--skip-embedding", action='store_true', default=False)
    parser.add_argument("--skip-data-plots", action='store_true', default=False)
    
    # umap relatet
    parser.add_argument("--max-umap-size", type=int, default=50000)
    # tsne related
    parser.add_argument("--use-tsne", action='store_true', default=False)
    parser.add_argument("--max-tsne-size", type=int, default=1000)
    # data file related
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--data-randomize", type=str, default="none")  # total or day or none
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-mini-batch-size", type=int, default=1)
    parser.add_argument("--test-num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=0)    
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)

    args = parser.parse_args()

    print('command line args: ', json.dumps(vars(args)))

    if output_dir == "":
        output_dir = args.data_set+"_vis"
    print('output_dir:', output_dir)
    
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if args.data_set == "kaggle":
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


    if args.skip_embedding == False:
        visualize_embeddings_umap(emb_l      = dlrm.emb_l,
                                  output_dir = output_dir,
                                  max_size   = args.max_umap_size)

        if args.use_tsne == True:
            visualize_embeddings_tsne(emb_l      = dlrm.emb_l,
                                      output_dir = output_dir,
                                      max_size   = args.max_tsne_size)

    # data visualization and analysis
    if args.raw_data_file is not "" or args.processed_data_file is not "":

        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)

#        print(train_data.y)
#        print(train_data.X_int)
#        print(train_data.X_cat)
#        print(train_data[0]

        if args.skip_data_plots == False:
            visualize_data_umap(data=train_data, data_ld=train_ld, info="train")
            visualize_data_umap(data=test_data, data_ld=test_ld, info="test")


        # analyse categorical variables
        analyse_categorical_data(X_cat=train_data.X_cat, n_days=10, output_dir=output_dir)




'''
        # chance based classification
        # class imbalance
        tr_clk_prob = sum(train_data.y)/len(train_data.y)
        print('Targets', sum(train_data.y), len(train_data.y), tr_clk_prob)

        sim_chance = []
        for i in range(0,len(train_data.y)):
#            if random.uniform(0,1) <= (1.0-tr_clk_prob):
            if random.uniform(0,1) <= 1.0:
                sim_chance.append(0)
            else:
                sim_chance.append(1)
                
        #print(sim_chance)
        print('Sim', sum(sim_chance), len(sim_chance), sum(sim_chance)/len(sim_chance))
        print(accuracy_score(train_data.y, sim_chance))
'''



