# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
#
# This script performs the visualization of the embedding tables created in
# DLRM during the training procedure. We use two popular techniques for
# visualization: umap (https://umap-learn.readthedocs.io/en/latest/) and
# tsne (https://scikit-learn.org/stable/modules/generated/sklearn.manifold.TSNE.html).
# These links also provide instructions on how to install these packages
# in different environments.
#
# Warning: the size of the data to be visualized depends on the RAM on your machine.
#
#
# Connand line examples:
#
# Full analysis of embeddings and data representations for Criteo Kaggle data:
# $python ./tools/visualize.py --data-set=kaggle --load-model=../dlrm-2020-05-25/criteo.pytorch-e-0-i-110591 
#         --raw-data-file=../../criteo/input/train.txt --skip-categorical-analysis 
#         --processed-data-file=../../criteo/input/kaggleAdDisplayChallenge_processed.npz
#
#
# To run just the analysis of categoricala data for Criteo Kaggle data set:
# $python ./tools/visualize.py --data-set=kaggle --load-model=../dlrm-2020-05-25/criteo.pytorch-e-0-i-110591 \
#         --raw-data-file=../../criteo/input/train.txt --data-randomize=none --processed-data-file=../../criteo/input/kaggleAdDisplayChallenge_processed.npz \
#         --skip-embedding --skip-data-plots
#
#
# The following command line arguments are available to the user:
#
#    --load-model                   - DLRM model file
#    --data-set                     - one of ["kaggle", "terabyte"]
#    --max-ind-range                - max index range used during the traning
#    --output-dir                   - output directory, if not specified, it will be traeted from the model and datset names
#    --max-umap-size                - max number of points to visualize using UMAP, default=50000
#    --use-tsne                     - use T-SNE
#    --max-tsne-size                - max number of points to visualize using T-SNE, default=1000)
#    --skip-embedding               - skips analysis of embedding tables
#    --umap-metric                  - metric for UMAP 
#    --skip-data-plots              - skips data plots
#    --skip-categorical-analysis    - skips categorical analysis
# 
#    # data file related
#    --raw-data-file
#    --processed-data-file
#    --data-sub-sample-rate
#    --data-randomize
#    --memory-map
#    --mini-batch-size
#    --num-workers
#    --test-mini-batch-size
#    --test-num-workers
#    --num-batches    
#    --mlperf-logging

import os
import sys
import argparse
import numpy as np
import umap
import hdbscan
import json
import torch
import math
import matplotlib
import matplotlib.pyplot as plt
import collections

from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score

from sklearn import manifold

import dlrm_data_pytorch as dp
from dlrm_s_pytorch import DLRM_Net


def visualize_embeddings_umap(emb_l, 
                              output_dir    = "",
                              max_size      = 500000, 
                              umap_metric   = "euclidean",
                              cat_counts    = None,
                              use_max_count = True):

    for k in range(0, len(emb_l)):

        E = emb_l[k].weight.detach().cpu().numpy()
        print("umap", E.shape)

        # create histogram of norms
        bins = 50
        norms = [np.linalg.norm(E[i], ord=2) for i in range(0,E.shape[0])]
#        plt.hist(norms, bins = bins)
#        plt.title("Cat norm hist var. "+str(k))
        hist, bins = np.histogram(norms, bins=bins)
        logbins = np.logspace(np.log10(bins[0]),np.log10(bins[-1]),len(bins))

        plt.figure(figsize=(8,8))
        plt.title("Categorical norms: " + str(k) + " cardinality " + str(len(cat_counts[k])))
        plt.hist(norms, bins=logbins)
        plt.xscale("log")
#        plt.legend()
        plt.savefig(output_dir+"/cat-norm-histogram-"+str(k)+".png")
        plt.close()

        if E.shape[0] < 20:
            print("Skipping small embedding")
            continue

        n_vis = min(max_size, E.shape[0])
        min_cnt = 0
        
#        reducer = umap.UMAP(random_state=42, n_neighbors=25, min_dist=0.1)
        reducer = umap.UMAP(random_state=42, metric=umap_metric)
        
        if use_max_count is False or n_vis == E.shape[0]:
            Y = reducer.fit_transform(E[:n_vis,:])
        else:
            
            # select values with couns > 1
            done  = False
            min_cnt = 1
            while done == False:
                el_cnt = (cat_counts[k] > min_cnt).sum()
                if el_cnt <= max_size:
                    done = True
                else:
                    min_cnt = min_cnt+1
           
            E1= []
            for i in range(0, E.shape[0]):
                if cat_counts[k][i] > min_cnt:
                    E1.append(E[i,:])
            
            print("max_count_len", len(E1), "mincount", min_cnt)
            Y = reducer.fit_transform(np.array(E1))

            n_vis = len(E1)

        plt.figure(figsize=(8,8))
        
        linewidth = 0
        size      = 1
        
        if Y.shape[0] < 2500:
            linewidth = 1 
            size      = 5

        if cat_counts is None:
            plt.scatter(-Y[:,0], -Y[:,1], s=size, marker=".", linewidth=linewidth)
        else:
            #print(cat_counts[k])
            n_disp = min(len(cat_counts[k]), Y.shape[0])
            cur_max = math.log(max(cat_counts[k]))
            norm_cat_count = [math.log(cat_counts[k][i]+1)/cur_max for i in range(0, len(cat_counts[k]))]
            plt.scatter(-Y[0:n_disp,0], -Y[0:n_disp,1], s=size, marker=".", linewidth=linewidth, c=np.array(norm_cat_count)[0:n_disp], cmap="viridis")
            plt.colorbar()
            
        plt.title("UMAP: categorical var. " + str(k) + "  (" + str(n_vis) + " of " + str(E.shape[0]) + ", min count " + str(min_cnt) + ")")
        plt.savefig(output_dir + "/cat-" + str(k) + "-" + str(n_vis) + "-of-" + str(E.shape[0]) + "-umap.png")
        plt.close()


def visualize_embeddings_tsne(emb_l, 
                              output_dir = "",
                              max_size   = 10000):

    for k in range(0, len(emb_l)):

        E = emb_l[k].weight.detach().cpu()    
        print("tsne", E.shape)

        if E.shape[0] < 20:
            print("Skipping small embedding")
            continue

        n_vis = min(max_size, E.shape[0])
        
        tsne = manifold.TSNE(init="pca", random_state=0, method="exact")
    
        Y = tsne.fit_transform(E[:n_vis,:])

        plt.figure(figsize=(8, 8))

        linewidth = 0
        if Y.shape[0] < 5000:
            linewidth = 1 

        plt.scatter(-Y[:,0], -Y[:,1], s=1, marker=".", linewidth=linewidth)
        
        plt.title("TSNE: categorical var. " + str(k) + "  (" + str(n_vis) + " of " + str(E.shape[0]) + ")")
        plt.savefig(output_dir + "/cat-" + str(k) + "-" + str(n_vis) + "-of-" + str(E.shape[0]) + "-tsne.png")
        plt.close()


def analyse_categorical_data(X_cat, n_days=10, output_dir=""):

    # analyse categorical variables
    n_vec = len(X_cat)
    n_cat = len(X_cat[0])
    n_days = n_days
    
    print("n_vec", n_vec, "n_cat", n_cat)
#    for c in train_data.X_cat:
#        print(n_cat, c)

    all_cat = np.array(X_cat)
    print("all_cat.shape", all_cat.shape)
    day_size = all_cat.shape[0]/n_days

    for i in range(0,n_cat):
        l_d   = []
        l_s1  = []
        l_s2  = []
        l_int = []
        l_rem = []

        cat = all_cat[:,i]
        print("cat", i, cat.shape)
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

            print(d, ",", len(s1), ",", len(s2), ",", len(intersect), ",", (len(s1)-len(intersect)))

        print("spit",    l_d)
        print("before",  l_s1)
        print("after",   l_s2)
        print("inters.", l_int)
        print("removed", l_rem)

        plt.figure(figsize=(8,8))
        plt.plot(l_d, l_s1,  "g", label="before")
        plt.plot(l_d, l_s2,  "r", label="after")
        plt.plot(l_d, l_int, "b", label="intersect")
        plt.plot(l_d, l_rem, "y", label="removed")
        plt.title("categorical var. "+str(i))
        plt.legend()
        plt.savefig(output_dir+"/cat-"+str(i).zfill(3)+".png")
        plt.close()


def analyse_categorical_counts(X_cat, emb_l=None, output_dir=""):

    # analyse categorical variables
    n_vec = len(X_cat)
    n_cat = len(X_cat[0])
    
    print("n_vec", n_vec, "n_cat", n_cat)
#    for c in train_data.X_cat:
#        print(n_cat, c)

    all_cat = np.array(X_cat)
    print("all_cat.shape", all_cat.shape)

    all_counts = []

    for i in range(0,n_cat):
        
        cat = all_cat[:,i]
        if emb_l is None:
            s      = set(cat)
            counts = np.zeros((len(s)))
            print("cat", i, cat.shape, len(s))
        else:
            s = emb_l[i].weight.detach().cpu().shape[0]
            counts = np.zeros((s))
            print("cat", i, cat.shape, s)

        for d in range(0,n_vec):
            cv = int(cat[d])
            counts[cv] = counts[cv]+1

        all_counts.append(counts)

        if emb_l is None:
            plt.figure(figsize=(8,8))
            plt.plot(counts)
            plt.title("Categorical var "+str(i) + " cardinality " + str(len(counts)))
            #        plt.legend()
        else:
            E = emb_l[i].weight.detach().cpu().numpy()
            norms = [np.linalg.norm(E[i], ord=2) for i in range(0,E.shape[0])]

            fig, (ax0, ax1) = plt.subplots(2, 1)
            fig.suptitle("Categorical variable: " + str(i)+" cardinality "+str(len(counts)))

            ax0.plot(counts)
            ax0.set_yscale("log")
            ax0.set_title("Counts", fontsize=10)
    
            ax1.plot(norms)
            ax1.set_title("Norms", fontsize=10)

        plt.savefig(output_dir+"/cat_counts-"+str(i).zfill(3)+".png")
        plt.close()
    
    return all_counts
    

def dlrm_output_wrap(dlrm, X, lS_o, lS_i, T):

    all_feat_vec = []
    all_cat_vec  = []
    x_vec        = None
    t_out        = None
    c_out        = None
    z_out        = []
    p_out        = None

    z_size = len(dlrm.top_l)

    x = dlrm.apply_mlp(X, dlrm.bot_l)
    # debug prints
    #print("intermediate")
    #print(x[0].detach().cpu().numpy())
    x_vec = x[0].detach().cpu().numpy()
    all_feat_vec.append(x_vec)
#    all_X.append(x[0].detach().cpu().numpy())

    # process sparse features(using embeddings), resulting in a list of row vectors
    ly = dlrm.apply_emb(lS_o, lS_i, dlrm.emb_l)

    for e in ly:
        #print(e.detach().cpu().numpy())
        all_feat_vec.append(e[0].detach().cpu().numpy())
        all_cat_vec.append(e[0].detach().cpu().numpy())

    all_feat_vec= np.concatenate(all_feat_vec, axis=0)
    all_cat_vec= np.concatenate(all_cat_vec, axis=0)

#    all_features.append(all_feat_vec)
#    all_cat.append(all_cat_vec)
    t_out = int(T.detach().cpu().numpy()[0,0])
#    all_T.append(int(T.detach().cpu().numpy()[0,0]))

    z = dlrm.interact_features(x, ly)
    # print(z.detach().cpu().numpy())
#    z_out = z.detach().cpu().numpy().flatten()
    z_out.append(z.detach().cpu().numpy().flatten())
#    all_z[0].append(z.detach().cpu().numpy().flatten())

        # obtain probability of a click (using top mlp)
#        print(dlrm.top_l)
#        p = dlrm.apply_mlp(z, dlrm.top_l)

    for i in range(0, z_size):
        z = dlrm.top_l[i](z)

#        if i < z_size-1:
#            curr_z = z.detach().cpu().numpy().flatten()
        z_out.append(z.detach().cpu().numpy().flatten())
#            all_z[i+1].append(curr_z)
#            print("z append", i)
            
#        print("z",i, z.detach().cpu().numpy().flatten().shape)

    p = z

    # clamp output if needed
    if 0.0 < dlrm.loss_threshold and dlrm.loss_threshold < 1.0:
        z = torch.clamp(p, min=dlrm.loss_threshold, max=(1.0 - dlrm.loss_threshold))
    else:
        z = p

    class_thresh = 0.0 #-0.25
    zp = z.detach().cpu().numpy()[0,0]+ class_thresh
    
    p_out = int(zp+0.5)
    if p_out > 1:
        p_out = 1
    if p_out < 0:
        p_out = 0

#    all_pred.append(int(z.detach().cpu().numpy()[0,0]+0.5))

    #print(int(z.detach().cpu().numpy()[0,0]+0.5))
    if int(p_out) == t_out:
        c_out = 0
    else:
        c_out = 1

    return all_feat_vec, x_vec, all_cat_vec, t_out, c_out, z_out, p_out


def create_umap_data(dlrm, data_ld, max_size=50000, offset=0,  info=""):
    
    all_features = []
    all_X        = []
    all_cat      = []
    all_T        = []
    all_c        = []
    all_z        = []
    all_pred     = []
    
    z_size = len(dlrm.top_l)
    print("z_size", z_size)
    for i in range(0, z_size):
        all_z.append([])
    
    for j, (X, lS_o, lS_i, T) in enumerate(data_ld):

        if j < offset:
            continue
        
        if j >= max_size+offset:
            break
        
        af, x, cat, t, c, z, p = dlrm_output_wrap(dlrm, X, lS_o, lS_i, T)
       
        all_features.append(af)
        all_X.append(x)
        all_cat.append(cat)
        all_T.append(t)
        all_c.append(c)
        all_pred.append(p)
        
        for i in range(0, z_size):
            all_z[i].append(z[i])

#    # calculate classifier metrics 
    ac = accuracy_score(all_T, all_pred)
    f1 = f1_score(all_T, all_pred)
    ps = precision_score(all_T, all_pred)
    rc = recall_score(all_T, all_pred)

    print(info, "accuracy", ac, "f1", f1, "precision", ps, "recall", rc)

    return all_features, all_X, all_cat, all_T, all_z, all_c, all_pred


def plot_all_data_3(umap_Y,
                    umap_T,
                    train_Y          = None, 
                    train_T          = None, 
                    test_Y           = None, 
                    test_T           = None, 
                    total_train_size = "", 
                    total_test_size  = "", 
                    info             = "",
                    output_dir       = "",
                    orig_space_dim   = 0):
    
    size = 1
    colors = ["red","green"]

    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig.suptitle("UMAP: " + info + " space dim "+str(orig_space_dim))

    ax0.scatter(umap_Y[:,0], umap_Y[:,1], s=size, c=umap_T, cmap=matplotlib.colors.ListedColormap(colors), marker=".", linewidth=0)
    ax0.set_title("UMAP ("+str(len(umap_T))+" of "+ total_train_size+")", fontsize=7)
    
    if train_Y is not None and train_T is not None:
        ax1.scatter(train_Y[:,0], train_Y[:,1], s=size, c=train_T, cmap=matplotlib.colors.ListedColormap(colors), marker=".", linewidth=0)
        ax1.set_title("Train ("+str(len(train_T))+" of "+ total_train_size+")", fontsize=7)

    if test_Y is not None and test_T is not None:
        ax2.scatter(test_Y[:,0], test_Y[:,1], s=size, c=test_T, cmap=matplotlib.colors.ListedColormap(colors), marker=".", linewidth=0)
        ax2.set_title("Test ("+str(len(test_T))+" of "+ total_test_size+")", fontsize=7)

    plt.savefig(output_dir+"/"+info+"-umap.png")
    plt.close()


def plot_one_class_3(umap_Y,
                     umap_T,
                     train_Y,
                     train_T,
                     test_Y, 
                     test_T, 
                     target           = 0, 
                     col              = "red", 
                     total_train_size = "", 
                     total_test_size  = "", 
                     info             = "",
                     output_dir       = "",
                     orig_space_dim   = 0):
    
    size = 1
    
    fig, (ax0, ax1, ax2) = plt.subplots(1, 3)
    fig.suptitle("UMAP: "+ info + " space dim "+str(orig_space_dim))

    ind_l_umap     = [i for i,x in enumerate(umap_T) if x == target]
    Y_umap_l       = np.array([umap_Y[i,:] for i in ind_l_umap])

    ax0.scatter(Y_umap_l[:,0], Y_umap_l[:,1], s=size, c=col, marker=".", linewidth=0)
    ax0.set_title("UMAP, ("+str(len(umap_T))+" of "+ total_train_size+")", fontsize=7)
    
    if train_Y is not None and train_T is not None:
        ind_l_test = [i for i,x in enumerate(train_T) if x == target]
        Y_test_l   = np.array([train_Y[i,:] for i in ind_l_test])
        
        ax1.scatter(Y_test_l[:,0], Y_test_l[:,1], s=size, c=col, marker=".", linewidth=0)
        ax1.set_title("Train, ("+str(len(train_T))+" of "+ total_train_size+")", fontsize=7)

    if test_Y is not None and test_T is not None:
        ind_l_test = [i for i,x in enumerate(test_T) if x == target]
        Y_test_l   = np.array([test_Y[i,:] for i in ind_l_test])

        ax2.scatter(Y_test_l[:,0], Y_test_l[:,1], s=size, c=col, marker=".", linewidth=0)
        ax2.set_title("Test, ("+str(len(test_T))+" of "+ total_test_size+")", fontsize=7)

    plt.savefig(output_dir+"/"+info+"-umap.png")
    plt.close()


def visualize_umap_data(umap_Y,
                        umap_T,
                        umap_C,
                        umap_P,
                        train_Y, 
                        train_T, 
                        train_C,
                        train_P,
                        test_Y           = None,
                        test_T           = None, 
                        test_C           = None,
                        test_P           = None,
                        total_train_size = "", 
                        total_test_size  = "",  
                        info             = "",
                        output_dir       = "",
                        orig_space_dim   = 0):

    # all classes
    plot_all_data_3(umap_Y           = umap_Y,
                    umap_T           = umap_T,
                    train_Y          = train_Y,
                    train_T          = train_T, 
                    test_Y           = test_Y, 
                    test_T           = test_T, 
                    total_train_size = total_train_size,
                    total_test_size  = total_test_size,
                    info             = info,
                    output_dir       = output_dir,
                    orig_space_dim   = orig_space_dim)

    # all predictions
    plot_all_data_3(umap_Y           = umap_Y,
                    umap_T           = umap_P,
                    train_Y          = train_Y,
                    train_T          = train_P, 
                    test_Y           = test_Y, 
                    test_T           = test_P, 
                    total_train_size = total_train_size,
                    total_test_size  = total_test_size,
                    info             = info+", all-predictions",
                    output_dir       = output_dir,
                    orig_space_dim   = orig_space_dim)

    
    # class 0
    plot_one_class_3(umap_Y           = umap_Y,
                     umap_T           = umap_T,
                     train_Y          = train_Y,
                     train_T          = train_T,
                     test_Y           = test_Y, 
                     test_T           = test_T, 
                     target           = 0, 
                     col              = "red", 
                     total_train_size = total_train_size, 
                     total_test_size  = total_test_size, 
                     info             = info+" class " + str(0),
                     output_dir       = output_dir,
                     orig_space_dim   = orig_space_dim)

    # class 1
    plot_one_class_3(umap_Y           = umap_Y,
                     umap_T           = umap_T,
                     train_Y          = train_Y,
                     train_T          = train_T,
                     test_Y           = test_Y, 
                     test_T           = test_T, 
                     target           = 1, 
                     col              = "green", 
                     total_train_size = total_train_size, 
                     total_test_size  = total_test_size, 
                     info             = info + " class " + str(1),
                     output_dir       = output_dir,
                     orig_space_dim   = orig_space_dim)

    # correct classification
    plot_one_class_3(umap_Y           = umap_Y,
                     umap_T           = umap_C,
                     train_Y          = train_Y,
                     train_T          = train_C,
                     test_Y           = test_Y, 
                     test_T           = test_C, 
                     target           = 0, 
                     col              = "green", 
                     total_train_size = total_train_size, 
                     total_test_size  = total_test_size, 
                     info             = info + " correct ",
                     output_dir       = output_dir,
                     orig_space_dim   = orig_space_dim)

    # errors
    plot_one_class_3(umap_Y           = umap_Y,
                     umap_T           = umap_C,
                     train_Y          = train_Y,
                     train_T          = train_C,
                     test_Y           = test_Y, 
                     test_T           = test_C, 
                     target           = 1, 
                     col              = "red", 
                     total_train_size = total_train_size, 
                     total_test_size  = total_test_size, 
                     info             = info + " errors ",
                     output_dir       = output_dir,
                     orig_space_dim   = orig_space_dim)

    # prediction 0
    plot_one_class_3(umap_Y           = umap_Y,
                     umap_T           = umap_P,
                     train_Y          = train_Y,
                     train_T          = train_P,
                     test_Y           = test_Y, 
                     test_T           = test_P, 
                     target           = 0, 
                     col              = "red", 
                     total_train_size = total_train_size, 
                     total_test_size  = total_test_size, 
                     info             = info + " predict-0 ",
                     output_dir       = output_dir,
                     orig_space_dim   = orig_space_dim)

    # prediction 1
    plot_one_class_3(umap_Y           = umap_Y,
                     umap_T           = umap_P,
                     train_Y          = train_Y,
                     train_T          = train_P,
                     test_Y           = test_Y, 
                     test_T           = test_P, 
                     target           = 1, 
                     col              = "green", 
                     total_train_size = total_train_size, 
                     total_test_size  = total_test_size, 
                     info             = info + " predict-1 ",
                     output_dir       = output_dir,
                     orig_space_dim   = orig_space_dim)

def hdbscan_clustering(umap_data, train_data, test_data, info="", output_dir=""):

    clusterer       = hdbscan.HDBSCAN(min_samples=10, min_cluster_size=500, prediction_data=True)
    umap_labels     = clusterer.fit_predict(umap_data)
    train_labels, _ = hdbscan.approximate_predict(clusterer, train_data)
    test_labels,  _ = hdbscan.approximate_predict(clusterer, test_data)

    fig, ((ax00, ax01, ax02), (ax10, ax11, ax12)) = plt.subplots(2, 3)
    fig.suptitle("HDBSCAN clastering: "+ info )

    # plot umap data
    umap_clustered = (umap_labels >= 0)
    umap_coll = collections.Counter(umap_clustered)
    print("umap_clustered", umap_coll)
#    print("umap_data", umap_data.shape)
#    print("~umap_clustered", umap_clustered.count(False), ~umap_clustered)
    ax00.scatter(umap_data[~umap_clustered, 0],
                 umap_data[~umap_clustered, 1],
                 c=(0.5, 0.5, 0.5),
                 s=0.1,
                 alpha=0.5)
    ax00.set_title("UMAP Outliers " + str(umap_coll[False]), fontsize=7)
    ax10.scatter(umap_data[umap_clustered, 0],
                 umap_data[umap_clustered, 1],
                 c=umap_labels[umap_clustered],
                 s=0.1,
                 cmap="Spectral")
    ax10.set_title("UMAP Inliers " + str(umap_coll[True]), fontsize=7)
    
    # plot train data
    train_clustered = (train_labels >= 0)
    train_coll = collections.Counter(train_clustered)
    ax01.scatter(train_data[~train_clustered, 0],
                 train_data[~train_clustered, 1],
                 c=(0.5, 0.5, 0.5),
                 s=0.1,
                 alpha=0.5)
    ax01.set_title("Train Outliers " + str(train_coll[False]), fontsize=7)
    ax11.scatter(train_data[train_clustered, 0],
                 train_data[train_clustered, 1],
                 c=train_labels[train_clustered],
                 s=0.1,
                 cmap="Spectral")
    ax11.set_title("Train Inliers " + str(train_coll[True]), fontsize=7)
    
    # plot test data
    test_clustered = (test_labels >= 0)
    test_coll = collections.Counter(test_clustered)
    ax02.scatter(test_data[~test_clustered, 0],
                 test_data[~test_clustered, 1],
                 c=(0.5, 0.5, 0.5),
                 s=0.1,
                 alpha=0.5)
    ax02.set_title("Tets Outliers " + str(test_coll[False]), fontsize=7)
    ax12.scatter(test_data[test_clustered, 0],
                 test_data[test_clustered, 1],
                 c=test_labels[test_clustered],
                 s=0.1,
                 cmap="Spectral")
    ax12.set_title("Test Inliers " + str(test_coll[True]), fontsize=7)
    
    plt.savefig(output_dir+"/"+info+"-hdbscan.png")
    plt.close()


def visualize_all_data_umap(dlrm, 
                            train_ld, 
                            test_ld       = None, 
                            max_umap_size = 50000,
                            output_dir    = "",
                            umap_metric   = "euclidean"):

    data_ratio = 1
    
    print("creating umap data")
    umap_train_feat, umap_train_X, umap_train_cat, umap_train_T, umap_train_z, umap_train_c, umap_train_p = create_umap_data(dlrm=dlrm, data_ld=train_ld, max_size=max_umap_size, offset=0, info="umap")
    
    # transform train and test data
    train_feat, train_X, train_cat, train_T, train_z, train_c, train_p = create_umap_data(dlrm=dlrm, data_ld=train_ld, max_size=max_umap_size*data_ratio, offset=max_umap_size, info="train")
    test_feat,  test_X,  test_cat,  test_T,  test_z,  test_c,  test_p  = create_umap_data(dlrm=dlrm, data_ld=test_ld,  max_size=max_umap_size*data_ratio, offset=0,             info="test")

    print("umap_train_feat", np.array(umap_train_feat).shape)
    reducer_all_feat = umap.UMAP(random_state=42, metric=umap_metric)
    umap_feat_Y = reducer_all_feat.fit_transform(umap_train_feat)

    train_feat_Y = reducer_all_feat.transform(train_feat)
    test_feat_Y  = reducer_all_feat.transform(test_feat)
    
    visualize_umap_data(umap_Y           = umap_feat_Y,
                        umap_T           = umap_train_T,
                        umap_C           = umap_train_c,
                        umap_P           = umap_train_p,
                        train_Y          = train_feat_Y, 
                        train_T          = train_T, 
                        train_C          = train_c,
                        train_P          = train_p,
                        test_Y           = test_feat_Y,
                        test_T           = test_T, 
                        test_C           = test_c,
                        test_P           = test_p,
                        total_train_size = str(len(train_ld)), 
                        total_test_size  = str(len(test_ld)), 
                        info             = "all-features",
                        output_dir       = output_dir,
                        orig_space_dim   = np.array(umap_train_feat).shape[1])

    hdbscan_clustering(umap_data  = umap_feat_Y, 
                       train_data = train_feat_Y, 
                       test_data  = test_feat_Y, 
                       info       = "umap-all-features", 
                       output_dir = output_dir)

#    hdbscan_clustering(umap_data  = np.array(umap_train_feat), 
#                       train_data = np.array(train_feat), 
#                       test_data  = np.array(test_feat), 
#                       info       = "all-features", 
#                       output_dir = output_dir)

    print("umap_train_X", np.array(umap_train_X).shape)
    reducer_X = umap.UMAP(random_state=42, metric=umap_metric)
    umap_X_Y = reducer_X.fit_transform(umap_train_X)

    train_X_Y = reducer_X.transform(train_X)
    test_X_Y  = reducer_X.transform(test_X)

    visualize_umap_data(umap_Y           = umap_X_Y,
                        umap_T           = umap_train_T,
                        umap_C           = umap_train_c,
                        umap_P           = umap_train_p,
                        train_Y          = train_X_Y, 
                        train_T          = train_T, 
                        train_C          = train_c,
                        train_P          = train_p,
                        test_Y           = test_X_Y,
                        test_T           = test_T, 
                        test_C           = test_c,
                        test_P           = test_p,
                        total_train_size = str(len(train_ld)), 
                        total_test_size  = str(len(test_ld)), 
                        info             = "cont-features",
                        output_dir       = output_dir,
                        orig_space_dim   = np.array(umap_train_X).shape[1])

    print("umap_train_cat", np.array(umap_train_cat).shape)
    reducer_cat = umap.UMAP(random_state=42, metric=umap_metric)
    umap_cat_Y = reducer_cat.fit_transform(umap_train_cat)

    train_cat_Y = reducer_cat.transform(train_cat)
    test_cat_Y  = reducer_cat.transform(test_cat)

    visualize_umap_data(umap_Y           = umap_cat_Y,
                        umap_T           = umap_train_T,
                        umap_C           = umap_train_c,
                        umap_P           = umap_train_p,
                        train_Y          = train_cat_Y, 
                        train_T          = train_T, 
                        train_C          = train_c,
                        train_P          = train_p,
                        test_Y           = test_cat_Y,
                        test_T           = test_T, 
                        test_C           = test_c,
                        test_P           = test_p,
                        total_train_size = str(len(train_ld)), 
                        total_test_size  = str(len(test_ld)), 
                        info             = "cat-features",
                        output_dir       = output_dir,
                        orig_space_dim   = np.array(umap_train_cat).shape[1])

    # UMAP for z data
    for i in range(0,len(umap_train_z)):
        print("z", i, np.array(umap_train_z[i]).shape)
        reducer_z = umap.UMAP(random_state=42, metric=umap_metric)
        umap_z_Y = reducer_z.fit_transform(umap_train_z[i])

        train_z_Y = reducer_z.transform(train_z[i])
        test_z_Y  = reducer_z.transform(test_z[i])

        visualize_umap_data(umap_Y           = umap_z_Y,
                            umap_T           = umap_train_T,
                            umap_C           = umap_train_c,
                            umap_P           = umap_train_p,
                            train_Y          = train_z_Y, 
                            train_T          = train_T, 
                            train_C          = train_c,
                            train_P          = train_p,
                            test_Y           = test_z_Y,
                            test_T           = test_T, 
                            test_C           = test_c,
                            test_P           = test_p,
                            total_train_size = str(len(train_ld)), 
                            total_test_size  = str(len(test_ld)), 
                            info             = "z-features-"+str(i),
                            output_dir       = output_dir,
                            orig_space_dim   = np.array(umap_train_z[i]).shape[1])


def analyze_model_data(output_dir,
                       dlrm,
                       train_ld,
                       test_ld,
                       train_data,
                       skip_embedding            = False,
                       use_tsne                  = False,
                       max_umap_size             = 50000,
                       max_tsne_size             = 10000,
                       skip_categorical_analysis = False,
                       skip_data_plots           = False,
                       umap_metric               = "euclidean"):

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    if skip_embedding is False:

        cat_counts = None
        
        cat_counts = analyse_categorical_counts(X_cat=train_data.X_cat, emb_l=dlrm.emb_l, output_dir=output_dir)

        visualize_embeddings_umap(emb_l       = dlrm.emb_l,
                                  output_dir  = output_dir,
                                  max_size    = max_umap_size,
                                  umap_metric = umap_metric,
                                  cat_counts  = cat_counts)

        if use_tsne is True:
            visualize_embeddings_tsne(emb_l      = dlrm.emb_l,
                                      output_dir = output_dir,
                                      max_size   = max_tsne_size)

    # data visualization and analysis
    if skip_data_plots is False:
        visualize_all_data_umap(dlrm=dlrm, train_ld=train_ld, test_ld=test_ld, max_umap_size=max_umap_size, output_dir=output_dir, umap_metric=umap_metric)

    # analyse categorical variables
    if skip_categorical_analysis is False and args.data_randomize == "none":
        analyse_categorical_data(X_cat=train_data.X_cat, n_days=10, output_dir=output_dir)



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
#    parser.add_argument("--mlperf-bin-loader", action="store_true", default=False)
    parser.add_argument("--output-dir", type=str, default="")
    parser.add_argument("--skip-embedding", action="store_true", default=False)
    parser.add_argument("--umap-metric", type=str, default="euclidean")
    parser.add_argument("--skip-data-plots", action="store_true", default=False)
    parser.add_argument("--skip-categorical-analysis", action="store_true", default=False)
    
    # umap relatet
    parser.add_argument("--max-umap-size", type=int, default=50000)
    # tsne related
    parser.add_argument("--use-tsne", action="store_true", default=False)
    parser.add_argument("--max-tsne-size", type=int, default=1000)
    # data file related
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    parser.add_argument("--data-sub-sample-rate", type=float, default=0.0)  # in [0, 1]
    parser.add_argument("--data-randomize", type=str, default="total")  # none, total or day or none
    parser.add_argument("--memory-map", action="store_true", default=False)
    parser.add_argument("--mini-batch-size", type=int, default=1)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--test-mini-batch-size", type=int, default=1)
    parser.add_argument("--test-num-workers", type=int, default=0)
    parser.add_argument("--num-batches", type=int, default=0)    
    # mlperf logging (disables other output and stops early)
    parser.add_argument("--mlperf-logging", action="store_true", default=False)

    args = parser.parse_args()

    print("command line args: ", json.dumps(vars(args)))

    if output_dir == "":
        output_dir = args.data_set+"-"+os.path.split(args.load_model)[-1]+"-vis_all"
    print("output_dir:", output_dir)
    
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

    # check input parameters
    if args.data_randomize != "none" and args.skip_categorical_analysis is not True:
        print("Incorrect option for categoricat analysis, use:  --data-randomize=none")
        sys.exit(-1)

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

        ld_model = torch.load(args.load_model, map_location=torch.device("cpu"))
        dlrm.load_state_dict(ld_model["state_dict"])

        print("Model loaded", args.load_model)
        #print(dlrm)

    z_size = len(dlrm.top_l)
    for i in range(0, z_size):
         print("z", i, dlrm.top_l[i])

    # load data
    train_data = None
    test_data  = None
    
    if args.raw_data_file is not "" or args.processed_data_file is not "":
        train_data, train_ld, test_data, test_ld = dp.make_criteo_data_and_loaders(args)

    analyze_model_data(output_dir                = output_dir,
                       dlrm                      = dlrm,
                       train_ld                  = train_ld,
                       test_ld                   = test_ld,
                       train_data                = train_data,
                       skip_embedding            = args.skip_embedding,
                       use_tsne                  = args.use_tsne,
                       max_umap_size             = args.max_umap_size,
                       max_tsne_size             = args.max_tsne_size,
                       skip_categorical_analysis = args.skip_categorical_analysis,
                       skip_data_plots           = args.skip_data_plots,
                       umap_metric               = args.umap_metric)

