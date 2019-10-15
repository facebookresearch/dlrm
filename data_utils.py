# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the DLRM benchmark
#
# Utility function(s) to download and pre-process public data sets
#   - Criteo Kaggle Display Advertising Challenge Dataset
#   (https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/)
#   - Criteo Terabyte Dataset
#   (https://labs.criteo.com/2013/12/download-terabyte-click-logs/)
#
# After downloading dataset, run:
#   getCriteoAdData(datafile="<path-to-train.txt>",
#                   o_filename=kaggleAdDisplayChallenge_processed.npz,
#                   max_ind_range=-1,
#                   days=7,
#                   criteo_kaggle=True")
#   getCriteoAdData(datafile="<path-to-day_{0,...,23}>",
#                   o_filename=terabyte_processed.npz,
#                   max_ind_range=-1,
#                   days=24,
#                   criteo_kaggle=False")

from __future__ import absolute_import, division, print_function, unicode_literals

import sys
# import os
from os import path
# import io
# from io import StringIO
# import collections as coll

import numpy as np
import torch


def convertUStringToDistinctIntsDict(mat, convertDicts, counts):
    # Converts matrix of unicode strings into distinct integers.
    #
    # Inputs:
    #     mat (np.array): array of unicode strings to convert
    #     convertDicts (list): dictionary for each column
    #     counts (list): number of different categories in each column
    #
    # Outputs:
    #     out (np.array): array of output integers
    #     convertDicts (list): dictionary for each column
    #     counts (list): number of different categories in each column

    # check if convertDicts and counts match correct length of mat
    if len(convertDicts) != mat.shape[1] or len(counts) != mat.shape[1]:
        print("Length of convertDicts or counts does not match input shape")
        print("Generating convertDicts and counts...")

        convertDicts = [{} for _ in range(mat.shape[1])]
        counts = [0 for _ in range(mat.shape[1])]

    # initialize output
    out = np.zeros(mat.shape)

    for j in range(mat.shape[1]):
        for i in range(mat.shape[0]):
            # add to convertDict and increment count
            if mat[i, j] not in convertDicts[j]:
                convertDicts[j][mat[i, j]] = counts[j]
                counts[j] += 1
            out[i, j] = convertDicts[j][mat[i, j]]

    return out, convertDicts, counts


def convertUStringToDistinctIntsUnique(mat, mat_uni, counts):
    # mat is an array of 0,...,# samples, with each being 26 categorical features

    # check if mat_unique and counts match correct length of mat
    if len(mat_uni) != mat.shape[1] or len(counts) != mat.shape[1]:
        print("Length of mat_unique or counts does not match input shape")
        print("Generating mat_unique and counts...")

        mat_uni = [np.array([]) for _ in range(mat.shape[1])]
        counts = [0 for _ in range(mat.shape[1])]

    # initialize output
    out = np.zeros(mat.shape)
    ind_map = [np.array([]) for _ in range(mat.shape[1])]

    # find out and assign unique ids to features
    for j in range(mat.shape[1]):
        m = mat_uni[j].size
        mat_concat = np.concatenate((mat_uni[j], mat[:, j]))
        mat_uni[j], ind_map[j] = np.unique(mat_concat, return_inverse=True)
        out[:, j] = ind_map[j][m:]
        counts[j] = mat_uni[j].size

    return out, mat_uni, counts


def processCriteoAdData(d_path, npzfile, split, convertDicts, pre_comp_counts):
    # Process Kaggle Display Advertising Challenge or Terabyte Dataset
    # by converting unicode strings in X_cat to integers and
    # converting negative integer values in X_int.
    #
    # Loads data in the form "{kaggle|terabyte}_day_i.npz" where i is the day.
    #
    # Inputs:
    #   d_path (str): path for {kaggle|terabyte}_day_i.npz files
    #   split (int): total number of splits in the dataset (typically 7 or 24)

    # process data if not all files exist
    for i in range(split):
        filename_i = str(d_path) + npzfile + "_{0}_processed.npz".format(i)

        if path.exists(filename_i):
            print("Using existing " + filename_i, end="\r")
        else:
            with np.load(d_path + npzfile + "_{0}.npz".format(i)) as data:
                # categorical features
                '''
                # Approach 1a: using empty dictionaries
                X_cat, convertDicts, counts = convertUStringToDistinctIntsDict(
                    data["X_cat"], convertDicts, counts
                )
                '''
                '''
                # Approach 1b: using empty np.unique
                X_cat, convertDicts, counts = convertUStringToDistinctIntsUnique(
                    data["X_cat"], convertDicts, counts
                )
                '''
                # Approach 2a: using pre-computed dictionaries
                X_cat_t = np.zeros(data["X_cat_t"].shape)
                for j in range(26):
                    for i, x in enumerate(data["X_cat_t"][j, :]):
                        X_cat_t[j, i] = convertDicts[j][x]
                '''
                # Approach 2b: using pre-computed dictionaries (unrolled)
                X_cat = np.zeros(data["X_cat"].shape)
                print("\nshape  " + str(data["X_cat"].shape), end="\n")
                for i in range(data["X_cat"].shape[0]):
                    print(i, end="\n")
                    X_cat[i,0] = convertDicts[0][data["X_cat"][i,0]]
                    X_cat[i,1] = convertDicts[1][data["X_cat"][i,1]]
                    X_cat[i,2] = convertDicts[2][data["X_cat"][i,2]]
                    X_cat[i,3] = convertDicts[3][data["X_cat"][i,3]]
                    X_cat[i,4] = convertDicts[4][data["X_cat"][i,4]]
                    X_cat[i,5] = convertDicts[5][data["X_cat"][i,5]]
                    X_cat[i,6] = convertDicts[6][data["X_cat"][i,6]]
                    X_cat[i,7] = convertDicts[7][data["X_cat"][i,7]]
                    X_cat[i,8] = convertDicts[8][data["X_cat"][i,8]]
                    X_cat[i,9] = convertDicts[9][data["X_cat"][i,9]]
                    X_cat[i,10] = convertDicts[10][data["X_cat"][i,10]]
                    X_cat[i,11] = convertDicts[11][data["X_cat"][i,11]]
                    X_cat[i,12] = convertDicts[12][data["X_cat"][i,12]]
                    X_cat[i,13] = convertDicts[13][data["X_cat"][i,13]]
                    X_cat[i,14] = convertDicts[14][data["X_cat"][i,14]]
                    X_cat[i,15] = convertDicts[15][data["X_cat"][i,15]]
                    X_cat[i,16] = convertDicts[16][data["X_cat"][i,16]]
                    X_cat[i,17] = convertDicts[17][data["X_cat"][i,17]]
                    X_cat[i,18] = convertDicts[18][data["X_cat"][i,18]]
                    X_cat[i,19] = convertDicts[19][data["X_cat"][i,19]]
                    X_cat[i,20] = convertDicts[20][data["X_cat"][i,20]]
                    X_cat[i,21] = convertDicts[21][data["X_cat"][i,21]]
                    X_cat[i,22] = convertDicts[22][data["X_cat"][i,22]]
                    X_cat[i,23] = convertDicts[23][data["X_cat"][i,23]]
                    X_cat[i,24] = convertDicts[24][data["X_cat"][i,24]]
                    X_cat[i,25] = convertDicts[25][data["X_cat"][i,25]]
                '''
                # continuous features
                X_int = data["X_int"]
                X_int[X_int < 0] = 0
                # targets
                y = data["y"]

            np.savez_compressed(
                filename_i,
                # X_cat = X_cat,
                X_cat=np.transpose(X_cat_t),  # transpose of the data
                X_int=X_int,
                y=y,
            )
            print("Processed " + filename_i, end="\r")

    # sanity check (applicable only if counts have been pre-computed & are re-computed)
    # for j in range(26):
    #    if pre_comp_counts[j] != counts[j]:
    #        sys.exit("ERROR: Sanity check on counts has failed")
    # print("\nSanity check on counts passed")

    return


def concatCriteoAdData(d_path, npzfile, split, o_filename):
    # Concatenates different splits/days and saves the result.
    #
    # Inputs:
    #   split (int): total number of splits in the dataset (typically 7 or 24)
    #   d_path (str): path for {kaggle|terabyte}_day_i.npz files
    #   o_filename (str): output file name
    #
    # Output:
    #   o_file (str): output file path

    print("Concatenating multiple days into %s.npz file" % str(d_path + o_filename))

    # load and concatenate data
    for i in range(split):
        filename_i = d_path + npzfile + "_{0}_processed.npz".format(i)
        with np.load(filename_i) as data:
            if i == 0:
                X_cat = data["X_cat"]
                X_int = data["X_int"]
                y = data["y"]

            else:
                X_cat = np.concatenate((X_cat, data["X_cat"]))
                X_int = np.concatenate((X_int, data["X_int"]))
                y = np.concatenate((y, data["y"]))
        print("Loaded day:", i, "y = 1:", len(y[y == 1]), "y = 0:", len(y[y == 0]))

    with np.load(d_path + npzfile + "_counts.npz") as data:
        counts = data["counts"]
    print("Loaded counts!")

    np.savez_compressed(
        d_path + o_filename + ".npz",
        X_cat=X_cat,
        X_int=X_int,
        y=y,
        counts=counts,
    )

    return d_path + o_filename + ".npz"


def transformCriteoAdData(X_cat, X_int, y, days, split, randomize, cuda):
    # Transforms Criteo Kaggle or terabyte data by applying log transformation
    # on dense features and converting everything to appropriate tensors.
    #
    # Inputs:
    #     X_cat (ndarray): array of integers corresponding to preprocessed
    #                      categorical features
    #     X_int (ndarray): array of integers corresponding to dense features
    #     y (ndarray): array of bool corresponding to labels
    #     split (bool): flag for splitting dataset into training/validation/test
    #                     sets
    #     randomize (str): determines randomization scheme
    #         "none": no randomization
    #         "day": randomizes each day"s data (only works if split = True)
    #         "total": randomizes total dataset
    #     cuda (bool): flag for enabling CUDA and transferring data to GPU
    #
    # Outputs:
    #     if split:
    #         X_cat_train (tensor): sparse features for training set
    #         X_int_train (tensor): dense features for training set
    #         y_train (tensor): labels for training set
    #         X_cat_val (tensor): sparse features for validation set
    #         X_int_val (tensor): dense features for validation set
    #         y_val (tensor): labels for validation set
    #         X_cat_test (tensor): sparse features for test set
    #         X_int_test (tensor): dense features for test set
    #         y_test (tensor): labels for test set
    #     else:
    #         X_cat (tensor): sparse features
    #         X_int (tensor): dense features
    #         y (tensor): label

    # define initial set of indices
    indices = np.arange(len(y))

    # split dataset
    if split:
        indices = np.array_split(indices, days)

        # randomize each day"s dataset
        if randomize == "day" or randomize == "total":
            for i in range(len(indices)):
                indices[i] = np.random.permutation(indices[i])

        train_indices = np.concatenate(indices[:-1])
        test_indices = indices[-1]
        val_indices, test_indices = np.array_split(test_indices, 2)

        print("Defined training and testing indices...")

        # randomize all data in training set
        if randomize == "total":
            train_indices = np.random.permutation(train_indices)
            print("Randomized indices...")

        # create training, validation, and test sets
        X_cat_train = X_cat[train_indices]
        X_int_train = X_int[train_indices]
        y_train = y[train_indices]

        X_cat_val = X_cat[val_indices]
        X_int_val = X_int[val_indices]
        y_val = y[val_indices]

        X_cat_test = X_cat[test_indices]
        X_int_test = X_int[test_indices]
        y_test = y[test_indices]

        print("Split data according to indices...")

        # convert to tensors
        if cuda:
            X_cat_train = torch.tensor(X_cat_train, dtype=torch.long).pin_memory()
            X_int_train = torch.log(
                torch.tensor(X_int_train, dtype=torch.float) + 1
            ).pin_memory()
            y_train = torch.tensor(y_train.astype(np.float32)).pin_memory()

            X_cat_val = torch.tensor(X_cat_val, dtype=torch.long).pin_memory()
            X_int_val = torch.log(
                torch.tensor(X_int_val, dtype=torch.float) + 1
            ).pin_memory()
            y_val = torch.tensor(y_val.astype(np.float32)).pin_memory()

            X_cat_test = torch.tensor(X_cat_test, dtype=torch.long).pin_memory()
            X_int_test = torch.log(
                torch.tensor(X_int_test, dtype=torch.float) + 1
            ).pin_memory()
            y_test = torch.tensor(y_test.astype(np.float32)).pin_memory()
        else:
            X_cat_train = torch.tensor(X_cat_train, dtype=torch.long)
            X_int_train = torch.log(torch.tensor(X_int_train, dtype=torch.float) + 1)
            y_train = torch.tensor(y_train.astype(np.float32))

            X_cat_val = torch.tensor(X_cat_val, dtype=torch.long)
            X_int_val = torch.log(torch.tensor(X_int_val, dtype=torch.float) + 1)
            y_val = torch.tensor(y_val.astype(np.float32))

            X_cat_test = torch.tensor(X_cat_test, dtype=torch.long)
            X_int_test = torch.log(torch.tensor(X_int_test, dtype=torch.float) + 1)
            y_test = torch.tensor(y_test.astype(np.float32))

        print("Converted to tensors...done!")

        return (
            X_cat_train,
            X_int_train,
            y_train,
            X_cat_val,
            X_int_val,
            y_val,
            X_cat_test,
            X_int_test,
            y_test,
        )

    else:

        # randomize data
        if randomize == "total":
            indices = np.random.permutation(indices)

            print("Randomized indices...")

        X_cat = torch.tensor(X_cat[indices], dtype=torch.long)
        X_int = torch.log(torch.tensor(X_int[indices], dtype=torch.float) + 1)
        y = torch.tensor(y[indices].astype(np.float32))

        print("Converted to tensors...done!")

        return X_cat, X_int, y


def getCriteoAdData(
        datafile,
        o_filename,
        max_ind_range=-1,
        days=7,
        criteo_kaggle=True
):
    # Passes through entire dataset and defines dictionaries for categorical
    # features and determines the number of total categories.
    #
    # Inputs:
    #    datafile : path to downloaded raw data file
    #    o_filename (str): saves results under o_filename if filename is not ""
    #
    # Output:
    #   o_file (str): output file path

    #split the datafile into path and filename
    lstr = datafile.split("/")
    d_path = "/".join(lstr[0:-1]) + "/"
    npzfile = lstr[-1].split(".")[0] + "_day" if criteo_kaggle else lstr[-1]

    # count number of datapoints in training set
    total_count = 0
    if criteo_kaggle:
        # WARNING: The raw data consists of a single train.txt file
        # Each line in the file is a sample, consisting of 13 continuous and
        # 26 categorical features (an extra space indicates that feature is
        # missing and will be interpreted as 0).
        if path.exists(str(datafile)):
            print("Reading data from path=%s" % (str(datafile)))

            # file train.txt
            with open(str(datafile)) as f:
                for _ in f:
                    total_count += 1
        else:
            sys.exit("ERROR: Criteo Kaggle Display Ad Challenge Dataset path is invalid; please download from https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use")
    else:
        # WARNING: The raw data consist of day_0.gz,... ,day_23.gz text files
        # Each line in the file is a sample, consisting of 13 continuous and
        # 26 categorical features (an extra space indicates that feature is
        # missing and will be interpreted as 0).
        total_per_file = []
        for i in range(days):
            datafile_i = datafile + "_" + str(i)  # + ".gz"
            if path.exists(str(datafile_i)):
                print("Reading data from path=%s" % (str(datafile_i)))

                # file day_<number>
                total_per_file_count = 0
                with open(str(datafile_i)) as f:
                    for _ in f:
                        total_per_file_count += 1
                total_per_file.append(total_per_file_count)
                total_count += total_per_file_count
            else:
                sys.exit("ERROR: Criteo Terabyte Dataset path is invalid; please download from https://labs.criteo.com/2013/12/download-terabyte-click-logs")
    print("Total number of samples:", total_count)

    # determine length of split over days = {7|24}
    num_data_per_split, extras = divmod(total_count, days)
    if criteo_kaggle:
        print(
            "Samples are divided into splits, with num_data_per_split="
            + str(num_data_per_split) + " and extras=" + str(extras)
        )
    else:
        print("Samples are divided into splits by files representing each day")

    # process a file worth of data and reinitialize data
    # note that a file main contain a single or multiple splits
    def process_one_file(
            datafile,
            npzfile,
            total_per_file,
            split_offset,
            num_data_per_split,
            extras
    ):
        with open(str(datafile)) as f:
            # init variables
            count = 0
            split = split_offset
            # determine number of elements in a split and zero-out data
            if extras > 0:
                num_data_in_split = num_data_per_split + 1
                extras -= 1
            else:
                num_data_in_split = num_data_per_split
            y = np.zeros(num_data_in_split, dtype="i4")  # 4 byte int
            X_int = np.zeros((num_data_in_split, 13), dtype="i4")  # 4 byte int
            X_cat = np.zeros((num_data_in_split, 26), dtype="i4")  # 4 byte int

            for i, line in enumerate(f):
                # process a line (data point)
                # Approach 1: custom type
                '''
                # generate tuple for dtype and filling values
                # 1 label, 13 continuous and 26 categorical features
                criteo_type = np.dtype(
                    [
                        ("label", ("i4", 1)),
                        ("int_feature", ("i4", 13)),
                        ("cat_feature", ("U8", 26))
                    ]
                )
                data = np.genfromtxt(StringIO(line), dtype=criteo_type, delimiter="\t")
                y[i - count] = data["label"]
                X_int[i - count] = data["int_feature"]
                X_cat[i - count] = data["cat_feature"]
                '''
                #Approach 2: plain python
                line = line.split('\t')
                for j in range(len(line)):
                    if (line[j] == '') or (line[j] == '\n'):
                        line[j] = '0'

                y[i - count] = np.int32(line[0])
                X_int[i - count] = np.array(line[1:14], dtype=np.int32)
                if max_ind_range > 0:
                    X_cat[i - count] = np.array(
                        list(map(lambda x: int(x, 16) % max_ind_range, line[14:])),
                        dtype=np.int32
                    )
                else:
                    X_cat[i - count] = np.array(
                        list(map(lambda x: int(x, 16), line[14:])),
                        dtype=np.int32
                    )
                # count uniques
                # for j in range(26):
                #     convertDicts[j][X_cat[i - count][j]] = 1
                # count unique (unrolled)
                convertDicts[0][X_cat[i - count][0]] = 1
                convertDicts[1][X_cat[i - count][1]] = 1
                convertDicts[2][X_cat[i - count][2]] = 1
                convertDicts[3][X_cat[i - count][3]] = 1
                convertDicts[4][X_cat[i - count][4]] = 1
                convertDicts[5][X_cat[i - count][5]] = 1
                convertDicts[6][X_cat[i - count][6]] = 1
                convertDicts[7][X_cat[i - count][7]] = 1
                convertDicts[8][X_cat[i - count][8]] = 1
                convertDicts[9][X_cat[i - count][9]] = 1
                convertDicts[10][X_cat[i - count][10]] = 1
                convertDicts[11][X_cat[i - count][11]] = 1
                convertDicts[12][X_cat[i - count][12]] = 1
                convertDicts[13][X_cat[i - count][13]] = 1
                convertDicts[14][X_cat[i - count][14]] = 1
                convertDicts[15][X_cat[i - count][15]] = 1
                convertDicts[16][X_cat[i - count][16]] = 1
                convertDicts[17][X_cat[i - count][17]] = 1
                convertDicts[18][X_cat[i - count][18]] = 1
                convertDicts[19][X_cat[i - count][19]] = 1
                convertDicts[20][X_cat[i - count][20]] = 1
                convertDicts[21][X_cat[i - count][21]] = 1
                convertDicts[22][X_cat[i - count][22]] = 1
                convertDicts[23][X_cat[i - count][23]] = 1
                convertDicts[24][X_cat[i - count][24]] = 1
                convertDicts[25][X_cat[i - count][25]] = 1
                # debug prints
                print(
                    "Load %d/%d   Split: %d   Samples: %d  Label True: %d  Stored: %d"
                    % (
                        i,
                        total_per_file,
                        split,
                        num_data_in_split,
                        np.int32(line[0]),  # data["label"],
                        y[i - count],
                    ),
                    end="\r",
                )

                # store num_data_in_split samples or extras at the end of file
                if (i == (count + num_data_in_split - 1)) or (i == total_per_file - 1):
                    # count uniques
                    # X_cat_t  = np.transpose(X_cat)
                    # for j in range(26):
                    #     for x in X_cat_t[j,:]:
                    #         convertDicts[j][x] = 1
                    # store parsed
                    filename_s = d_path + npzfile + "_{0}.npz".format(split)
                    if path.exists(filename_s):
                        print("\nSkip existing " + filename_s)
                    else:
                        np.savez_compressed(
                            filename_s,
                            X_int=X_int,
                            # X_cat=X_cat,
                            X_cat_t=np.transpose(X_cat),  # transpose of the data
                            y=y,
                        )
                        print("\nSaved " + npzfile + "_{0}.npz!".format(split))

                    # determine number of elements in a split and zero-out data
                    if extras > 0:
                        num_data_in_split = num_data_per_split + 1
                        extras -= 1
                    else:
                        num_data_in_split = num_data_per_split
                    y = np.zeros(num_data_in_split, dtype="i4")
                    X_int = np.zeros((num_data_in_split, 13), dtype="i4")
                    X_cat = np.zeros((num_data_in_split, 26), dtype="i4")
                    # update variables
                    split += 1
                    count += i - count + 1   # num_data_in_split

        return split, count

    # create all splits (reuse existing files if possible)
    recreate_flag = False
    convertDicts = [{} for _ in range(26)]
    if criteo_kaggle:
        # in this case there are multiple splits in a day
        for i in range(days):
            npzfile_i = d_path + npzfile + "_{0}.npz".format(i)
            if path.exists(npzfile_i):
                print("Skip existing " + npzfile_i)
            else:
                recreate_flag = True
        if recreate_flag:
            split, count = process_one_file(
                datafile,
                npzfile,
                total_count,
                0,
                num_data_per_split,
                extras
            )
    else:
        # in this case there is a single split in each day
        for i in range(days):
            datafile_i = datafile + "_" + str(i)  # + ".gz"
            npzfile_i = d_path + npzfile + "_{0}.npz".format(i)
            if path.exists(npzfile_i):
                print("Skip existing " + npzfile_i)
            else:
                recreate_flag = True
                split, count = process_one_file(
                    datafile_i,
                    npzfile,
                    total_per_file[i],
                    i,
                    total_per_file[i],
                    0
                )

    # intermediate files
    counts = np.zeros(26, dtype=np.int32)
    if recreate_flag:
        # create dictionaries
        for j in range(26):
            for i, x in enumerate(convertDicts[j]):
                convertDicts[j][x] = i
            dict_file_j = d_path + npzfile + "_unique_feat_{0}.npz".format(j)
            if not path.exists(dict_file_j):
                np.savez_compressed(dict_file_j, unique=np.array(list(convertDicts[j])))
            counts[j] = len(convertDicts[j])
        # store (uniques and) counts
        count_file = d_path + npzfile + "_counts.npz"
        if not path.exists(count_file):
            np.savez_compressed(count_file, counts=counts)
    else:
        # create dictionaries (from existing files)
        for j in range(26):
            with np.load(d_path + npzfile + "_unique_feat_{0}.npz".format(j)) as data:
                unique = data["unique"]
            for i, x in enumerate(unique):
                convertDicts[j][x] = i
        # load (uniques and) counts
        with np.load(d_path + npzfile + "_counts.npz") as data:
            counts = data["counts"]

    # process all splits
    processCriteoAdData(d_path, npzfile, days, convertDicts, counts)
    o_file = concatCriteoAdData(d_path, npzfile, days, o_filename)

    return o_file


def loadDataset(dataset, max_ind_range, num_samples, raw_path="", pro_data=""):
    if dataset == "kaggle":
        days = 7
        df_exists = path.exists(str(pro_data))
        if df_exists:
            print("Reading pre-processed Criteo Kaggle data=%s" % (str(pro_data)))
            file = str(pro_data)
        else:
            print("Reading raw Criteo Kaggle data=%s" % (str(raw_path)))
            o_filename = "kaggleAdDisplayChallenge_processed"
            file = getCriteoAdData(raw_path, o_filename, max_ind_range, days, True)
    elif dataset == "terabyte":
        days = 24
        df_exists = path.exists(str(pro_data))
        if df_exists:
            print("Reading pre-processed Criteo Terabyte data=%s" % (str(pro_data)))
        else:
            print("Reading raw Criteo Terabyte data=%s" % (str(raw_path)))
            o_filename = "terabyte_processed"
            file = getCriteoAdData(raw_path, o_filename, max_ind_range, days, False)
    else:
        raise(ValueError("Data set option is not supported"))

    # load and preprocess data
    with np.load(file) as data:

        X_int = data["X_int"]
        X_cat = data["X_cat"]
        y = data["y"]
        counts = data["counts"]

    return X_cat, X_int, y, counts, days


if __name__ == "__main__":
    ### import packages ###
    import argparse

    ### parse arguments ###
    parser = argparse.ArgumentParser(
        description="Preprocess Criteo dataset"
    )
    # model related parameters
    parser.add_argument("--max-ind-range", type=int, default=-1)
    parser.add_argument("--data-set", type=str, default="kaggle")  # or terabyte
    parser.add_argument("--raw-data-file", type=str, default="")
    parser.add_argument("--processed-data-file", type=str, default="")
    args = parser.parse_args()

    loadDataset(
        args.data_set,
        args.max_ind_range,
        -1,
        args.raw_data_file,
        args.processed_data_file
    )
