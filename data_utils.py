# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Description: generate inputs and targets for the DLRM benchmark
#
# Utility function(s) to download and pre-process public data sets
#   - Kaggle Display Advertising Challenge Dataset
#       (https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/)
#
# After downloading dataset, run:
#   getKaggleCriteoAdData(datafile="<path-to-train.txt>", o_filename=kaggle_processed.npz")
#
# TODO: add support for other data-sets

from __future__ import absolute_import, division, print_function, unicode_literals

import os.path
from io import StringIO
from os import path

import numpy as np
import torch


def convertUStringToDistinctInts(mat, convertDicts, counts):
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
    out = torch.zeros(mat.shape)

    for j in range(mat.shape[1]):
        for i in range(mat.shape[0]):
            # add to convertDict and increment count
            if mat[i, j] not in convertDicts[j]:
                convertDicts[j][mat[i, j]] = counts[j]
                counts[j] += 1
            out[i, j] = convertDicts[j][mat[i, j]]

    return out, convertDicts, counts


def processKaggleCriteoAdData(split, d_path):
    # Process Kaggle Display Advertising Challenge Dataset by converting unicode strings
    # in X_cat to integers and converting negative integer values in X_int.
    #
    # Loads data in the form "kaggle_day_i.npz" where i is the day.
    #
    # Inputs:
    #   split (int): total number of splits in the dataset (typically 7)
    #   d_path (str): path for kaggle_day_i.npz files

    convertDicts = []
    counts = []

    # check if processed file already exists
    idx = 1
    while idx <= split:
        if path.exists(str(d_path) + "kaggle_day_{0}_processed.npz".format(idx)):
            idx += 1
        else:
            break

    # process data if not all files exist
    if idx <= split:
        for i in range(1, split + 1):
            with np.load(str(d_path) + "kaggle_day_{0}.npz".format(i)) as data:

                X_cat, convertDicts, counts = convertUStringToDistinctInts(
                    data["X_cat"], convertDicts, counts
                )
                X_int = data["X_int"]
                X_int[X_int < 0] = 0
                y = data["y"]

            np.savez_compressed(
                str(d_path) + "kaggle_day_{0}_processed.npz".format(i),
                X_cat=X_cat,
                X_int=X_int,
                y=y,
            )
            print("Processed kaggle_day_{0}.npz...".format(i), end="\r")

        np.savez_compressed(str(d_path) + "kaggle_counts.npz", counts=counts)
    else:
        print("Using existing %skaggle_day_*_processed.npz files" % str(d_path))

    return


def concatKaggleCriteoAdData(split, d_path, o_filename):
    # Concatenates different days of Kaggle data and saves.
    #
    # Inputs:
    #   split (int): total number of splits in the dataset (typically 7)
    #   d_path (str): path for kaggle_day_i.npz files
    #   o_filename (str): output file name
    #
    # Output:
    #   o_file (str): output file path

    print ("Concatenating multiple day kaggle data into %s.npz file" % str(d_path + o_filename))

    # load and concatenate data
    for i in range(1, split + 1):
        with np.load(str(d_path) + "kaggle_day_{0}_processed.npz".format(i)) as data:

            if i == 1:
                X_cat = data["X_cat"]
                X_int = data["X_int"]
                y = data["y"]

            else:
                X_cat = np.concatenate((X_cat, data["X_cat"]))
                X_int = np.concatenate((X_int, data["X_int"]))
                y = np.concatenate((y, data["y"]))

        print("Loaded day:", i, "y = 1:", len(y[y == 1]), "y = 0:", len(y[y == 0]))

    with np.load(str(d_path) + "kaggle_counts.npz") as data:

        counts = data["counts"]

    print("Loaded counts!")

    np.savez_compressed(
        str(d_path) + str(o_filename) + ".npz",
        X_cat=X_cat,
        X_int=X_int,
        y=y,
        counts=counts,
    )

    return str(d_path) + str(o_filename) + ".npz"


def transformCriteoAdData(X_cat, X_int, y, split, randomize, cuda):
    # Transforms Kaggle data by applying log transformation on dense features and
    # converting everything to appropriate tensors.
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

        indices = np.array_split(indices, 7)

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


def getKaggleCriteoAdData(datafile="", o_filename=""):
    # Passes through entire dataset and defines dictionaries for categorical
    # features and determines the number of total categories.
    #
    # Inputs:
    #    datafile : path to downloaded raw data file
    #    o_filename (str): saves results under o_filename if filename is not ""
    #
    # Output:
    #   o_file (str): output file path

    d_path = "./kaggle_data/"

    # determine if intermediate data path exists
    if path.isdir(str(d_path)):
        print("Saving intermediate data files at %s" % (d_path))
    else:
        os.mkdir(str(d_path))
        print("Created %s for storing intermediate data files" % (d_path))

    # determine if data file exists (train.txt)
    if path.exists(str(datafile)):
        print("Reading data from path=%s" % (str(datafile)))
    else:
        print(
            "Path of Kaggle Display Ad Challenge Dataset is invalid; please download from https://labs.criteo.com/2014/09/kaggle-contest-dataset-now-available-academic-use/"
        )
        exit(0)

    # count number of datapoints in training set
    total_count = 0
    with open(str(datafile)) as f:
        for _ in f:
            total_count += 1

    print("Total number of datapoints:", total_count)

    # determine length of split over 7 days
    split = 1
    num_data_per_split, extras = divmod(total_count, 7)

    # generate tuple for dtype and filling values
    type = np.dtype(
        [("label", ("i4", 1)), ("int_feature", ("i4", 13)), ("cat_feature", ("U8", 26))]
    )

    # initialize data to store
    if extras > 0:
        num_data_in_split = num_data_per_split + 1
        extras -= 1

    y = np.zeros(num_data_in_split, dtype="i4")
    X_int = np.zeros((num_data_in_split, 13), dtype="i4")
    X_cat = np.zeros((num_data_in_split, 26), dtype="U8")

    # check if files exist
    while split <= 7:
        if path.exists(str(str(d_path) + "kaggle_day_{0}.npz".format(split))):
            split += 1
        else:
            split = 1
            break

    count = 0
    if split == 1:
        # load training data
        with open(str(datafile)) as f:

            for i, line in enumerate(f):

                # store day"s worth of data and reinitialize data
                if i == (count + num_data_in_split):
                    np.savez_compressed(
                        str(d_path) + "kaggle_day_{0}.npz".format(split),
                        X_int=X_int,
                        X_cat=X_cat,
                        y=y,
                    )

                    print("\nSaved kaggle_day_{0}.npz!".format(split))

                    split += 1
                    count += num_data_in_split

                    if extras > 0:
                        num_data_in_split = num_data_per_split + 1
                        extras -= 1

                    y = np.zeros(num_data_in_split, dtype="i4")
                    X_int = np.zeros((num_data_in_split, 13), dtype="i4")
                    X_cat = np.zeros((num_data_in_split, 26), dtype="U8")

                data = np.genfromtxt(StringIO(line), dtype=type, delimiter="\t")

                y[i - count] = data["label"]
                X_int[i - count] = data["int_feature"]
                X_cat[i - count] = data["cat_feature"]

                print(
                    "Loading %d/%d   Split: %d   No Data in Split: %d  true label: %d  stored label: %d"
                    % (
                        i,
                        total_count,
                        split,
                        num_data_in_split,
                        data["label"],
                        y[i - count],
                    ),
                    end="\r",
                )

        np.savez_compressed(
            str(d_path) + "kaggle_day_{0}.npz".format(split),
            X_int=X_int,
            X_cat=X_cat,
            y=y,
        )

        print("\nSaved kaggle_day_{0}.npz!".format(split))
    else:
        print("Using existing %skaggle_day_*.npz files" % str(d_path))

    processKaggleCriteoAdData(split, d_path)
    o_file = concatKaggleCriteoAdData(split, d_path, o_filename)

    return o_file


def loadDataset(dataset, num_samples, df_path="", data=""):
    if dataset == "kaggle":
        df_exists = path.exists(str(data))
        if df_exists:
            print("Reading from pre-processed data=%s" % (str(data)))
            file = str(data)
        else:
            o_filename = "kaggleAdDisplayChallenge_processed"
            file = getKaggleCriteoAdData(df_path, o_filename)
    elif dataset == "terabyte":
        file = "./terbyte_data/tb_processed.npz"
        df_exists = path.exists(str(file))
        if df_exists:
            print("Reading Terabyte data-set processed data from %s" % file)
        else:
            raise (
                ValueError(
                    "Terabyte data-set processed data file %s does not exist !!" % file
                )
            )

    # load and preprocess data
    with np.load(file) as data:

        X_int = data["X_int"]
        X_cat = data["X_cat"]
        y = data["y"]
        counts = data["counts"]

    return X_cat, X_int, y, counts
