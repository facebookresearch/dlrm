#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

display_help() {
   echo "Three command line arguments are required."
   echo "Example usage:"
   echo "bash process_Criteo_1TB_Click_Logs_dataset.sh \\"
   echo "./criteo_1tb/raw_input_dataset_dir \\"
   echo "./criteo_1tb/temp_intermediate_files_dir \\"
   echo "./criteo_1tb/numpy_contiguous_shuffled_output_dataset_dir"
   exit 1
}

[ -z "$1" ] && display_help
[ -z "$2" ] && display_help
[ -z "$3" ] && display_help

# Input directory containing the raw Criteo 1TB Click Logs dataset files in tsv format.
# The 24 dataset filenames in the directory should be day_{0..23} with no .tsv extension.
raw_tsv_criteo_files_dir=$(readlink -m "$1")

# Directory to store temporary intermediate output files created by preprocessing steps 1 and 2.
temp_files_dir=$(readlink -m "$2")

# Directory to store temporary intermediate output files created by preprocessing step 1.
step_1_output_dir="$temp_files_dir/temp_output_of_step_1"

# Directory to store temporary intermediate output files created by preprocessing step 2.
step_2_output_dir="$temp_files_dir/temp_output_of_step_2"

# Directory to store the final preprocessed Criteo 1TB Click Logs dataset.
step_3_output_dir=$(readlink -m "$3")

# Step 1. Split the dataset into 3 sets of 24 numpy files:
# day_{0..23}_dense.npy, day_{0..23}_labels.npy, and day_{0..23}_sparse.npy (~24hrs)
set -x
mkdir -p "$step_1_output_dir"
date
python -m torchrec.datasets.scripts.npy_preproc_criteo --input_dir "$raw_tsv_criteo_files_dir" --output_dir "$step_1_output_dir" || exit

# Step 2. Convert all sparse indices in day_{0..23}_sparse.npy to contiguous indices and save the output.
# The output filenames are day_{0..23}_sparse_contig_freq.npy
mkdir -p "$step_2_output_dir"
date
python -m torchrec.datasets.scripts.contiguous_preproc_criteo --input_dir "$step_1_output_dir" --output_dir "$step_2_output_dir" --frequency_threshold 0 || exit

date
for i in {0..23}
do
   name="$step_2_output_dir/day_$i""_sparse_contig_freq.npy"
   renamed="$step_2_output_dir/day_$i""_sparse.npy"
   echo "Renaming $name to $renamed"
   mv "$name" "$renamed"
done

# Step 3. Shuffle the dataset's samples in days 0 through 22. (~20hrs)
# Day 23's samples are not shuffled and will be used for the validation set and test set.
mkdir -p "$step_3_output_dir"
date
python -m torchrec.datasets.scripts.shuffle_preproc_criteo --input_dir_labels_and_dense "$step_1_output_dir" --input_dir_sparse "$step_2_output_dir" --output_dir_shuffled "$step_3_output_dir" --random_seed 0 || exit
date
