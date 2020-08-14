# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import sys
sys.path.append("..")
sys.path.append("../..")
import utils

if __name__ == '__main__':
    utils.parse_profile_file_to_graph("data/vgg_v100.csv", "data/graphs/vgg_v100")
    utils.parse_profile_file_to_graph("data/vgg.csv", "data/graphs/vgg")
    utils.parse_profile_file_to_graph("data/alexnet.csv", "data/graphs/alexnet")
