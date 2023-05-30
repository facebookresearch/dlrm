#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

base_url="https://storage.googleapis.com/criteo-cail-datasets/day_"
for i in {0..23}; do
  url="$base_url$i.gz"
  echo Downloading "$url"
  wget "$url"
done
