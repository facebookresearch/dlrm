#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import logging
from dataclasses import dataclass
from typing import List, Dict, Optional

import torch
import torchrec.distributed as trec_dist
from torchrec.datasets.criteo import (  # noqa
    DEFAULT_INT_NAMES,
    DEFAULT_CAT_NAMES,
    CAT_FEATURE_COUNT,
)
from torchrec.inference.model_packager import load_pickle_config
from torchrec.inference.modules import (
    PredictFactory,
    PredictModule,
    quantize_embeddings,
)
from torchrec.models.dlrm import DLRM
from torchrec.modules.embedding_configs import EmbeddingBagConfig
from torchrec.modules.embedding_modules import EmbeddingBagCollection
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor

logger: logging.Logger = logging.getLogger(__name__)

# OSS Only


@dataclass
class DLRMModelConfig:
    dense_arch_layer_sizes: List[int]
    dense_in_features: int
    embedding_dim: int
    id_list_features_keys: List[str]
    num_embeddings_per_feature: List[int]
    num_embeddings: int
    over_arch_layer_sizes: List[int]


class DLRMPredictModule(PredictModule):
    """
    nn.Module to wrap DLRM model to use for inference.

    Args:
        embedding_bag_collection (EmbeddingBagCollection): collection of embedding bags
            used to define SparseArch.
        dense_in_features (int): the dimensionality of the dense input features.
        dense_arch_layer_sizes (List[int]): the layer sizes for the DenseArch.
        over_arch_layer_sizes (List[int]): the layer sizes for the OverArch. NOTE: The
            output dimension of the InteractionArch should not be manually specified
            here.
        id_list_features_keys (List[str]): the names of the sparse features. Used to
            construct a batch for inference.
        dense_device: (Optional[torch.device]).
    """

    def __init__(
        self,
        embedding_bag_collection: EmbeddingBagCollection,
        dense_in_features: int,
        dense_arch_layer_sizes: List[int],
        over_arch_layer_sizes: List[int],
        id_list_features_keys: List[str],
        dense_device: Optional[torch.device] = None,
    ) -> None:
        module = DLRM(
            embedding_bag_collection=embedding_bag_collection,
            dense_in_features=dense_in_features,
            dense_arch_layer_sizes=dense_arch_layer_sizes,
            over_arch_layer_sizes=over_arch_layer_sizes,
            dense_device=dense_device,
        )
        super().__init__(module, dense_device)

        self.id_list_features_keys: List[str] = id_list_features_keys

    def predict_forward(
        self, batch: Dict[str, torch.Tensor]
    ) -> Dict[str, torch.Tensor]:
        """
        Args:
            batch (Dict[str, torch.Tensor]): currently expects input dense features
                to be mapped to the key "float_features" and input sparse features
                to be mapped to the key "id_list_features".

        Returns:
            Dict[str, torch.Tensor]: output of inference.
        """
        try:
            predictions = self.predict_module(
                batch["float_features"],
                KeyedJaggedTensor(
                    keys=self.id_list_features_keys,
                    lengths=batch["id_list_features.lengths"],
                    values=batch["id_list_features.values"],
                ),
            )
        except Exception as e:
            logger.info(e)
            raise e
        return {"default": predictions.to(torch.device("cpu")).float()}


class DLRMPredictFactory(PredictFactory):
    def __init__(self) -> None:
        self.model_config: DLRMModelConfig = load_pickle_config(
            "config.pkl", DLRMModelConfig
        )

    def create_predict_module(self, rank: int, world_size: int) -> torch.nn.Module:
        logging.basicConfig(level=logging.INFO)
        device = torch.device("cuda", rank)
        torch.cuda.set_device(device)
        trec_dist.DistributedModelParallel.SHARE_SHARDED = True

        eb_configs = [
            EmbeddingBagConfig(
                name=f"t_{feature_name}",
                embedding_dim=self.model_config.embedding_dim,
                num_embeddings=self.model_config.num_embeddings_per_feature[feature_idx]
                if self.model_config.num_embeddings is None
                else self.model_config.num_embeddings,
                feature_names=[feature_name],
            )
            for feature_idx, feature_name in enumerate(
                self.model_config.id_list_features_keys
            )
        ]
        ebc = EmbeddingBagCollection(tables=eb_configs, device=torch.device("meta"))

        module = DLRMPredictModule(
            embedding_bag_collection=ebc,
            dense_in_features=self.model_config.dense_in_features,
            dense_arch_layer_sizes=self.model_config.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.model_config.over_arch_layer_sizes,
            dense_device=device,
        )
        module = quantize_embeddings(module, dtype=torch.qint8, inplace=True)
        return trec_dist.DistributedModelParallel(
            module=module,
            device=device,
            env=trec_dist.ShardingEnv.from_local(world_size, rank),
            init_data_parallel=False,
        )

    def batching_metadata(self) -> Dict[str, str]:
        return {
            "float_features": "dense",
            "id_list_features": "sparse",
        }
