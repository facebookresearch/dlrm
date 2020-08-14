# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

from torch.optim.optimizer import required

from optimizer import OptimizerWithWeightStashing

class SGDWithWeightStashing(OptimizerWithWeightStashing):
    """
    SGD optimizer with weight stashing.
    """
    def __init__(self, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, lr=required, momentum=0,
                 dampening=0, weight_decay=0, nesterov=False, verbose_freq=0,
                 macrobatch=False):
        super(SGDWithWeightStashing, self).__init__(
            optim_name='SGD',
            modules=modules, master_parameters=master_parameters,
            model_parameters=model_parameters, loss_scale=loss_scale,
            num_versions=num_versions, lr=lr, momentum=momentum,
            dampening=dampening, weight_decay=weight_decay,
            nesterov=nesterov, verbose_freq=verbose_freq,
            macrobatch=macrobatch,
        )
