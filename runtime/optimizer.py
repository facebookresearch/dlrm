# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import torch.optim
import time

from collections import deque  # Efficient ring buffer implementation.

class Version:
    def __init__(self, version=0):
        self.version = version

    def __repr__(self):
        return "v%d" % self.version

    def incr(self):
        return Version(version=self.version+1)

class OptimizerWithWeightStashing(torch.optim.Optimizer):
    """Wrapper class that adds weight stashing to a vanilla torch.optim.Optimizer.

    Arguments:
        - optim_name: the name of optimizer, required to create the corresponding
                      base_optimizer (torch.optim.{optim_name}).
        - optimizer_args: the keyword arguments passed to base_optimizer.
    """

    def __init__(self, optim_name, modules, master_parameters, model_parameters,
                 loss_scale, num_versions, verbose_freq=0, macrobatch=False,
                 **optimizer_args):
        self.modules = modules
        self.master_parameters = master_parameters
        self.model_parameters = model_parameters  # model_parameters is None if not fp16.
        self.loss_scale = loss_scale

        # Only need at most 2 versions if using macrobatching.
        if macrobatch:
            num_versions = min(2, num_versions)
        self.num_versions = num_versions
        self.base_optimizer = getattr(torch.optim, optim_name)(
            master_parameters, **optimizer_args)
        self.latest_version = Version()
        self.current_version = Version()
        self.initialize_queue()
        self.verbose_freq = verbose_freq
        self.batch_counter = 0

        # If macrobatching, push and pop versions at the right rate.
        if macrobatch:
            self.update_interval = self.num_versions
        else:
            self.update_interval = 1

    def __getattr__(self, key):
        """Relay the unknown key to base_optimizer."""
        return getattr(self.base_optimizer, key)

    def initialize_queue(self):
        self.queue = deque(maxlen=self.num_versions)
        for i in range(self.num_versions):
            self.queue.append(self.get_params(clone=True))
        self.buffered_state_dicts = self.queue[0][0]

    def get_params(self, clone):
        if clone:
            state_dicts = []
            for module in self.modules:
                state_dict = module.state_dict()
                for key in state_dict:
                    state_dict[key] = state_dict[key].clone()
                state_dicts.append(state_dict)
        else:
            for i, module in enumerate(self.modules):
                state_dict = module.state_dict()
                for key in state_dict:
                    # Running_mean and running_var for batchnorm layers should
                    # accumulate normally.
                    if "running_" in key:
                        continue
                    if "mask" in key:
                        self.buffered_state_dicts[i][key] = state_dict[key].clone()
                    else:
                        self.buffered_state_dicts[i][key].copy_(state_dict[key])
            state_dicts = self.buffered_state_dicts
        return state_dicts, self.latest_version

    def set_params(self, state_dicts, version):
        for (state_dict, module) in zip(state_dicts, self.modules):
            cur_state_dict = module.state_dict()
            for key in state_dict:
                # Don't update running_mean and running_var; these should
                # accumulate normally.
                # mask might have a different shape, so don't copy it to
                # the module this way.
                if "running_" in key or "mask" in key:
                    state_dict[key] = cur_state_dict[key]
            module.load_state_dict(state_dict)

            # Load the mask.
            for key in state_dict:
                if "mask" in key:
                    attribute_names = key.split(".")
                    attribute = module
                    for attribute_name in attribute_names:
                        attribute = getattr(attribute, attribute_name)
                    # NOTE: Do we need to clone here?
                    attribute = state_dict[key]
        self.current_version = version

    def load_old_params(self):
        if self.num_versions > 1:
            self.set_params(*self.queue[0])

    def load_new_params(self):
        if self.num_versions > 1:
            self.set_params(*self.queue[-1])

    def zero_grad(self):
        if self.batch_counter % self.update_interval == 0:
            self.base_optimizer.zero_grad()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                                          and returns the loss.
        """
        # Update the gradient every `update_interval` steps.
        if self.batch_counter % self.update_interval != self.update_interval - 1:
            self.batch_counter += 1
            return None

        log_timing = self.verbose_freq > 0 and self.batch_counter % self.verbose_freq == 0
        if log_timing:
            start_time = time.time()
        if self.model_parameters is not None:
            import apex.fp16_utils as fp16_utils
            fp16_utils.model_grads_to_master_grads(self.model_parameters,
                                                   self.master_parameters)
            # TODO: This division might not be in the right place, given that
            # scaling happens right after. Look into this if problems arise.
            if self.loss_scale != 1.0:
                for parameter in self.master_parameters:
                    parameter.grad.data = parameter.grad.data / self.loss_scale

        for p in self.param_groups[0]['params']:
            if p.grad is not None:
                p.grad.div_(self.update_interval)

        loss = self.base_optimizer.step()
        if self.model_parameters is not None:
            import apex.fp16_utils as fp16_utils
            fp16_utils.master_params_to_model_params(self.model_parameters,
                                                     self.master_parameters)
        self.latest_version = self.latest_version.incr()
        if self.num_versions > 1:
            self.buffered_state_dicts = self.queue[0][0]
            self.queue.append(self.get_params(clone=False))

        if log_timing:
            print("Optimizer step took: %.3f" % (time.time() - start_time))
        self.batch_counter += 1
        return loss
