# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import os
import pickle
import torch


class ActivationAndGradientLogger:
    def __init__(self, directory):
        self.directory = directory
        try:
            os.mkdir(self.directory)
        except:
            pass
        self.iteration = 0
        self.forward_counter = 0
        self.backward_counter = 0

    def reset_counters(self):
        self.forward_counter = 0
        self.backward_counter = 0

    def hook_modules(self, module, iteration):
        self.iteration = iteration
        sub_directory = os.path.join(self.directory, str(iteration))
        try:
            os.mkdir(sub_directory)
        except:
            pass
        self.hook_modules_helper(module, sub_directory)

    def hook_modules_helper(self, module, sub_directory):
        sub_modules = module.__dict__['_modules']

        for name, sub_module in sub_modules.items():
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0:
                # Recursively visit this module's descendants.
                self.hook_modules_helper(sub_module, sub_directory)
            else:
                def forward_hook(*args):
                    activation = args[2]
                    filename = os.path.join(sub_directory, 'activations.%d.pkl' % self.forward_counter)
                    with open(filename, 'wb') as f:
                        torch.save(activation, f)
                    self.forward_counter += 1

                def backward_hook(*args):
                    gradient = args[2]
                    filename = os.path.join(sub_directory, 'gradients.%d.pkl' % self.backward_counter)
                    with open(filename, 'wb') as f:
                        torch.save(gradient, f)
                    self.backward_counter += 1

                sub_module.register_forward_hook(forward_hook)
                sub_module.register_backward_hook(backward_hook)

    def unhook_modules(self, module):
        self.unhook_modules_helper(module)
        self.reset_counters()

    def unhook_modules_helper(self, module):
        sub_modules = module.__dict__['_modules']

        for name, sub_module in sub_modules.items():
            if sub_module is None or isinstance(sub_module, torch.nn.Module) is False:
                break

            sub_sub_modules = sub_module.__dict__['_modules']
            if len(sub_sub_modules) > 0:
                # Recursively visit this module's descendants.
                self.unhook_modules_helper(sub_module)
            else:
                sub_module.reset_hooks()
