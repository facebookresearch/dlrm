# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.


import torch
from torch.optim import Optimizer


class RWSAdagrad(Optimizer):
    """Implements Row Wise Sparse Adagrad algorithm.

    Arguments:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float, optional): learning rate (default: 1e-2)
        lr_decay (float, optional): learning rate decay (default: 0)
        weight_decay (float, optional): weight decay (L2 penalty) (default: 0)
        eps (float, optional): term added to the denominator to improve
            numerical stability (default: 1e-10)

    """

    def __init__(self, params, lr=1e-2, lr_decay=0.0, weight_decay=0.0, initial_accumulator_value=0.0, eps=1e-10):
        if not 0.0 <= lr:
            raise ValueError("Invalid learning rate: {}".format(lr))
        if not 0.0 <= lr_decay:
            raise ValueError("Invalid lr_decay value: {}".format(lr_decay))
        if not 0.0 <= weight_decay:
            raise ValueError("Invalid weight_decay value: {}".format(weight_decay))
        if not 0.0 <= initial_accumulator_value:
            raise ValueError("Invalid initial_accumulator_value value: {}".format(initial_accumulator_value))
        if not 0.0 <= eps:
            raise ValueError("Invalid epsilon value: {}".format(eps))

        self.defaults = dict(lr=lr, lr_decay=lr_decay, eps=eps, weight_decay=weight_decay,
                        initial_accumulator_value=initial_accumulator_value)
        super(RWSAdagrad, self).__init__(params, self.defaults)

        self.momentum_initialized = False

        for group in self.param_groups:
            for p in group['params']:
                self.state[p]['step'] = 0

    def share_memory(self):
        for group in self.param_groups:
            for p in group['params']:
                state = self.state[p]
                if p.grad.data.is_sparse:
                    state['momentum'].share_memory_()
                else:
                    state['sum'].share_memory_()

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue

                if not self.momentum_initialized :
                    if p.grad.data.is_sparse:
                        self.state[p]['momentum'] = torch.full(
                            [p.data.shape[0]],
                            self.defaults["initial_accumulator_value"],
                            dtype=torch.float32,
                        )
                    else:
                        self.state[p]['sum'] = torch.full_like(p.data,
                            self.defaults["initial_accumulator_value"],
                            dtype=torch.float32,
                        )

                grad = p.grad
                state = self.state[p]

                state['step'] += 1

                if group['weight_decay'] != 0:
                    if p.grad.data.is_sparse:
                        raise RuntimeError("weight_decay option is not compatible with sparse gradients")
                    grad = grad.add(group['weight_decay'], p.data)

                clr = group['lr'] / (1.0 + (state['step'] - 1.0) * group['lr_decay'])

                if grad.is_sparse:
                    grad = grad.coalesce()  # the update is non-linear so indices must be unique
                    grad_indices = grad._indices()
                    grad_values = grad._values()
                    size = grad.size()

                    def make_sparse(values, row_wise):
                        constructor = grad.new
                        matrix_size = [size[0]] if row_wise else size
                        return constructor(grad_indices, values, matrix_size)

                    if grad_values.numel() > 0:
                        momentum_update = make_sparse(grad_values.pow(2).mean(dim=1), True)
                        state['momentum'].add_(momentum_update)  # update momentum
                        std = state['momentum'].sparse_mask(momentum_update.coalesce())
                        std_values = std._values().sqrt_().add_(group['eps'])
                        p.data.add_(make_sparse(grad_values / std_values.view(std_values.size()[0], 1), False), alpha=-clr)

                else:
                    state['sum'].addcmul_(grad, grad, value=1.0)
                    std = state['sum'].sqrt().add_(group['eps'])
                    p.data.addcdiv_(grad, std, value=-clr)

        self.momentum_initialized = True

        return loss
