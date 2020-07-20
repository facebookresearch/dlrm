# Version on July 16th 2020, written by Wenbo Ren (WenboRen@GitHub)

import torch
import torch.nn.parallel
import torch.distributed
import torch.multiprocessing
import torch.utils.data
import torch.utils.data.distributed
import torch.distributed as dist
import math


# In principle, the only public interface
def get_distributed_optimizer(alg, local_optimizer, rank, world_size, group, arg_dict):
    """Public interface to get the distributed optimizer specified by alg

    Arguments:
        alg (str): specifies the distributed optimization algorithm.
            Current version supports
                -(Post-)local SGD. When initial_steps = 0, Post-local SGD
                    reduces to local SGD
            Raise error if alg is not specified or supported

        local_optimizer (torch.optim.optimizer): The optimizer used for the
            local optimization

        rank (int): the rank of this process

        world_size (int): number of processes in the group this process
            belongs to

        group (list of int): the list of process ids that belong to this process
            group

        arg_dict (dictionary <str key type>): specifies the arguments for
            the distributed optimizer.

    Returns an instace of the distrbuted optimizer for the specified algorithm

    Note: If want to let some param_groups not processed by distributed
        optimizer, then add a pair ('allow_data_parallelism', False) in the
        corresponding param_group of the local_optimizer.

    Supported algorthms:
    1. (Post)-Local SGD.
        Introduction:
            First perform the initial step method for $initial_steps steps, and
            then perform the local SGD algorithm, i.e., each process steps on
            its own samples, and all processes average the models after every
            $local_steps steps. When $initial_steps = 0, (post)-local SGD
            algorithm reduces to local SGD.
            "Lin, T., Stich, S. U., Patel, K. K., & Jaggi, M. (2018).
            Don't Use Large Mini-Batches, Use Local SGD.
            arXiv preprint arXiv:1808.07217."
        Arguments:
            'initial_steps' (int): number of initial global steps (default 0)
            'local_steps' (int): number of local steps betwen two weights
                average (default 4)
            'init_step_method' (str): method for running initial steps (default
                'single_process')
        init_step_method:
            1. 'multiple_processes': Run initial steps on all processes and
                average models after each step.
            2. 'single_process': Run initial steps on one process and then
                copy the model to other processes.
    2. Hierarchical Local SGD
        Introduction:
            Split the processes to multiple nodes, and each node has multiple
            processes. Each process of each node steps with its own samples.
            For every certain number of steps, inside each node, the processes
            call all reduces to average their models (node-level sync). For every
            crtain number of node-level sncs, all processes of all nodes call
            all reduces to average their models (global sync).
            "Lin, T., Stich, S. U., Patel, K. K., & Jaggi, M. (2018).
            Don't Use Large Mini-Batches, Use Local SGD.
            arXiv preprint arXiv:1808.07217."
        Arguments:
            'local_sync_freq' (int): number of steps between two node-level syncs.
            'global_sync_freq' (int): number of node-level syncs between two
                global syncs.
            'node_size' (int): the number of processes inside the node that
                continas this process.
            'node_id' (int): the index of the node that this process belongs to

    Addons: based on the distributed optimizers, apply addons algorithms.
    Supported addons:
    1. Injecting noise.
        Introduction:
            Before the optimizer steps, injecting noise to the gradients of the
            parameters. It also supports linear variance decay, i.e., after each
            step, reduce the variance of the noise by a given factor.
        Arguments:
            'add_noise' (Boolean): Set it True to trigger on injecting noise.
            'noise_type' (str): the type of the noise. Current version supports
                'gaussian' and 'multiplicative_gaussian', where for gaussian noise,
                we have gradient = gradien + noise, and for multiplicative_gaussian
                noise, we hve gradient = gradient *(1 + noise).
            'std' (float): the standard deviation of the noise (optional if
                'variance' has been specified).
            'variance' (float): the variance of the noise (optional if 'std' has
                been specified).
            'linear_variance_decay' (float): set it postitie to trigger on linear
                variance decay, i.e., after each step, the variace is decrease by
                variance * linear_variance_decay.
    2. Slow Momentum.
        Inroduction:
            After every certain number of steps: 1) do a global model average;
            2) compute the model change compared to the last global model average;
            3) use the model change to update the slow momentum; 4) update the
            model parameters according the slow momentum. See details in
            "Wang, J., Tantia, V., Ballas, N., & Rabbat, M. (2019).
            SlowMo: Improving communication-efficient distributed SGD with slow
            momentum. arXiv preprint arXiv:1910.00643."
        Arguments:
            'slow_momentum' (boolean): set true to trigger on slow momentum.
            'inner_loop_steps' (int): number of steps between two momentum updates.
            'slow_learning_rate" (float): the value of slow_learning_rate.
            'slow_momentum_factor' (float): the value of slow_learning_factor.

    """
    if alg in {'local_sgd', 'post_local_sgd'}:
        return PostLocalSGD(local_optimizer, rank, world_size, group, arg_dict)
    elif alg == 'hierarchical_local_sgd':
        return HierarchicalLocalSGD(local_optimizer, rank, world_size, group, arg_dict)
    raise ValueError('Algorithm {} not specified or not supported. \
        Supported algorithms include \'local_sgd\', \'post_local_sgd\', and \
        \'hierarchical_local_sgd\''.format(alg))


# Virtual class, father of all optimization algorithms
# Should not declare an instance of it
class DistributedOptimizer():
    # Public, constructor
    def __init__(self, local_optimizer, rank, world_size, group, arg_dict):
        self._rank = rank
        self._world_size = world_size
        self._group = group
        self.set_process_group()
        self.local_optimizer = local_optimizer

        self._step_counter = 0

        # self._state is used for sharing state information
        self._state = {
            'disable_synchronization': False,
            'synced': False,
        }

        # self._addons is for registering the addons of the distributed optimizer.
        self._addons = {}
        # The following method is to load the states and parameters of the addons.
        self.load_addons(arg_dict)

    # Public, awaiting implementation by successors
    def step(self):
        self._step_counter += 1

        # Process the addons
        if "add_noise" in self._addons:
            self.add_noise()

        with torch.no_grad():
            self.local_optimizer.step()
        self._state['synced'] = False

        # Process addons
        if 'slow_momentum' in self._addons:
            self.slow_momentum()

    # Public
    def zero_grad(self):
        self.local_optimizer.zero_grad()

    # Public
    # Current version of state_dict() does not support addons
    def state_dict(self):
        dict = self.local_optimizer.state_dict()
        dict['rank'] = self._rank
        dict['world_size'] = self._world_size
        dict['group'] = self._group
        dict['step_counter'] = self._step_counter
        return dict

    # Public
    # Current version of load_state_dict() does not support addons
    def load_state_dict(self, dict):
        self.local_optimizer.load_state_dict(dict)
        self._rank = dict['rank']
        self._world_size = dict['world_size']
        self._group = dict['group']
        self.set_process_group()
        self._step_counter = dict['step_counter']

    # Private, for setting the process group
    def set_process_group(self):
        self._process_group = dist.group.WORLD if self._group is None \
            else dist.new_group(ranks=self._group)

    # Private
    def load_addons(self, arg_dict):
        if "add_noise" in arg_dict and arg_dict["add_noise"]:
            self._addons["add_noise"] = True

            # Default noise type is Gaussian
            self._noise_type = self.load_value_from_dict(
                "noise_type", arg_dict, default_value="gaussian"
            )
            if self._noise_type not in {"gaussian", "multiplicative_gaussian"}:
                raise ValueError(
                    "Noise type {} not supported. Curent version \
                    supports \'gaussian\' and \'muliplicative_gaussian\'"
                    .format(arg_dict["noise_type"])
                )

            # Users can either give the standard deviation or the variance
            if "std" in arg_dict:
                self._std = arg_dict["std"]
            elif "variance" in arg_dict:
                self._std = math.sqrt(arg_dict["variance"])
            else:
                raise ValueError(
                    "neither \'std\' nor \'variance' is specified in \'ar_dict\'"
                )

            self._variance_ratio = 1.0
            if "linear_variance_decay" in arg_dict:
                self._linear_variance_decay = arg_dict["linear_variance_decay"]
                self._decay_method = "linear"
            else:
                self._decay_method = "none"

        if 'slow_momentum' in arg_dict and arg_dict['slow_momentum']:
            self._addons['slow_momentum'] = True
            self._inner_loop_steps = self.load_value_from_dict(
                'inner_loop_steps', arg_dict
            )
            self._slow_learning_rate = self.load_value_from_dict(
                'slow_learning_rate', arg_dict
            )
            self._slow_momentum_factor = self.load_value_from_dict(
                'slow_momentum_factor', arg_dict
            )

            self._momentum_buffer = []
            self._previous_param_data = []
            self.average_weights()
            for group in self.local_optimizer.param_groups:
                if self.should_process_group(group):
                    for param in group['params']:
                        device = param.data.get_device()
                        # Initialize momentums
                        self._momentum_buffer.append(
                            torch.zeros(param.data.shape, device=device)
                        )
                        # Memorize the initial model
                        self._previous_param_data.append(param.data.detach().clone())

    # Private
    # Add noise to the gradients
    def add_noise(self):
        if self._decay_method == "linear":
            self._variance_ratio -= self._linear_variance_decay
        
        if self._variance_ratio <= 0:
            # The variance has reached zero, no need to process "add_noise" any more
            self._addons.pop("add_noise")
            return

        std_ratio = math.sqrt(self._variance_ratio)
        for group in self.local_optimizer.param_groups:
            if self.should_process_group(group):
                for param in group["params"]:
                    device = param.data.get_device()
                    if self._noise_type == "gaussian":
                        param.grad.data += (
                            self._std * std_ratio
                            * torch.randn(param.grad.data.shape, device=device)
                        )
                    elif self._noise_type == "multiplicative_gaussian":
                        param.grad.data *= (
                            1 + self._std * std_ratio
                            * torch.randn(param.grad.data.shape, device=device)
                        )
                    else:
                        raise ValueError("noise type {} not supported".format(self._noise_type))

    # Private
    # Perform the slow momentum algorithm
    def slow_momentum(self):
        # Perform the slow momentum for every $inner_loop_steps steps
        if self._step_counter % self._inner_loop_steps != 0:
            return
        
        index = 0
        self.average_weights()
        for group in self.local_optimizer.param_groups:
            if self.should_process_group(group):
                learning_rate = group['lr']
                for param in group['params']:
                    # Update the momentum
                    self._momentum_buffer[index] = (
                        self._slow_momentum_factor * self._momentum_buffer[index]
                        + (self._previous_param_data[index] - param.data) / learning_rate
                    )
                    # Update the parameters
                    param.data = (
                        self._previous_param_data[index]
                        - self._slow_learning_rate * learning_rate * self._momentum_buffer[index]
                    )
                    # Memorize the current parameters
                    self._previous_param_data[index] = param.data.detach().clone()
                    index += 1

    # Private
    def load_value_from_dict(self, key, dict, default_value=None):
        if key in dict:
            return dict[key]
        elif default_value is not None:
            return default_value
        else:
            raise ValueError("\'{}\' is not specified in \'arg_dict\'".format(key))

    # Private
    def should_process_group(self, group):
        return (
            "allow_data_parallelism" not in group
            or group["allow_data_parallelism"]
        )

    # Private
    def average_weights(self, all_reduce_group=None, size=None):
        if self._state['synced'] or self._state['disable_synchronization']:
            return

        if size is None:
            size = self._world_size
        if all_reduce_group is None:
            all_reduce_group = self._process_group

        for group in self.local_optimizer.param_groups:
            if self.should_process_group(group):
                for param in group['params']:
                    dist.all_reduce(
                        param.data,
                        op=dist.ReduceOp.SUM,
                        group=all_reduce_group
                    )
                    param.data /= size

        if size == self._world_size:
            self._state['synced'] = True

    # Private
    # Broadcast the weights from src to all other processes in broadcast_group
    def broadcast_weights(self, src=None, broadcast_group=None):
        if self._state['synced'] or self._state['disable_synchronization']:
            return

        if src is None:
            if self._process_group is None:
                src = 0
            elif self._leader is not None:
                src = self._leader
            else:
                raise ValueError('src {} unrecognized'.format(src))

        if broadcast_group is None:
            broadcast_group = self._process_group

        for group in self.local_optimizer.param_groups:
            if self.should_process_group(group):
                for param in group['params']:
                    dist.broadcast(param.data, src, broadcast_group)

        if broadcast_group == self._process_group:
            self._state['synced'] = True


# Implement the (post-)local SGD algrithm
class PostLocalSGD(DistributedOptimizer):
    # Private constant
    # Set default values to minimize the chance of raising an error
    _default_local_steps = 8
    _default_initial_steps = 0
    _default_initial_step_method = 'single_process'

    # Public constructor
    def __init__(self, local_optimizer, rank, world_size, group, arg_dict):
        super().__init__(local_optimizer, rank, world_size, group, arg_dict)

        self._local_steps = self.load_value_from_dict(
            'local_steps', arg_dict, default_value=self._default_local_steps
        )
        self._initial_steps = self.load_value_from_dict(
            'initial_steps', arg_dict,
            default_value=self._default_initial_steps
        )
        self._initial_step_method = self.load_value_from_dict(
            'initial_step_method', arg_dict,
            default_value=self._default_initial_step_method
        )

        self._leader = 0 if group is None else group[0]

        if self._initial_step_method == 'single_process' \
                and self._initial_steps > 0:
            self._state['disable_synchronization'] = True

    # Override
    """
    In the previous version, only one process will step during the initial
    steps, which has been shown to be not good when initial_steps is large.
    We changed the logic, i.e., in the initial steps, all processes will
    step but after these steps and the broadcast, only the parameters for
    the embedding lookups will be kept. The new version has better performance.
    """
    @torch.no_grad()
    def step(self):
        super().step()
        
        if self._initial_step_method == 'multiple_processes':
            if self._step_counter <= self._initial_steps \
                    or self._step_counter % self._local_steps == 0:
                self.average_weights()

        elif self._initial_step_method == 'single_process':
            if self._step_counter == self._initial_steps:
                self._state['disable_synchronization'] = False
                self.broadcast_weights(self._leader)

            if self._step_counter > self._initial_steps \
                    and self._step_counter % self._local_steps == 0:
                self.average_weights()

        else:
            raise ValueError('initial step method {} not specified or supported, \
                supported methods include \'single_process\' and \'multiple_processes\''
                .format(self._initial_step_method))

    # Override
    # Current version of state_dict() does not support addons
    def state_dict(self):
        dict = super().state_dict()
        dict['local_steps'] = self._local_steps
        dict['initial_steps'] = self._initial_steps
        dict['initial_step_method'] = self._initial_step_method
        return dict

    # Override
    # Current version of load_state_dict() does not support addons
    def load_state_dict(self, dict):
        super().load_state_dict(dict)
        self._local_steps = dict['local_steps']
        self._initial_steps = dict['initial_steps']
        self._initial_step_method = dict['initial_step_method']


# Implement the hierarchical local SGD algrithm
class HierarchicalLocalSGD(DistributedOptimizer):
    # Private constant
    # Set default values to minimize the chance of raising an error
    _default_local_sync_freq = 4
    _default_global_sync_freq = 4

    # Public constructor
    def __init__(self, local_optimizer, rank, world_size, group, arg_dict):
        super().__init__(local_optimizer, rank, world_size, group, arg_dict)

        self._local_sync_freq = self.load_value_from_dict(
            'local_sync_freq', arg_dict,
            default_value=self._default_local_sync_freq
        )
        self._global_sync_freq = self.load_value_from_dict(
            'global_sync_freq', arg_dict,
            default_value=self._default_global_sync_freq
        )
        self._node_size = self.load_value_from_dict('node_size', arg_dict)
        self._node_id = self.load_value_from_dict('node_id', arg_dict)

        self._num_nodes = self._world_size // self._node_size
        if self._num_nodes * self._node_size != self._world_size:
            raise ValueError('world_size cannot be divided by node_size')

        self.set_node_process_groups()

        self._sync_counter = 0

    # Private
    # Prepare node_id, node_size, and num_nodes before calling this method
    def set_node_process_groups(self):
        for i in range(self._num_nodes):
            node_process_group = dist.new_group(
                ranks=list(range(i * self._node_size, (1 + i) * self._node_size)))
            if i == self._node_id:
                self._node_process_group = node_process_group

    # Override
    @torch.no_grad()
    def step(self):
        super().step()

        if self._step_counter % self._local_sync_freq == 0:
            if self._sync_counter % self._global_sync_freq == 0:
                self.average_weights(self._process_group)
            else:
                self.average_weights(self._node_process_group, self._node_size)
            self._sync_counter += 1

    # Override
    def state_dict(self):
        dict = super().state_dict()
        dict['local_sync_freq'] = self._local_sync_freq
        dict['global_sync_freq'] = self._global_sync_freq
        dict['node_size'] = self._node_size
        dict['node_id'] = self._node_id
        dict['num_nodes'] = self._num_nodes
        dict['sync_counter'] = self._sync_counter
        return dict

    # Override
    def load_state_dict(self, dict):
        super().load_state_dict(dict)
        self._local_sync_freq = dict['local_sync_freq']
        self._global_sync_freq = dict['global_sync_freq']
        self._node_size = dict['node_size']
        self._node_id = dict['node_id']
        self._num_nodes = dict['num_nodes']
        self._sync_counter = dict['sync_counter']
        self.set_node_process_groups()

