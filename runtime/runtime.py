# Copyright (c) Microsoft Corporation.
# Licensed under the MIT license.

import collections
import itertools
import time
import torch
import torch.distributed as dist

import communication
import runtime_utilities

IMAGE_CLASSIFICATION = "image_classification"
TRANSLATION = "translation"
SPEECH_TO_TEXT = "speech_to_text"
RECOMMENDATION = "recommendation"


class ModulesWithDependencies:
    def __init__(self, modules_with_dependencies):
        self._modules = []
        self._all_input_names = []
        self._all_output_names = []
        for (module, input_names, output_names) in modules_with_dependencies:
            self._modules.append(module)
            self._all_input_names.append(input_names)
            self._all_output_names.append(output_names)

    def modules(self):
        return self._modules

    def all_input_names(self):
        return self._all_input_names

    def all_output_names(self):
        return self._all_output_names

    def is_input_tensor(self, tensor_name):
        for module_input_names in self._all_input_names:
            if tensor_name in module_input_names:
                return True
        return False


class StageRuntime:
    def __init__(self, model, distributed_backend, fp16, loss_scale,
                 training_tensor_shapes, eval_tensor_shapes,
                 training_tensor_dtypes, inputs_module_destinations,
                 target_tensor_names, configuration_maps, master_addr,
                 rank, local_rank, num_ranks_in_server, verbose_freq,
                 model_type, enable_recompute=False):
        # Metadata needed for forward and backward pass within this stage.
        self.tensors = []
        self.gradients = {}
        self.distributed_backend = distributed_backend
        self.fp16 = fp16
        self.loss_scale = loss_scale
        self.training_tensor_shapes = training_tensor_shapes
        self.eval_tensor_shapes = eval_tensor_shapes
        self.training_tensor_dtypes = training_tensor_dtypes
        self.model_type = model_type
        self.target_tensor_names = target_tensor_names

        # print("enter initialize")
        # import sys; sys.stdout.flush()

        self.initialize(model, inputs_module_destinations, configuration_maps,
                        master_addr, rank, local_rank, num_ranks_in_server)
        
        # print("leave initialize")
        # import sys; sys.stdout.flush()

        self.verbose_freq = verbose_freq
        self.forward_only = False

        self.forward_stats = runtime_utilities.RuntimeStats(forward=True)
        self.backward_stats = runtime_utilities.RuntimeStats(forward=False)

        # Enable recomputation to prevent the need to save activations
        # computed from the forward pass for the backward pass.
        self.enable_recompute = enable_recompute

        # Disable recomputation for the last stage.
        if rank == num_ranks_in_server - 1:
            self.enable_recompute = False
        
        # for debug
        torch.autograd.set_detect_anomaly(True)


    def initialize(self, model, inputs_module_destinations,
                   configuration_maps, master_addr, rank,
                   local_rank, num_ranks_in_server):
        self.send_ranks = {}
        self.receive_ranks = {}
        self.rank = rank
        self.local_rank = local_rank
        self.stage = None
        self.tensor_tags = {}
        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0
        self.criterion_input_name = str(model[-1][1][0])

        tensor_tag = 1
        for (_, input_tensors, output_tensors) in model:
            for input_tensor in input_tensors:
                if input_tensor not in self.tensor_tags:
                    self.tensor_tags[input_tensor] = tensor_tag
                    tensor_tag += 1
            for output_tensor in output_tensors:
                if output_tensor not in self.tensor_tags:
                    self.tensor_tags[output_tensor] = tensor_tag
                    tensor_tag += 1
        for target_tensor_name in sorted(self.target_tensor_names):
            self.tensor_tags[target_tensor_name] = tensor_tag
            tensor_tag += 1
        self.tensor_tags["ack"] = tensor_tag
        tensor_tag += 1

        module_to_stage_map = configuration_maps['module_to_stage_map']
        stage_to_rank_map = configuration_maps['stage_to_rank_map']
        stage_to_depth_map = configuration_maps['stage_to_depth_map']
        # pipleline debug
        #print("module_to_state_map: ", module_to_stage_map)

        if module_to_stage_map is None:
            # If IP addresses not specified, resort to all layers on
            # single machine.
            assert self.rank is None
            self.modules_with_dependencies = ModulesWithDependencies(model)
            self.is_criterion = True
            self.rank_in_stage = 0
            self.num_ranks = 1
            self.num_ranks_in_first_stage = 1
            self.num_ranks_in_previous_stage = 0
            self.num_ranks_in_next_stage = 0
            self.num_stages = 1
            self.num_ranks_in_stage = 1
            self.num_warmup_minibatches = 0
            self.comm_handler = None
        else:
            assert len(module_to_stage_map) == len(model)
            assert self.rank is not None

            stage_to_module_map = collections.defaultdict(list)
            for module in range(len(module_to_stage_map)):
                stage_to_module_map[module_to_stage_map[module]].append(module)

            rank_to_stage_map = {}
            for stage in stage_to_rank_map:
                for rank in stage_to_rank_map[stage]:
                    rank_to_stage_map[rank] = stage

            # Now, use this mapping to determine the modules contained in
            # each stage.
            assert 0 <= self.rank < len(rank_to_stage_map)
            self.num_ranks = len(rank_to_stage_map)
            self.num_stages = len(stage_to_module_map)
            #print("num_stages: ", self.num_stages)
            self.stage = rank_to_stage_map[self.rank]
            #print("my stage: ", self.stage)
            self.rank_in_stage = stage_to_rank_map[self.stage].index(self.rank)
            self.num_ranks_in_stage = len(stage_to_rank_map[self.stage])
            self.num_ranks_in_first_stage = len(stage_to_rank_map[0])
            self.num_ranks_in_previous_stage = 0
            self.ranks_in_previous_stage = []
            if self.stage > 0:
                self.num_ranks_in_previous_stage = len(
                    stage_to_rank_map[self.stage - 1])
                self.ranks_in_previous_stage = stage_to_rank_map[self.stage - 1]
            self.num_ranks_in_next_stage = 0
            self.ranks_in_next_stage = []
            if self.stage < self.num_stages - 1:
                self.num_ranks_in_next_stage = len(
                    stage_to_rank_map[self.stage + 1])
                self.ranks_in_next_stage = stage_to_rank_map[self.stage + 1]
            modules = stage_to_module_map[self.stage]
            self.modules_with_dependencies = ModulesWithDependencies(
                [model[module] for module in modules])
            self.is_criterion = self.stage == (self.num_stages - 1)
            if stage_to_depth_map is not None:
                self.num_warmup_minibatches = stage_to_depth_map[
                    str(self.stage)]
            else:
                self.num_warmup_minibatches = self.num_ranks - 1
                for i in range(self.stage):
                    self.num_warmup_minibatches -= len(
                        stage_to_rank_map[i])
                self.num_warmup_minibatches = self.num_warmup_minibatches // \
                    self.num_ranks_in_stage

            # To determine where tensors should be sent and received, first
            # determine the "producing" and "consuming" module IDs of each
            # tensor. We then use the corresponding machine ranks to send
            # and receive tensors.
            master_port = 12345
            self.comm_handler = communication.CommunicationHandler(
                master_addr=master_addr,
                master_port=master_port,
                rank=self.rank,
                local_rank=self.local_rank,
                num_ranks_in_server=num_ranks_in_server,
                world_size=self.num_ranks,
                fp16=self.fp16,
                backend=self.distributed_backend)

            for i in range(len(model)):
                for j in range(i+1, len(model)):
                    for tensor_name in model[i][2]: # module output
                        if tensor_name in model[j][1]: # module input
                            if module_to_stage_map[i] == \
                                module_to_stage_map[j]:
                                continue
                            # For now, assume that each stage is served by only
                            # a single machine.
                            if module_to_stage_map[j] == self.stage:
                                #print("receive ", tensor_name, " from rank ", 
                                #      stage_to_rank_map[module_to_stage_map[i]])
                                self.receive_ranks[tensor_name] = \
                                    stage_to_rank_map[module_to_stage_map[i]]
                            if module_to_stage_map[i] == self.stage:
                                #print("send ", tensor_name, " to rank ", 
                                #      stage_to_rank_map[module_to_stage_map[j]])
                                self.send_ranks[tensor_name] = \
                                    stage_to_rank_map[module_to_stage_map[j]]
            
            # handle modle inputs (passed through the pipeline)
            for model_inputs in inputs_module_destinations.keys():
                destination_stage = module_to_stage_map[
                    inputs_module_destinations[model_inputs]]
                if destination_stage > self.stage:
                    self.send_ranks[model_inputs] = \
                        self.ranks_in_next_stage

                if 0 < self.stage <= destination_stage:
                    self.receive_ranks[model_inputs] = \
                        self.ranks_in_previous_stage

                if destination_stage > 0:
                    if model_inputs not in self.tensor_tags:
                        self.tensor_tags[model_inputs] = tensor_tag
                        tensor_tag += 1

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            #modules[i] = modules[i].cuda() # TODO: avoid duplicated modules on GPUs.
            if self.fp16:
                import apex.fp16_utils as fp16_utils
                modules[i] = fp16_utils.BN_convert_float(modules[i].half())

        # Initialize all groups in the same order on every worker.
        if stage_to_rank_map is not None:
            groups = []
            for stage in range(self.num_stages):
                ranks = stage_to_rank_map[stage]
                if len(ranks) > 1:
                    groups.append(dist.new_group(ranks=ranks))
                else:
                    groups.append(None)
            group = groups[self.stage]
        else:
            group = None

        # self.modules_with_dependencies contains a list of PyTorch
        # modules, along with a list of user-defined input and output
        # tensor names. We use our module_executor.ModuleExecutor
        # class to wrap these dependencies, and use run_forward and
        # run_backward methods downstream.
        num_parameters = 0
        for i in range(len(modules)):
            if group is not None:
                if ((i < (len(modules)-1) and self.is_criterion)
                    or not self.is_criterion):
                    num_parameters += \
                        sum(x.size()[0] * x.size()[1]
                            if len(x.size()) > 1 else x.size()[0]
                            for x in modules[i].parameters() if x.size())
                    modules[i] = torch.nn.parallel.DistributedDataParallel(
                        modules[i],
                        process_group=group,
                        device_ids=[local_rank],
                        output_device=local_rank)
        if self.num_ranks_in_stage > 1:
            module_size = 4. * num_parameters
            print("Replicating stage: ranks=%d, module_size=%.3f" % (
                self.num_ranks_in_stage, module_size))

        if self.fp16:
            self.master_parameters = []
            self.model_parameters = []
            for i in range(len(modules)):
                import apex.fp16_utils as fp16_utils
                module_parameters, module_master_parameters = \
                    fp16_utils.prep_param_lists(modules[i])
                self.master_parameters.extend(module_master_parameters)
                self.model_parameters.extend(module_parameters)
        else:
            self.master_parameters = list(self.parameters())
            self.model_parameters = None

        if self.comm_handler is not None:
            self.comm_handler.initialize(
                self.receive_ranks,
                self.send_ranks,
                self.tensor_tags,
                self.target_tensor_names,
                self.training_tensor_dtypes,
                self.rank_in_stage,
                self.num_ranks_in_stage,
                self.ranks_in_previous_stage,
                self.ranks_in_next_stage)

    @property
    def target(self):
        return self.tensors[-1]["target"]

    def modules(self):
        return self.modules_with_dependencies.modules()

    def parameters(self):
        parameter_iterators = []
        for module in self.modules_with_dependencies.modules():
            parameter_iterators.append(module.parameters())
        return itertools.chain(*parameter_iterators)

    def state_dict(self):
        state_dict = collections.OrderedDict()
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            state_dict["module%d" % i] = module.state_dict()
        if self.fp16:
            state_dict["master_parameters"] = self.master_parameters
        return state_dict

    def load_state_dict(self, state_dict):
        for i, module in enumerate(self.modules_with_dependencies.modules()):
            module.load_state_dict(state_dict["module%d" % i])
        if self.fp16:
            saved_master_parameters = state_dict["master_parameters"]
            for master_parameter, saved_master_parameter in zip(
                self.master_parameters, saved_master_parameters):
                master_parameter.data.copy_(saved_master_parameter.data)

    def cuda(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i] = modules[i].cuda()

    def zero_grad(self):
        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].zero_grad()

    def train(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.training_tensor_shapes
        self.forward_only = False

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=False)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].train()

    def eval(self, num_iterations):
        self.tensors = []
        self.gradients = {}
        self.tensor_shapes = self.eval_tensor_shapes
        self.tensor_shapes["ack"] = (1,)
        self.forward_only = True

        self.forward_minibatch_id = 0
        self.backward_minibatch_id = 0

        if self.comm_handler is not None:
            self.comm_handler.set_tensor_shapes(self.tensor_shapes)
            self.comm_handler.start_helper_threads(
                num_iterations, forward_only=True)

        modules = self.modules_with_dependencies.modules()
        for i in range(len(modules)):
            modules[i].eval()

    def set_loader(self, loader):
        if loader is not None:
            self.loader_iter = iter(loader)
        else:
            self.loader_iter = None

    def receive_tensors_forward(self):
        if self.forward_only and len(self.tensors) > 0:
            self.tensors.pop(0)
        self.tensors.append({})
        if self.loader_iter is not None:
            input = next(self.loader_iter)
            if self.model_type == TRANSLATION:
                (input, target) = input
                src, src_length = input
                tgt, tgt_length = target

                self.tensors[-1]["input0"] = src.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = torch.LongTensor(src_length).cuda(
                    non_blocking=True)
                self.tensors[-1]["input2"] = tgt[:-1].cuda(non_blocking=True)
                self.tensors[-1]["target"] = tgt[1:].cuda().contiguous().view(-1)
                self.tensors[-1]["target_length"] = \
                    torch.tensor([int(sum(torch.LongTensor(tgt_length) - 1))],
                                 dtype=torch.int).cuda()
            elif self.model_type == IMAGE_CLASSIFICATION:
                (input, target) = input
                if self.fp16:
                    input = input.half()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
            elif self.model_type == SPEECH_TO_TEXT:
                input, target, input_percentages, target_sizes = input
                input_sizes = input_percentages.mul_(int(input.size(3))).int()
                self.tensors[-1]["input0"] = input.cuda(non_blocking=True)
                self.tensors[-1]["input1"] = input_sizes.cuda(non_blocking=True)
                self.tensors[-1]["target"] = target.cuda(non_blocking=True)
                self.tensors[-1]["target_length"] = target_sizes.cuda(
                    non_blocking=True)
            elif self.model_type == RECOMMENDATION:
                # unpack_batch TODO: consider weights
                #print("input: ", input)
                X, lS_o, lS_i, T = input[0], input[1], input[2], input[3]
                self.tensors[-1]["input0"] = X.cuda(non_blocking=True)
                #print("In runtime, T", T)
                self.tensors[-1]["target"] = T.cuda(non_blocking=True)
                #if isinstance(lS_i, list):
                    # index, offset for input i, input i+1
                assert(len(lS_o) == len(lS_i))
                for k, S_i in enumerate(lS_i):
                    S_o = lS_o[k]
                    self.tensors[-1]["input"+str(2*k+1)] = S_i.cuda(non_blocking=True)
                    self.tensors[-1]["input"+str(2*k+2)] = S_o.cuda(non_blocking=True)
                # else:
                #     raise Exception("Not Implement Yet.")
        else:
            # Receive all required tensors from upstream machines.
            for input_name in self.receive_ranks:
                if input_name == "ack":
                    continue

                self.tensors[-1][input_name] = \
                    self.comm_handler.recv(
                        input_name,
                        forward_minibatch_id=self.forward_minibatch_id,
                        backward_minibatch_id=self.backward_minibatch_id,
                        backward=False)
                #print("receive tensor: ", input_name, self.tensors[-1][input_name])

                self.forward_stats.stats['receive_tensors_size'] += \
                    (self.tensors[-1][input_name].element_size() *
                     self.tensors[-1][input_name].nelement())

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(
                sending=False)

    def send_tensors_forward(self):
        # Send all required tensors downstream.
        for output_name in self.send_ranks:
            if output_name == "ack":
                continue

            self.comm_handler.send(
                output_name,
                self.tensors[-1][output_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=False)
            #print("send tensor: ", output_name, self.tensors[-1][output_name])


            self.forward_stats.stats['send_tensors_size'] += \
                (self.tensors[-1][output_name].element_size() *
                 self.tensors[-1][output_name].nelement())

    def receive_tensors_backward(self):
        # Receive all required gradients from downstream
        # machines.
        for output_name in self.send_ranks:
             if output_name in self.target_tensor_names:
                continue

             self.gradients[output_name] = \
                self.comm_handler.recv(
                    output_name,
                    forward_minibatch_id=self.forward_minibatch_id,
                    backward_minibatch_id=self.backward_minibatch_id,
                    backward=True)

             self.backward_stats.stats['receive_tensors_size'] += \
                 (self.gradients[output_name].element_size() *
                  self.gradients[output_name].nelement())

    def send_tensors_backward(self):
        # Send all required gradients upstream.
        for input_name in self.receive_ranks:
            if input_name in self.target_tensor_names:
                continue

            self.comm_handler.send(
                input_name,
                self.gradients[input_name],
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            self.backward_stats.stats['send_tensors_size'] += \
                (self.gradients[input_name].element_size() *
                 self.gradients[input_name].nelement())
            
            # print("backward send tensor: ", input_name, " size: ", 
            #      (self.gradients[input_name].element_size() *
            #      self.gradients[input_name].nelement()))


        if self.num_ranks_in_previous_stage > 0:
            # Used to track where to send tensors in the
            # backward pass.
            self.comm_handler.increment_messaging_index(
                sending=True)

    def run_forward(self, recompute_step=False):
        """Run forward pass.
        """
        # Receive tensors from previous worker.
        #start = time.time()
        #print("start to receive tensors")
        #import sys; sys.stdout.flush()
        self.receive_tensors_forward()
        #print("receive tensors forward: ", time.time()-start)
        #import sys; sys.stdout.flush()
        tensors = self.tensors[-1]

        # Run forward pass.
        #start = time.time()
        self._run_forward(tensors)
        #print("_run_forward: ", time.time()-start)

        # Send tensors forward.
        #start = time.time()
        self.send_tensors_forward()
        #print("send tensors forward: ", time.time()-start)
        #import sys; sys.stdout.flush()

        if self.verbose_freq > 0 and self.forward_minibatch_id % self.verbose_freq == 0:
            self.forward_stats.print_stats()
        self.forward_stats.reset_stats()
        self.forward_minibatch_id += 1

    def _run_forward(self, tensors):
        # Perform forward pass through model (self.modules_with_dependencies already
        # has modules in topological order).
        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()
        for i, (module, input_names, output_names) in \
                enumerate(zip(modules, all_input_names, all_output_names)):
            if i == (len(modules) - 1) and self.is_criterion:
                # If layer is criterion (loss).
                #print("Enter the criterion (loss layer")
                #print("target tensor: ", tensors["target"])
                if self.model_type == SPEECH_TO_TEXT:
                    output = tensors["output"].transpose(0, 1).float()
                    output_sizes = tensors["output_sizes"].cpu()
                    target = tensors["target"].cpu()
                    target_sizes = tensors["target_length"].cpu()
                    input0_size = tensors["input0_size"].cpu()
                    module_outputs = [module(output, target, output_sizes, target_sizes) / input0_size[0]]
                else:
                    module_outputs = [module(tensors[input_name],
                                             tensors["target"])
                                      for input_name in input_names]
                    module_outputs = [sum(module_outputs)]
            else:
                # If layer is non-criterion.
                module_outputs = module(*[tensors[input_name]
                                          for input_name in input_names])
                if not isinstance(module_outputs, tuple):
                    module_outputs = (module_outputs,)
                module_outputs = list(module_outputs)

            for (output_name, module_output) in zip(output_names, module_outputs):
                tensors[output_name] = module_output

        self.output = tensors[input_names[0]]
        if self.is_criterion and self.model_type == TRANSLATION:
            loss_per_batch = tensors[output_names[0]] * tensors[self.criterion_input_name].size(1)
            loss_per_token = loss_per_batch / tensors["target_length"][0].item()
            self.loss = loss_per_token
        elif self.is_criterion:
            self.loss = tensors[output_names[0]]
        else:
            self.loss = 1

    def run_backward(self):
        # Receive input gradients needed for backward pass.
        #start = time.time()
        self.receive_tensors_backward()
        #print("receive tensors backward: ", time.time()-start)
        #start = time.time()
        # Backward pass through modules in reverse order.
        inputs = {}
        outputs = {}
        input_gradients = {}
        output_gradients = {}

        # Get input and output names spanning all modules in this stage.
        all_input_names_set = set()
        all_output_names_set = set()

        modules = self.modules_with_dependencies.modules()
        all_input_names = self.modules_with_dependencies.all_input_names()
        all_output_names = self.modules_with_dependencies.all_output_names()

        for (input_names, output_names) in zip(all_input_names, all_output_names):
            for input_name in input_names:
                all_input_names_set.add(input_name)
            for output_name in output_names:
                all_output_names_set.add(output_name)

        tensors = self.tensors.pop(0)
        # Set inputs, outputs, and output_gradients.
        # Only set outputs/output_gradients for tensors that are not inputs of
        # other modules in this stage.
        # Similarly, only set inputs for tensors that are not outputs of other
        # modules in this stage.
        for (module, input_names, output_names) in \
            zip(reversed(modules), reversed(all_input_names), reversed(all_output_names)):
            for output_name in output_names:
                if output_name not in all_input_names_set:
                    if output_name not in self.gradients:
                        output_gradients[output_name] = None
                    else:
                        output_gradients[output_name] = self.gradients[output_name]
                    if tensors[output_name].requires_grad:
                        outputs[output_name] = tensors[output_name]
            for input_name in input_names:
                if input_name not in all_output_names_set:
                    inputs[input_name] = tensors[input_name]

        # Hook to record input gradients.
        def hook_wrapper(input_name):
            def hook(input_gradient):
                input_gradients[input_name] = input_gradient
            return hook

        for input_name in inputs:
            if input_name != "input0" and input_name != "input1" and input_name != "input2" \
                    and inputs[input_name].requires_grad:
                inputs[input_name].register_hook(hook_wrapper(input_name))

        if "loss" in outputs:
            outputs["loss"] *= self.loss_scale

        # Perform backward pass.
        torch.autograd.backward(tuple([outputs[output_name] for output_name in outputs]),
                                grad_tensors=tuple([output_gradients[output_name]
                                                    for output_name in outputs]))

        # Input tensors don't need gradients.
        for input_name in inputs:
            if not inputs[input_name].requires_grad:
                self.gradients[input_name] = inputs[input_name]
                continue

            if input_name != "input0" and input_name != "input1" and input_name != "input2" and input_name != "input":
                self.gradients[input_name] = input_gradients[input_name]

        #print("perform backward: ", time.time()-start)
        # Send output gradients.
        #start = time.time()
        self.send_tensors_backward()
        #print("send tensors backward: ", time.time()-start)
        #import sys; sys.stdout.flush()
        if self.verbose_freq > 0 and self.backward_minibatch_id % self.verbose_freq == 0:
            self.backward_stats.print_stats()
        self.backward_stats.reset_stats()
        self.backward_minibatch_id += 1

    def num_tokens(self):
        return self.tensors[-1]["target_length"][0].item()

    def run_ack(self):
        # No need for ack if running on a single worker.
        if self.rank is None:
            return

        # Receive ack from next stage. Send ack to previous stage.
        if self.stage < (self.num_stages-1):
            self.comm_handler.recv(
                "ack",
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)
        if self.stage > 0:
            self.comm_handler.send(
                "ack",
                torch.zeros(self.tensor_shapes["ack"],
                            dtype=torch.int64).cuda(),
                forward_minibatch_id=self.forward_minibatch_id,
                backward_minibatch_id=self.backward_minibatch_id,
                backward=True)

            # Used to track where to receive forward from.
            self.comm_handler.increment_messaging_index(sending=True)

        self.backward_minibatch_id += 1

    def wait(self):
        if self.comm_handler is not None:
            self.comm_handler.wait()

    def num_iterations(self, loader_size):
        """ Determines number of iterations for this stage

        TODO: don't currently support uneven configurations.
        """
        if self.stage == 0 or self.stage is None:
            return loader_size

        num_iterations = loader_size * self.num_ranks_in_first_stage
        assert num_iterations % self.num_ranks_in_stage == 0
        num_iterations = num_iterations // self.num_ranks_in_stage
        #print(self.num_ranks_in_stage, num_iterations)

        return num_iterations

    def get_adjusted_learning_rate(self, base_lr):
        if self.stage == 0:
            return base_lr

        adjusted_lr = float(base_lr) * float(self.num_ranks_in_stage) \
                      / float(self.num_ranks_in_first_stage)

        return adjusted_lr
