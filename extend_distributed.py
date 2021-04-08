# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
import builtins
import os
import sys

import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.autograd.profiler import record_function
from torch.nn.parallel import DistributedDataParallel as DDP


try:
    import torch_ccl
except ImportError as e:
    # print(e)
    torch_ccl = False

try:
    import torch_ucc
except ImportError as e:
    torch_ucc = False


my_rank = -1
my_size = -1
my_local_rank = -1
my_local_size = -1
alltoall_supported = False
a2a_impl = os.environ.get("DLRM_ALLTOALL_IMPL", "")

myreq = None


def env2int(env_list, default=-1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0:
            return val
    return default


def get_my_slice(n):
    k, m = divmod(n, my_size)
    return slice(
        my_rank * k + min(my_rank, m), (my_rank + 1) * k + min(my_rank + 1, m), 1
    )


def get_split_lengths(n):
    k, m = divmod(n, my_size)
    if m == 0:
        splits = None
        my_len = k
    else:
        splits = [(k + 1) if i < m else k for i in range(my_size)]
        my_len = splits[my_rank]
    return (my_len, splits)


def init_distributed(rank=-1, local_rank=-1, size=-1, use_gpu=False, backend=""):
    global myreq
    global my_rank
    global my_size
    global my_local_rank
    global my_local_size
    global a2a_impl
    global alltoall_supported

    # guess MPI ranks from env (works for IMPI, OMPI and MVAPICH2)
    num_mpi_ranks = env2int(
        ["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE", "WORLD_SIZE"]
    )
    if backend == "" and num_mpi_ranks > 1:
        if torch_ccl and env2int(["CCL_WORKER_COUNT"]) > 0:
            backend = "ccl"
        elif use_gpu and dist.is_nccl_available():
            backend = "nccl"
        elif dist.is_mpi_available():
            backend = "mpi"
        else:
            print(
                "WARNING: MPI multi-process launch detected but PyTorch MPI backend not available."
            )
            backend = "gloo"

    if backend != "":
        # guess Rank and size
        if rank == -1:
            rank = env2int(
                ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK", "RANK"], 0
            )
        if size == -1:
            size = env2int(
                [
                    "PMI_SIZE",
                    "OMPI_COMM_WORLD_SIZE",
                    "MV2_COMM_WORLD_SIZE",
                    "WORLD_SIZE",
                ],
                1,
            )
        if not os.environ.get("RANK", None) and rank != -1:
            os.environ["RANK"] = str(rank)
        if not os.environ.get("WORLD_SIZE", None) and size != -1:
            os.environ["WORLD_SIZE"] = str(size)
        if not os.environ.get("MASTER_PORT", None):
            os.environ["MASTER_PORT"] = "29500"
        if not os.environ.get("MASTER_ADDR", None):
            local_size = env2int(
                [
                    "MPI_LOCALNRANKS",
                    "OMPI_COMM_WORLD_LOCAL_SIZE",
                    "MV2_COMM_WORLD_LOCAL_SIZE",
                ],
                1,
            )
            if local_size != size and backend != "mpi":
                print(
                    "Warning: Looks like distributed multinode run but MASTER_ADDR env not set, using '127.0.0.1' as default"
                )
                print(
                    "If this run hangs, try exporting rank 0's hostname as MASTER_ADDR"
                )
            os.environ["MASTER_ADDR"] = "127.0.0.1"

    if size > 1:
        if local_rank == -1:
            my_local_rank = env2int(
                [
                    "MPI_LOCALRANKID",
                    "OMPI_COMM_WORLD_LOCAL_RANK",
                    "MV2_COMM_WORLD_LOCAL_RANK",
                    "LOCAL_RANK",
                ],
                0,
            )
        else:
            my_local_rank = local_rank
        my_local_size = env2int(
            [
                "MPI_LOCALNRANKS",
                "OMPI_COMM_WORLD_LOCAL_SIZE",
                "MV2_COMM_WORLD_LOCAL_SIZE",
            ],
            1,
        )
        if use_gpu:
            if my_local_size > torch.cuda.device_count():
                print(
                    "Not sufficient GPUs available... local_size = %d, ngpus = %d"
                    % (my_local_size, torch.cuda.device_count())
                )
                sys.exit(1)
            torch.cuda.set_device(my_local_rank)
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        if my_rank == 0:
            print("Running on %d ranks using %s backend" % (my_size, backend))
        if hasattr(dist, "all_to_all_single"):
            try:
                t = torch.zeros([4])
                if use_gpu:
                    t = t.cuda()
                dist.all_to_all_single(t, t)
                alltoall_supported = True
            except RuntimeError as err:
                print("fail to enable all_to_all_single primitive: %s" % err)
        if a2a_impl == "alltoall" and alltoall_supported == False:
            print(
                "Requested DLRM_ALLTOALL_IMPL=%s but backend %s does not support it, use scatter/gather based alltoall"
                % (a2a_impl, backend)
            )
            a2a_impl = "scatter"
        if a2a_impl != "":
            print("Using DLRM_ALLTOALL_IMPL=%s" % a2a_impl)
    else:
        my_rank = 0
        my_size = 1
        my_local_rank = 0
        my_local_size = 1
    print_all(
        "world size: %d, current rank: %d, local rank: %d"
        % (my_size, my_rank, my_local_rank)
    )
    myreq = Request()


class Request(object):
    def __init__(self):
        self.req = None
        self.tensor = None
        self.WaitFunction = All2All_Scatter_Wait

    def wait(self):
        ret = self.WaitFunction.apply(*self.tensor)
        self.req = None
        self.tensor = None
        return ret


class All2All_ScatterList_Req(Function):
    @staticmethod
    def forward(ctx, a2a_info, *inputs):
        global myreq
        batch_split_lengths = (
            a2a_info.global_batch_partition_slices
            if a2a_info.global_batch_partition_slices
            else a2a_info.local_batch_num
        )
        table_split_lengths = (
            a2a_info.global_table_wise_parition_slices
            if a2a_info.global_table_wise_parition_slices
            else [a2a_info.local_table_num] * my_size
        )
        gather_list = []
        req_list = []
        for i in range(my_size):
            for j in range(table_split_lengths[i]):
                out_tensor = inputs[0].new_empty(
                    [a2a_info.local_batch_num, a2a_info.emb_dim]
                )
                scatter_list = (
                    list(inputs[j].split(batch_split_lengths, dim=0))
                    if i == my_rank
                    else []
                )
                req = dist.scatter(out_tensor, scatter_list, src=i, async_op=True)
                gather_list.append(out_tensor)
                req_list.append(req)
        myreq.req = req_list
        myreq.tensor = tuple(gather_list)
        myreq.a2a_info = a2a_info
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        for r in myreq.req:
            r.wait()
        myreq.req = None
        grad_inputs = myreq.tensor
        myreq.tensor = None
        return (None, *grad_inputs)


class All2All_ScatterList_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        ctx.a2a_info = myreq.a2a_info
        for r in myreq.req:
            r.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        a2a_info = ctx.a2a_info
        grad_output = [t.contiguous() for t in grad_output]
        batch_split_lengths = (
            a2a_info.global_batch_partition_slices
            if a2a_info.global_batch_partition_slices
            else [a2a_info.local_batch_num] * my_size
        )
        per_rank_table_splits = (
            a2a_info.global_table_wise_parition_slices
            if a2a_info.global_table_wise_parition_slices
            else [a2a_info.local_table_num] * my_size
        )
        grad_inputs = [
            grad_output[0].new_empty([ctx.a2a_info.batch_size, ctx.a2a_info.emb_dim])
            for _ in range(a2a_info.local_table_num)
        ]
        req_list = []
        ind = 0
        for i in range(my_size):
            for j in range(per_rank_table_splits[i]):
                gather_list = (
                    list(grad_inputs[j].split(batch_split_lengths, dim=0))
                    if i == my_rank
                    else None
                )
                req = dist.gather(grad_output[ind], gather_list, dst=i, async_op=True)
                req_list.append(req)
                ind += 1
        myreq.req = req_list
        myreq.tensor = grad_inputs
        return tuple(grad_output)


class All2All_Scatter_Req(Function):
    @staticmethod
    def forward(ctx, a2a_info, *inputs):
        global myreq
        batch_split_lengths = (
            a2a_info.global_batch_partition_slices
            if a2a_info.global_batch_partition_slices
            else a2a_info.local_batch_num
        )
        table_split_lengths = (
            a2a_info.global_table_wise_parition_slices
            if a2a_info.global_table_wise_parition_slices
            else [a2a_info.local_table_num] * my_size
        )
        input = torch.cat(inputs, dim=1)
        scatter_list = list(input.split(batch_split_lengths, dim=0))
        gather_list = []
        req_list = []
        for i in range(my_size):
            out_tensor = input.new_empty(
                [a2a_info.local_batch_num, table_split_lengths[i] * a2a_info.emb_dim]
            )
            req = dist.scatter(
                out_tensor, scatter_list if i == my_rank else [], src=i, async_op=True
            )
            gather_list.append(out_tensor)
            req_list.append(req)
        myreq.req = req_list
        myreq.tensor = tuple(gather_list)
        myreq.a2a_info = a2a_info
        ctx.a2a_info = a2a_info
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        for r in myreq.req:
            r.wait()
        myreq.req = None
        grad_input = myreq.tensor
        grad_inputs = grad_input.split(ctx.a2a_info.emb_dim, dim=1)
        myreq.tensor = None
        return (None, *grad_inputs)


class All2All_Scatter_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        ctx.a2a_info = myreq.a2a_info
        for r in myreq.req:
            r.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        assert len(grad_output) == my_size
        scatter_list = [t.contiguous() for t in grad_output]
        a2a_info = ctx.a2a_info
        batch_split_lengths = (
            a2a_info.global_batch_partition_slices
            if a2a_info.global_batch_partition_slices
            else a2a_info.local_batch_num
        )
        table_split_lengths = (
            a2a_info.global_table_wise_parition_slices
            if a2a_info.global_table_wise_parition_slices
            else [a2a_info.local_table_num] * my_size
        )
        grad_input = grad_output[0].new_empty(
            [a2a_info.batch_size, a2a_info.emb_dim * a2a_info.local_table_num]
        )
        gather_list = list(grad_input.split(batch_split_lengths, dim=0))
        req_list = []
        for i in range(my_size):
            req = dist.gather(
                scatter_list[i],
                gather_list if i == my_rank else [],
                dst=i,
                async_op=True,
            )
            req_list.append(req)
        myreq.req = req_list
        myreq.tensor = grad_input
        return grad_output


class All2All_Req(Function):
    @staticmethod
    def forward(ctx, a2a_info, *inputs):
        global myreq
        with record_function("DLRM alltoall_req_fwd_single"):
            batch_split_lengths = a2a_info.global_batch_partition_slices
            if batch_split_lengths:
                batch_split_lengths = [
                    m * a2a_info.emb_dim * a2a_info.local_table_num
                    for m in batch_split_lengths
                ]
            table_split_lengths = a2a_info.global_table_wise_parition_slices
            if table_split_lengths:
                table_split_lengths = [
                    a2a_info.local_batch_num * e * a2a_info.emb_dim
                    for e in table_split_lengths
                ]
            input = torch.cat(inputs, dim=1).view([-1])
            output = input.new_empty(
                [
                    a2a_info.global_table_num
                    * a2a_info.local_batch_num
                    * a2a_info.emb_dim
                ]
            )
            req = dist.all_to_all_single(
                output, input, table_split_lengths, batch_split_lengths, async_op=True
            )

            myreq.req = req
            myreq.tensor = []
            myreq.tensor.append(output)
            myreq.tensor = tuple(myreq.tensor)
            a2a_info.batch_split_lengths = batch_split_lengths
            a2a_info.table_split_lengths = table_split_lengths
            myreq.a2a_info = a2a_info
            ctx.a2a_info = a2a_info
            return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        with record_function("DLRM alltoall_req_bwd_single"):
            a2a_info = ctx.a2a_info
            myreq.req.wait()
            myreq.req = None
            grad_input = myreq.tensor
            grad_inputs = grad_input.view([a2a_info.batch_size, -1]).split(
                a2a_info.emb_dim, dim=1
            )
            grad_inputs = [gin.contiguous() for gin in grad_inputs]
            myreq.tensor = None
            return (None, *grad_inputs)


class All2All_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        with record_function("DLRM alltoall_wait_fwd_single"):
            a2a_info = myreq.a2a_info
            ctx.a2a_info = a2a_info
            myreq.req.wait()
            myreq.req = None
            myreq.tensor = None
            table_split_lengths = (
                a2a_info.table_split_lengths
                if a2a_info.table_split_lengths
                else a2a_info.local_table_num
                * a2a_info.local_batch_num
                * a2a_info.emb_dim
            )
            outputs = output[0].split(table_split_lengths)
            outputs = tuple(
                [out.view([a2a_info.local_batch_num, -1]) for out in outputs]
            )
            return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        global myreq
        with record_function("DLRM alltoall_wait_bwd_single"):
            a2a_info = ctx.a2a_info
            grad_outputs = [gout.contiguous().view([-1]) for gout in grad_outputs]
            grad_output = torch.cat(grad_outputs)
            grad_input = grad_output.new_empty(
                [a2a_info.batch_size * a2a_info.local_table_num * a2a_info.emb_dim]
            )
            req = dist.all_to_all_single(
                grad_input,
                grad_output,
                a2a_info.batch_split_lengths,
                a2a_info.table_split_lengths,
                async_op=True,
            )
            myreq.req = req
            myreq.tensor = grad_input
            return (grad_output,)


class AllGather(Function):
    @staticmethod
    def forward(ctx, input, global_lengths, dim=0):
        if not isinstance(global_lengths, (list, tuple)):
            global_lengths = [global_lengths] * my_size

        assert len(global_lengths) == my_size
        assert global_lengths[my_rank] == input.size(dim)
        local_start = sum(global_lengths[:my_rank])

        output_size = list(input.size())

        ctx.dim = dim
        ctx.local_start = local_start
        ctx.local_length = global_lengths[my_rank]

        input = input.contiguous()
        if dim == 0:
            out_len = sum(global_lengths)
            output_size[dim] = out_len
            output = input.new_empty(output_size)
            gather_list = list(output.split(global_lengths, dim=0))
        else:
            gather_list = [torch.empty_like(input) for _ in range(my_size)]
            gather_list = []
            for length in global_lengths:
                output_size[dim] = length
                gather_list.append(input.new_empty(output_size))

        dist.all_gather(gather_list, input)

        if dim != 0:
            output = torch.cat(gather_list, dim=dim)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # print("Inside All2AllBackward")
        dim = ctx.dim
        start = ctx.local_start
        length = ctx.local_length

        grad_input = grad_output.narrow(dim, start, length)

        return (grad_input, None, None)


class All2AllInfo(object):
    pass


def alltoall(inputs, per_rank_table_splits):
    global myreq
    batch_size, emb_dim = inputs[0].size()
    a2a_info = All2AllInfo()
    a2a_info.local_table_num = len(inputs)
    a2a_info.global_table_wise_parition_slices = per_rank_table_splits
    (
        a2a_info.local_batch_num,
        a2a_info.global_batch_partition_slices,
    ) = get_split_lengths(batch_size)
    a2a_info.emb_dim = emb_dim
    a2a_info.batch_size = batch_size
    a2a_info.global_table_num = (
        sum(per_rank_table_splits)
        if per_rank_table_splits
        else a2a_info.local_table_num * my_size
    )

    if a2a_impl == "" and alltoall_supported or a2a_impl == "alltoall":
        # print("Using All2All_Req")
        output = All2All_Req.apply(a2a_info, *inputs)
        myreq.WaitFunction = All2All_Wait
    elif a2a_impl == "" or a2a_impl == "scatter":
        # print("Using All2All_Scatter_Req")
        output = All2All_Scatter_Req.apply(a2a_info, *inputs)
        myreq.WaitFunction = All2All_Scatter_Wait
    elif a2a_impl == "scatter_list":
        # print("Using All2All_ScatterList_Req")
        output = All2All_ScatterList_Req.apply(a2a_info, *inputs)
        myreq.WaitFunction = All2All_ScatterList_Wait
    else:
        print(
            "Unknown value set for DLRM_ALLTOALL_IMPL (%s), "
            "please use one of [alltoall, scatter, scatter_list]" % a2a_impl
        )
    return myreq


def all_gather(input, lengths, dim=0):
    if not lengths:
        lengths = [input.size(0)] * my_size
    return AllGather.apply(input, lengths, dim)


def barrier():
    if my_size > 1:
        dist.barrier()


# Override builtin print function to print only from rank 0
orig_print = builtins.print


def rank0_print(*args, **kwargs):
    if my_rank <= 0 or kwargs.get("print_all", False):
        orig_print(*args, **kwargs)


builtins.print = rank0_print

# Allow printing from all rank with explicit print_all
def print_all(*args, **kwargs):
    orig_print(*args, **kwargs)
