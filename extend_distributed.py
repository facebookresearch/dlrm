import logging
import os

import torch
import torch.distributed as dist
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.autograd.profiler import record_function


logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
logger = logging.getLogger(__name__)

my_rank = -1
my_size = -1
my_local_rank = -1
my_local_size = -1

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


def get_slice(r, n):
    k, m = divmod(n, my_size)
    return slice(r * k + min(r, m), (r + 1) * k + min(r + 1, m), 1)


def get_my_slice_sparse(num_emb_per_rank):
    if my_rank == 0:
        return slice(0, num_emb_per_rank[0], 1)
    cum_sum = sum(num_emb_per_rank[0:my_rank])
    return slice(cum_sum, cum_sum + num_emb_per_rank[my_rank], 1)


def get_slice_sparse(r, num_emb_per_rank):
    if r == 0:
        return slice(0, num_emb_per_rank[0], 1)
    cum_sum = sum(num_emb_per_rank[0:r])
    return slice(cum_sum, cum_sum + num_emb_per_rank[r], 1)


def split_is_valid(splits, sparse_param_args):
    for r in range(my_size):
        my_slice = get_slice_sparse(r, splits)
        sizes = [(arg[0] * arg[1] * 4) / 1000000 for arg in sparse_param_args[my_slice]]
        if sum(sizes) > 25000:  # 25 GB
            return False
    return True


def get_split_lengths(sparse_param_args, split_cap_multiplier):
    mine, splits = get_split_lengths_by_len(len(sparse_param_args))
    if split_is_valid(splits, sparse_param_args):
        return (mine, splits)
    else:
        return get_split_lengths_by_size(sparse_param_args, split_cap_multiplier)


def get_split_lengths_by_len(n):
    k, m = divmod(n, my_size)
    if m == 0:
        splits = None
        my_len = k
    else:
        splits = [(k + 1) if i < m else k for i in range(my_size)]
        my_len = splits[my_rank]
    return (my_len, splits)


def get_split_lengths_by_size(sparse_param_args, split_cap_multiplier):
    sizes = [(arg[0] * arg[1] * 4) / 1000000 for arg in sparse_param_args]
    total_size = sum(sizes)
    # This is a HACK to make it fit to 8 GPUs but should still work fine for more than 8 GPU
    size_per_gpu = split_cap_multiplier * total_size / my_size
    assert size_per_gpu < 25000  # more than 25 GB
    logger.info(f"total_size: {total_size/1000}GB, size_per_gpu: {size_per_gpu/1000}GB")
    counts = []
    size_so_far = 0
    count_so_far = 0
    gpu_index = 0
    for emb_size in sizes:
        size_so_far += emb_size
        count_so_far += 1
        if size_so_far > size_per_gpu:
            gpu_index += 1
            counts.append(count_so_far)
            size_so_far = 0
            count_so_far = 0

    if size_so_far > 0:
        counts.append(count_so_far)
    assert len(counts) == my_size, (
        f"number of splits ({len(counts)}) should match number of ranks ({my_size})",
    )
    logger.info(f"counts: {counts}")
    return (counts[my_rank], counts)


def init_distributed(rank=-1, size=-1, backend=""):
    global myreq
    global my_rank
    global my_size
    global my_local_rank
    global my_local_size

    # guess MPI ranks from env (works for IMPI, OMPI and MVAPICH2)
    num_mpi_ranks = env2int(["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"])

    if backend == "nccl":
        logger.info("Using nccl backend")
        # See here for environment variable details:
        # https://pytorch.org/docs/stable/distributed.html#environment-variable-initialization
        # Default setting is eth1 interface on rtptest1550.atn6
        # Needs to be modified based on node that runs rank 0
        if not os.environ.get("MASTER_ADDR", None):
            os.environ["MASTER_ADDR"] = "192.168.1.145"
        if not os.environ.get("MASTER_PORT", None):
            os.environ["MASTER_PORT"] = "29500"
        if rank == -1:
            rank = env2int(
                ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], -1
            )
            if not os.environ.get("RANK", None):
                os.environ["RANK"] = str(rank)
        if size == -1:
            size = env2int(
                ["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], -1
            )
            if not os.environ.get("WORLD_SIZE", None):
                os.environ["WORLD_SIZE"] = str(size)

    if backend == "" and num_mpi_ranks > 1:
        if dist.is_mpi_available():
            backend = "mpi"
        else:
            logger.info(
                "WARNING: MPI multi-process launch detected but PyTorch MPI backend not available."
            )
            backend = "gloo"

    if backend == "gloo":
        if not os.environ.get("MASTER_ADDR", None):
            os.environ["MASTER_ADDR"] = "127.0.0.1"
        if not os.environ.get("MASTER_PORT", None):
            os.environ["MASTER_PORT"] = "29500"
        # logger.info("MASTER_ADDR", os.environ['MASTER_ADDR'])
        # guess Rank and size
        if rank == -1:
            rank = env2int(
                ["PMI_RANK", "OMPI_COMM_WORLD_RANK", "MV2_COMM_WORLD_RANK"], -1
            )
        if size == -1:
            size = env2int(
                ["PMI_SIZE", "OMPI_COMM_WORLD_SIZE", "MV2_COMM_WORLD_SIZE"], -1
            )

    if rank != -1 or size != -1 or backend != "":
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        my_local_rank = env2int(
            [
                "MPI_LOCALRANKID",
                "OMPI_COMM_WORLD_LOCAL_RANK",
                "MV2_COMM_WORLD_LOCAL_RANK",
            ],
            0,
        )
        my_local_size = env2int(
            [
                "MPI_LOCALNRANKS",
                "OMPI_COMM_WORLD_LOCAL_SIZE",
                "MV2_COMM_WORLD_LOCAL_SIZE",
            ],
            1,
        )
        if my_rank == 0:
            logger.info("Running on %d ranks using %s backend" % (my_size, backend))
    else:
        my_rank = 0
        my_size = 1
        my_local_rank = 0
        my_local_size = 1
    myreq = Request()

    # Dummy alltoall for  NCCL warmup
    if backend == "nccl" and hasattr(dist, "all_to_all") and my_size > 1:
        device = torch.device("cuda", my_local_rank)
        input = torch.ones(1024 * my_size, device=device)
        output = torch.zeros(1024 * my_size, device=device)
        req = dist.all_to_all_single(output, input, async_op=True)
        req.wait()


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
    def forward(ctx, a2ai, *inputs):
        global myreq
        # logger.info("All2All_ScatterList_Req:forward")
        mb_split_lengths = a2ai.gNS if a2ai.gNS else a2ai.lN
        emb_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        gather_list = []
        req_list = []
        for i in range(my_size):
            for j in range(emb_split_lengths[i]):
                out_tensor = inputs[0].new_empty([a2ai.lN, a2ai.E])
                scatter_list = (
                    list(inputs[j].split(mb_split_lengths, dim=0))
                    if i == my_rank
                    else []
                )
                req = dist.scatter(out_tensor, scatter_list, src=i, async_op=True)
                gather_list.append(out_tensor)
                req_list.append(req)
        myreq.req = req_list
        myreq.tensor = tuple(gather_list)
        myreq.a2ai = a2ai
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        # logger.info("All2All_ScatterList_Req:backward")
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
        # logger.info("All2All_Scatter_Wait:forward")
        ctx.a2ai = myreq.a2ai
        for r in myreq.req:
            r.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        a2ai = ctx.a2ai
        grad_output = [t.contiguous() for t in grad_output]
        mb_split_lengths = a2ai.gNS if a2ai.gNS else [a2ai.lN] * my_size
        per_rank_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        grad_inputs = [
            grad_output[0].new_empty([ctx.a2ai.N, ctx.a2ai.E]) for _ in range(a2ai.lS)
        ]
        req_list = []
        ind = 0
        for i in range(my_size):
            for j in range(per_rank_split_lengths[i]):
                gather_list = (
                    list(grad_inputs[j].split(mb_split_lengths, dim=0))
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
    def forward(ctx, a2ai, *inputs):
        global myreq
        # logger.info("All2All_Scatter_Req:forward")
        mb_split_lengths = a2ai.gNS if a2ai.gNS else a2ai.lN
        emb_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        input = torch.cat(inputs, dim=1)
        scatter_list = list(input.split(mb_split_lengths, dim=0))
        gather_list = []
        req_list = []
        for i in range(my_size):
            out_tensor = input.new_empty([a2ai.lN, emb_split_lengths[i] * a2ai.E])
            req = dist.scatter(
                out_tensor, scatter_list if i == my_rank else [], src=i, async_op=True
            )
            gather_list.append(out_tensor)
            req_list.append(req)
        myreq.req = req_list
        myreq.tensor = tuple(gather_list)
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        # logger.info("All2All_Scatter_Req:backward")
        for r in myreq.req:
            r.wait()
        myreq.req = None
        grad_input = myreq.tensor
        grad_inputs = grad_input.split(ctx.a2ai.E, dim=1)
        myreq.tensor = None
        return (None, *grad_inputs)


class All2All_Scatter_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        # logger.info("All2All_Scatter_Wait:forward")
        ctx.a2ai = myreq.a2ai
        for r in myreq.req:
            r.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        # logger.info("All2All_Scatter_Wait:backward")
        assert len(grad_output) == my_size
        scatter_list = [t.contiguous() for t in grad_output]
        a2ai = ctx.a2ai
        mb_split_lengths = a2ai.gNS if a2ai.gNS else a2ai.lN
        # emb_split_lengths = a2ai.gSS if a2ai.gSS else [a2ai.lS] * my_size
        grad_input = grad_output[0].new_empty([a2ai.N, a2ai.E * a2ai.lS])
        gather_list = list(grad_input.split(mb_split_lengths, dim=0))
        req_list = []
        for i in range(my_size):
            # req = dist.scatter(gather_list[i], scatter_list if i == my_rank else [], src=i, async_op=True)
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
    def forward(ctx, a2ai, *inputs):
        global myreq
        # logger.info("All2All_Req:forward")
        mb_split_lengths = a2ai.gNS
        if mb_split_lengths:
            mb_split_lengths = [m * a2ai.E for m in mb_split_lengths]
        emb_split_lengths = a2ai.gSS
        if emb_split_lengths:
            emb_split_lengths = [a2ai.lN * e * a2ai.E for e in emb_split_lengths]
        input = torch.cat(inputs, dim=1).view([-1])
        output = input.new_empty([a2ai.S * a2ai.lN * a2ai.E])
        with record_function("## alltoall_bwd_single ##"):
            req = dist.all_to_all_single(
                output, input, emb_split_lengths, mb_split_lengths, async_op=True
            )

        myreq.req = req
        myreq.tensor = []
        myreq.tensor.append(output)
        myreq.tensor = tuple(myreq.tensor)
        a2ai.mb_split_lengths = mb_split_lengths
        a2ai.emb_split_lengths = emb_split_lengths
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        # logger.info("All2All_Req:backward")
        a2ai = ctx.a2ai
        myreq.req.wait()
        myreq.req = None
        grad_input = myreq.tensor
        grad_inputs = grad_input.view([a2ai.N, -1]).split(a2ai.E, dim=1)
        grad_inputs = [gin.contiguous() for gin in grad_inputs]
        myreq.tensor = None
        return (None, *grad_inputs)


class All2All_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        # logger.info("All2All_Wait:forward")
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        emb_split_lengths = (
            a2ai.emb_split_lengths
            if a2ai.emb_split_lengths
            else a2ai.lS * a2ai.lN * a2ai.E
        )
        outputs = tuple(
            [out.view([a2ai.lN, -1]) for out in output[0].split(emb_split_lengths)]
        )
        # outputs = [out.view([-1, a2ai.lN * a2ai.E]) for out in output[0].split(emb_split_lengths)]
        # outputs = tuple([o.view(-1, a2ai.E) for o in torch.cat(outputs, dim=0)])
        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        global myreq
        # logger.info("All2All_Wait:backward")
        a2ai = ctx.a2ai
        grad_outputs = [gout.contiguous().view([-1]) for gout in grad_outputs]
        grad_output = torch.cat(grad_outputs)
        grad_input = grad_output.new_empty([a2ai.N * a2ai.lS * a2ai.E])
        with record_function("## alltoall_bwd_single ##"):
            req = dist.all_to_all_single(
                grad_input,
                grad_output,
                a2ai.mb_split_lengths,
                a2ai.emb_split_lengths,
                async_op=True,
            )
        myreq.req = req
        myreq.tensor = grad_input
        return (grad_output,)


class All2Alls_Req(Function):
    """ All to All for input sparse data. The forward uses the all to all
        API, and the backward is dummy"
    """

    @staticmethod
    def forward(ctx, a2ai, *inputs):
        global myreq
        # logger.info("All2Alls_Req:forward")
        mb_split_lengths = a2ai.mb_split_lengths
        emb_split_lengths = (
            a2ai.emb_split_lengths if type(a2ai.emb_split_lengths) is list else None
        )
        input = torch.cat(inputs, dim=0).view(-1)
        output = input.new_empty(
            sum(emb_split_lengths) if emb_split_lengths else a2ai.emb_split_lengths
        )
        with record_function("## alltoalls_single ##"):
            req = dist.all_to_all_single(
                output, input, emb_split_lengths, mb_split_lengths, async_op=True
            )

        myreq.req = req
        myreq.tensor = []
        myreq.tensor.append(output)
        myreq.tensor = tuple(myreq.tensor)
        myreq.a2ai = a2ai
        ctx.a2ai = a2ai
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        # logger.info("All2Alls_Req:backward")
        return (None, *grad_output)


class All2Alls_Wait(Function):
    """ All to All Wait for input sparse data. The forward uses the wait
        API, and the backward is dummy"
    """

    @staticmethod
    def forward(ctx, *output):
        global myreq
        # logger.info("All2All_Wait:forward")
        a2ai = myreq.a2ai
        ctx.a2ai = a2ai
        myreq.req.wait()
        myreq.req = None
        myreq.tensor = None
        # We want the results to be lS * N * E.
        # But what we have is ranks * lS * lN * E
        # so we split and cat on the right dim, and resplit
        outputs = tuple([output[0]])
        # outputs = [out.view([-1, a2ai.lN * a2ai.E]) for out in outputs]
        # outputs = tuple(torch.cat(outputs, dim=1).flatten().split(a2ai.N*a2ai.E))

        return outputs

    @staticmethod
    def backward(ctx, *grad_outputs):
        global myreq
        # logger.info("All2All_Wait:backward")
        return (*grad_outputs,)


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
            for l in global_lengths:
                output_size[dim] = l
                gather_list.append(input.new_empty(output_size))

        dist.all_gather(gather_list, input)

        if dim != 0:
            output = torch.cat(gather_list, dim=dim)

        return output

    @staticmethod
    def backward(ctx, grad_output):
        # logger.info("Inside All2AllBackward")
        dim = ctx.dim
        start = ctx.local_start
        length = ctx.local_length

        grad_input = grad_output.narrow(dim, start, length)

        return (grad_input, None, None)


class All2AllInfo(object):
    pass


# alltoall for data distribution
# input: S, lN, E
# output lS, N, E
def alltoalls(inputs, output_split, input_split):
    global myreq
    a2ai = All2AllInfo()
    a2ai.emb_split_lengths = output_split
    a2ai.mb_split_lengths = input_split

    assert hasattr(dist, "all_to_all")
    # logger.info("Using All2All_Req")
    All2Alls_Req.apply(a2ai, *inputs)
    myreq.WaitFunction = All2Alls_Wait

    return myreq


def alltoall(inputs, per_rank_split_lengths):
    global myreq
    N, E = inputs[0].size()
    a2ai = All2AllInfo()
    a2ai.lS = len(inputs)
    a2ai.gSS = per_rank_split_lengths
    a2ai.lN, a2ai.gNS = get_split_lengths_by_len(N)
    a2ai.E = E
    a2ai.N = N
    a2ai.S = sum(a2ai.gSS) if a2ai.gSS else a2ai.lS * my_size

    if hasattr(dist, "all_to_all"):  # and not per_rank_split_lengths:
        # logger.info("Using All2All_Req")
        All2All_Req.apply(a2ai, *inputs)
        myreq.WaitFunction = All2All_Wait
    elif True:
        # logger.info("Using All2All_Scatter_Req")
        All2All_Scatter_Req.apply(a2ai, *inputs)
        myreq.WaitFunction = All2All_Scatter_Wait
    else:
        # logger.info("Using All2All_ScatterList_Req")
        All2All_ScatterList_Req.apply(a2ai, *inputs)
        myreq.WaitFunction = All2All_ScatterList_Wait
    return myreq


def all_gather(input, lengths, dim=0):
    # logger.info("lengths: ", lengths)
    if not lengths:
        lengths = [input.size(0)] * my_size
    return AllGather.apply(input, lengths, dim)


def barrier():
    if my_size > 1:
        dist.barrier()

