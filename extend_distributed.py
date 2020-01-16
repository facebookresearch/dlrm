import os
import torch
from torch.autograd import Function
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

my_rank = -1
my_size = -1
my_local_rank = -1
my_local_size = -1

myreq = None

def env2int(env_list, default = -1):
    for e in env_list:
        val = int(os.environ.get(e, -1))
        if val >= 0: return val
    return default

def get_my_slice(n):
    k, m = divmod(n, my_size)
    return slice(my_rank * k + min(my_rank, m), (my_rank+1) * k + min(my_rank+1, m), 1)

def get_split_lengths(n):
    k, m = divmod(n, my_size)
    if m == 0:
        return k
    else:
        return [(k+1) if i < m else k for i in range(my_size)]

def init_distributed(rank = -1, size = -1, backend=''):
    global myreq
    global my_rank
    global my_size
    global my_local_rank
    global my_local_size

    # guess MPI ranks from env (works for IMPI, OMPI and MVAPICH2)
    num_mpi_ranks = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE'])
    if backend == '' and num_mpi_ranks > 1:
        if dist.is_mpi_available():
            backend = 'mpi'
        else:
            print("WARNING: MPI multi-process launch detected but PyTorch MPI backend not available.")
            backend = 'gloo'

    if backend == 'gloo':
        if not os.environ.get('MASTER_ADDR', None): os.environ['MASTER_ADDR'] = '127.0.0.1'
        if not os.environ.get('MASTER_PORT', None): os.environ['MASTER_PORT'] = '29500'
        # print("MASTER_ADDR", os.environ['MASTER_ADDR'])
        #guess Rank and size
        if rank == -1:
            rank = env2int(['PMI_RANK', 'OMPI_COMM_WORLD_RANK', 'MV2_COMM_WORLD_RANK'], -1)
        if size == -1:
            size = env2int(['PMI_SIZE', 'OMPI_COMM_WORLD_SIZE', 'MV2_COMM_WORLD_SIZE'], -1)

    if rank != -1 or size != -1 or backend != '':
        dist.init_process_group(backend, rank=rank, world_size=size)
        my_rank = dist.get_rank()
        my_size = dist.get_world_size()
        my_local_rank = env2int(['MPI_LOCALRANKID', 'OMPI_COMM_WORLD_LOCAL_RANK', 'MV2_COMM_WORLD_LOCAL_RANK'], 0)
        my_local_size = env2int(['MPI_LOCALNRANKS', 'OMPI_COMM_WORLD_LOCAL_SIZE', 'MV2_COMM_WORLD_LOCAL_SIZE'], 1)
        if my_rank == 0: print("Running on %d ranks using %s backend" % (my_size, backend))
    else:
        my_rank = 0
        my_size = 1
        my_local_rank = 0
        my_local_size = 1
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
    def forward(ctx, per_rank_split_lengths, *inputs):
        global myreq
        mb_split_lengths = get_split_lengths(inputs[0].size(0))
        if not isinstance(mb_split_lengths, (list, tuple)):
            mb_split_lengths = [mb_split_lengths] * my_size
        local_mb = mb_split_lengths[my_rank]
        ifm = inputs[0].size(1)
        gather_list = []
        req_list = []
        for i in range(my_size):
            for j in range(per_rank_split_lengths[i]):
                out_tensor = inputs[0].new_empty([mb_split_lengths[i], ifm])
                scatter_list = list(inputs[j].split(mb_split_lengths, dim = 0)) if i == my_rank else []
                req = dist.scatter(out_tensor, scatter_list, src=i, async_op=True)
                gather_list.append(out_tensor)
                req_list.append(req)

        myreq.req = req_list
        myreq.tensor = tuple(gather_list)
        myreq.mb_split_lengths = mb_split_lengths
        myreq.per_rank_split_lengths = per_rank_split_lengths
        return myreq.tensor
                
    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        #print("All2All_Scatter_Req:backward")
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
        #print("All2All_Scatter_Wait:forward")
        ctx.mb_split_lengths = myreq.mb_split_lengths
        ctx.per_rank_split_lengths = myreq.per_rank_split_lengths
        for r in myreq.req:
            r.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        grad_output = [t.contiguous() for t in grad_output]
        ifm = grad_output[0].size(1)
        mb_split_lengths = ctx.mb_split_lengths
        per_rank_split_lengths = ctx.per_rank_split_lengths
        global_mb = sum(mb_split_lengths)
        grad_inputs = [grad_output[0].new_empty([global_mb, ifm]) for _ in range(per_rank_split_lengths[my_rank])]
        req_list = []
        ind = 0
        for i in range(my_size):
            for j in range(per_rank_split_lengths[i]):
                gather_list = list(grad_inputs[j].split(mb_split_lengths, dim = 0)) if i == my_rank else None
                req = dist.gather(grad_output[ind], gather_list, dst = i, async_op=True)
                req_list.append(req)

        myreq.req = req_list
        myreq.tensor = grad_inputs
        return tuple(grad_output)

class All2All_Scatter_Req(Function):
    @staticmethod
    def forward(ctx, input, per_rank_split_lengths):
        global myreq
        #print("All2All_Scatter_Req:forward")
        global_mb, ifm = input.size()
        mb_split_lengths = get_split_lengths(global_mb)
        if not isinstance(mb_split_lengths, (list, tuple)):
            mb_split_lengths = [mb_split_lengths] * my_size
        local_mb = mb_split_lengths[my_rank]
        assert(ifm == per_rank_split_lengths[my_rank])
        scatter_list = list(input.split(mb_split_lengths, dim=0))
        gather_list = []
        req_list = []
        for i in range(my_size):
            out_tensor = input.new_empty([local_mb, per_rank_split_lengths[i]])
            req = dist.scatter(out_tensor, scatter_list if i == my_rank else [], src=i, async_op=True)
            gather_list.append(out_tensor)
            req_list.append(req)

        myreq.req = req_list
        myreq.tensor = tuple(gather_list)
        myreq.mb_split_lengths = mb_split_lengths
        return myreq.tensor

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        #print("All2All_Scatter_Req:backward")
        for r in myreq.req:
            r.wait()
        myreq.req = None
        grad_input = myreq.tensor
        myreq.tensor = None
        return (grad_input, None)

class All2All_Scatter_Wait(Function):
    @staticmethod
    def forward(ctx, *output):
        global myreq
        #print("All2All_Scatter_Wait:forward")
        ctx.mb_split_lengths = myreq.mb_split_lengths
        for r in myreq.req:
            r.wait()
        myreq.req = None
        myreq.tensor = None
        return output

    @staticmethod
    def backward(ctx, *grad_output):
        global myreq
        #print("All2All_Scatter_Wait:backward")
        assert len(grad_output) == my_size
        scatter_list = [t.contiguous() for t in grad_output]
        local_mb, ifm = grad_output[my_rank].size()
        mb_split_lengths = ctx.mb_split_lengths
        grad_input = grad_output[0].new_empty([sum(mb_split_lengths), ifm])
        gather_list = list(grad_input.split(mb_split_lengths, dim=0))
        req_list = []
        for i in range(my_size):
            #req = dist.scatter(gather_list[i], scatter_list if i == my_rank else [], src=i, async_op=True)
            req = dist.gather(scatter_list[i], gather_list if i == my_rank else [], dst=i, async_op=True)
            req_list.append(req)
        myreq.req = req_list
        myreq.tensor = grad_input
        return grad_output

class AllGather(Function):

    @staticmethod
    def forward(ctx, input, global_lengths, dim=0):
        if not isinstance(global_lengths, (list, tuple)):
            global_lengths = [global_lengths] * my_size

        assert(len(global_lengths) == my_size)
        assert(global_lengths[my_rank] == input.size(dim))
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
        # print("Inside All2AllBackward")
        dim = ctx.dim
        start = ctx.local_start
        length = ctx.local_length

        grad_input = grad_output.narrow(dim, start, length)

        return (grad_input, None, None)

def alltoall(inputs, per_rank_split_lengths):
    global myreq
    fm = inputs[0].size(1)
    need_alltoallv = True
    if not isinstance(per_rank_split_lengths, (list, tuple)):
        per_rank_split_lengths = [per_rank_split_lengths] * my_size
        need_alltoallv = False

    if True:
        per_rank_split_lengths = [f * fm for f in per_rank_split_lengths]
        input = torch.cat(inputs, dim=1)
        assert(input.dim() == 2)
        assert(len(per_rank_split_lengths) == my_size)
        assert(per_rank_split_lengths[my_rank] == input.size(1))
        output = All2All_Scatter_Req.apply(input, per_rank_split_lengths)
        #myreq.tensor = output
        myreq.WaitFunction = All2All_Scatter_Wait
    else:
        output = All2All_ScatterList_Req.apply(per_rank_split_lengths, *inputs)
        myreq.WaitFunction = All2All_ScatterList_Wait
    return myreq

def all_gather(input, lengths, dim=0):
    return AllGather.apply(input, lengths, dim)


