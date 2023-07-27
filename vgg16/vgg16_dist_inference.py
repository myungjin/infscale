import os
import threading
import time
import sys
from functools import wraps

import torch
import torch.nn as nn
import torch.distributed.autograd as dist_autograd
import torch.distributed.rpc as rpc
import torch.multiprocessing as mp
import torch.optim as optim
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef

from torchvision.models.resnet import Bottleneck


#########################################################
#           Define Model Parallel VGG 16                #
#########################################################


num_classes = 1000
num_partitions = 3

def conv1x1(in_planes, out_planes):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1)

class VGG16Shard1(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super(VGG16Shard1, self).__init__()

        self.seq1 = nn.Sequential(
            # conv1
            nn.Conv2d(3, 64, 3, padding=1), # 5.66%
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), # 14.41%
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=False)
        ).to(device)
        self.seq2 = nn.Sequential(
            # conv2
            nn.Conv2d(64, 128, 3, padding=1), # 5.94%
            nn.ReLU(),
            nn.Conv2d(128, 128, 3, padding=1), # 10.00%
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=False)
        ).to(device)
        self.device = device
        self._lock = threading.Lock()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]
    
    def forward(self, x_ref):
        x = x_ref.to_here().to(self.device)
        with self._lock:
            x = self.seq1(x)
            out = self.seq2(x)
        return out.cpu()

class VGG16Shard2(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super(VGG16Shard2, self).__init__()

        self.seq1 = nn.Sequential(
            # conv3
            nn.Conv2d(128, 256, 3, padding=1), # 4.71 %
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), # 8.35 %
            nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), # 8.79 %
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=False)
        ).to(device)
        self.seq2 = nn.Sequential(
            # conv4
            nn.Conv2d(256, 512, 3, padding=1), # 4.16 %
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), # 7.79 %
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), # 7.81 %
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=False)
        ).to(device)

        self.device = device
        self._lock = threading.Lock()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]
    
    def forward(self, x_ref):
        x = x_ref.to_here().to(self.device)
        with self._lock:
            x = self.seq1(x)
            out = self.seq2(x)
        return out.cpu()

class VGG16Shard3(nn.Module):
    def __init__(self, device, *args, **kwargs):
        super(VGG16Shard3, self).__init__()

        self.device = device

        self.seq = nn.Sequential(
            # conv5
            nn.Conv2d(512, 512, 3, padding=1), # 2.04 %
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), # 2.04 %
            nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=1), # 2.11 %
            nn.ReLU(),
            nn.MaxPool2d(2, stride=2, return_indices=False)
        ).to(device)
        self.classifier = nn.Sequential(
            nn.Linear(512 * 7 * 7, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, 4096),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(4096, num_classes)
        ).to(device)

        self._lock = threading.Lock()

    def parameter_rrefs(self):
        r"""
        Create one RRef for each parameter in the given local module, and return a
        list of RRefs.
        """
        return [RRef(p) for p in self.parameters()]
    
    def forward(self, x_ref):
        x = x_ref.to_here().to(self.device)
        with self._lock:
            x = self.seq(x)
            # if x.shape[1:] != (512,7,7):
            #     raise Exception("X shape:{}".format(x.shape))
            y = x.reshape(x.shape[0], -1)
            # if y.shape[1] != 25088:
            #     raise Exception("Y shape:{}".format(y.shape))
            out = self.classifier(y)

        return out.cpu()

class RRDistVGG16(nn.Module):
    """
    Assemble two ResNet parts as an nn.Module and define pipelining logic
    """
    def __init__(self, split_size, workers, shards, devices, *args, **kwargs):
        super(RRDistVGG16, self).__init__()

        self.split_size = split_size
        self.shard_refs = [[] for i in range(num_partitions)]
        self.shard_ref_names = [[] for i in range(num_partitions)]
        self.shard_classes = [
            VGG16Shard1,
            VGG16Shard2,
            VGG16Shard3
        ]

        assert len(workers) >= len(shards)
        assert len(shards) <= len(devices)
        for i in range(len(shards)):
            id = shards[i] - 1
            assert id < num_partitions

            ref = rpc.remote(
                workers[i],
                self.shard_classes[id],
                args = (devices[i],) + args,
                kwargs = kwargs
            )
            self.shard_refs[id].append(ref)
            self.shard_ref_names[id].append(self.shard_classes[id])

    def forward(self, xs):
        # Split the input batch xs into micro-batches, and collect async RPC
        # futures into a list
        out_futures = []
        ref_rrp = [0] * num_partitions
        for x in iter(xs.split(self.split_size, dim=0)):
            x_rref = RRef(x)
            for i in range(num_partitions - 1):
                x_rref = self.shard_refs[i][ref_rrp[i]].remote().forward(x_rref)
                ref_rrp[i] = (ref_rrp[i] + 1) % len(self.shard_refs[i])

            z_fut = self.shard_refs[-1][ref_rrp[-1]].rpc_async().forward(x_rref)
            ref_rrp[-1] = (ref_rrp[-1] + 1) % len(self.shard_refs[-1])
            out_futures.append(z_fut)

        # collect and cat all output tensors into one tensor.
        return torch.cat(torch.futures.wait_all(out_futures))

    def parameter_rrefs(self):
        remote_params = []
        for i in range(num_partitions):
            remote_params.extend(self.shard_refs[i][0].remote().parameter_rrefs().to_here())

        return remote_params


#########################################################
#                   Run RPC Processes                   #
#########################################################

num_batches = 100
batch_size = 64
image_w = 224
image_h = 224


def run_master(split_size, num_workers, shards):

    file = open("./vgg16_mild_uneven.csv", "a")
    original_stdout = sys.stdout
    sys.stdout = file

    cuda_list = ["cuda:{}".format(i) for i in range(4)]
    model = RRDistVGG16(split_size, ["worker{}".format(i + 1) for i in range(num_workers)], devices=cuda_list + cuda_list + cuda_list, shards=shards)

    one_hot_indices = torch.LongTensor(batch_size) \
                           .random_(0, num_classes) \
                           .view(batch_size, 1)

    # generating inputs
    inputs = torch.randn(batch_size, 3, image_w, image_h)
    # labels = torch.zeros(batch_size, num_classes) \
    #                 .scatter_(1, one_hot_indices, 1)
    
    print("{}".format(shards),end=", ")
    tik = time.time()
    for i in range(num_batches):
        outputs = model(inputs)

    tok = time.time()
    print(f"{split_size}, {tok - tik}, {(num_batches * batch_size) / (tok - tik)}")

    sys.stdout = original_stdout


def run_worker(rank, world_size, split_size, shards):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '29500'

    # Higher timeout is added to accommodate for kernel compilation time in case of ROCm.
    options = rpc.TensorPipeRpcBackendOptions(num_worker_threads=256, rpc_timeout=300)

    if rank == 0:
        rpc.init_rpc(
            "master",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_master(split_size, num_workers=world_size - 1, shards=shards)
    else:
        rpc.init_rpc(
            f"worker{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        pass

    # block until all rpcs finish
    rpc.shutdown()


if __name__=="__main__":
    file = open("./vgg16_mild_uneven.log", "w")
    original_stdout = sys.stdout
    sys.stdout = file
    combo = [[1, 2, 3], [1, 1, 2, 3], [1, 1, 2, 2, 3]]
    for shards in combo:
        print("Placement:", shards)
        world_size = len(shards) + 1
        for split_size in [1, 2, 4, 8]:
            tik = time.time()
            mp.spawn(run_worker, args=(world_size, split_size, shards), nprocs=world_size, join=True)
            tok = time.time()
            print(f"size of micro-batches = {split_size}, execution time = {tok - tik} s, throughput = {(num_batches * batch_size) / (tok - tik)} samples/sec")

    sys.stdout = original_stdout