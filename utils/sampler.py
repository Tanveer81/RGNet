import copy
import os
import random
from typing import Optional
import torch
from torch.utils.data import Sampler, Dataset, DataLoader, DistributedSampler
import collections
import utils.misc as util
from rgnet.config import BaseOptions
import torch.distributed as dist

# https://discuss.pytorch.org/t/using-distributedsampler-in-combination-with-batch-sampler-to-make-sure-batches-have-sentences-of-similar-length/119824/3
class DS(Dataset):
    def __init__(self, data):
        super(DS, self).__init__()
        self.data = data
        self.counts = collections.Counter(self.data)
        self.movies = [[] for _ in range(len(self.counts.keys()))]
        for i, x in enumerate(self.data):
            self.movies[x].append(i)
    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

class MovieSampler(Sampler):
    def __init__(self, movies, batch_size, dist=False, seed=0, epoch=0):
        self.movies=movies
        self.batch_size=batch_size
        self.seed=seed
        self.epoch=epoch
        self.dist=dist
        self.batches = self.get_batches()
        self.num_samples = len(self.batches)

    def partition(self, list_in):
        if self.dist:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(list_in), generator=g).tolist()
            list_in = [list_in[i] for i in indices]
        else:
            random.shuffle(list_in)
        n = len(list_in)//self.batch_size
        batches = [list_in[i*self.batch_size:(i+1)*self.batch_size] for i in range(n)]
        if len(list_in) % self.batch_size!=0:
            batches.append(list_in[-self.batch_size:])
        return batches
    def get_batches(self):
        batches = []
        movies = copy.deepcopy(self.movies)
        for cls in movies:
            if len(cls) >= self.batch_size:
                batches.extend(self.partition(cls))
        if self.dist:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(batches), generator=g).tolist()
            batches = [batches[i] for i in indices]
        else:
            random.shuffle(batches)
        return batches
    def __iter__(self):
        return iter(self.get_batches())
    def __len__(self) -> int:
        return self.num_samples


class DistributedMovieSampler(DistributedSampler):
    def __init__(self, dataset: Dataset, num_replicas: Optional[int] = None,
                 rank: Optional[int] = None, shuffle: bool = False,
                 seed: int = 0, drop_last: bool = False, batch_size = 10) -> None:
        super().__init__(dataset=dataset,num_replicas=num_replicas,rank=rank,shuffle=shuffle,seed=seed,drop_last=drop_last)
        self.batch_size = batch_size
        batch_sampler = MovieSampler(self.dataset,batch_size=self.batch_size,dist=True,seed=self.seed,epoch=self.epoch)
        batches = batch_sampler.batches
        if len(batches) % self.num_replicas==0:
            padding_size=0
        else:
            padding_size = self.num_replicas - (len(batches) % self.num_replicas)
        batches += batches[:padding_size]
        batches = batches[self.rank:len(batches):self.num_replicas]
        self.batches=batches
        self.num_samples=len(self.batches)
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        return iter(self.batches)

#https://github.com/SeungjunNah/DeepDeblur-PyTorch/blob/master/src/data/sampler.py
class DistributedEvalSampler(Sampler):
    r"""
    DistributedEvalSampler is different from DistributedSampler.
    It does NOT add extra samples to make it evenly divisible.
    DistributedEvalSampler should NOT be used for training. The distributed processes could hang forever.
    See this issue for details: https://github.com/pytorch/pytorch/issues/22584
    shuffle is disabled by default

    DistributedEvalSampler is for evaluation purpose where synchronization does not happen every epoch.
    Synchronization should be done outside the dataloader loop.

    Sampler that restricts data loading to a subset of the dataset.

    It is especially useful in conjunction with
    :class:`torch.nn.parallel.DistributedDataParallel`. In such a case, each
    process can pass a :class`~torch.utils.data.DistributedSampler` instance as a
    :class:`~torch.utils.data.DataLoader` sampler, and load a subset of the
    original dataset that is exclusive to it.

    .. note::
        Dataset is assumed to be of constant size.

    Arguments:
        dataset: Dataset used for sampling.
        num_replicas (int, optional): Number of processes participating in
            distributed training. By default, :attr:`rank` is retrieved from the
            current distributed group.
        rank (int, optional): Rank of the current process within :attr:`num_replicas`.
            By default, :attr:`rank` is retrieved from the current distributed
            group.
        shuffle (bool, optional): If ``True`` (default), sampler will shuffle the
            indices.
        seed (int, optional): random seed used to shuffle the sampler if
            :attr:`shuffle=True`. This number should be identical across all
            processes in the distributed group. Default: ``0``.

    .. warning::
        In distributed mode, calling the :meth`set_epoch(epoch) <set_epoch>` method at
        the beginning of each epoch **before** creating the :class:`DataLoader` iterator
        is necessary to make shuffling work properly across multiple epochs. Otherwise,
        the same ordering will be always used.

    Example::

        >>> sampler = DistributedSampler(dataset) if is_distributed else None
        >>> loader = DataLoader(dataset, shuffle=(sampler is None),
        ...                     sampler=sampler)
        >>> for epoch in range(start_epoch, n_epochs):
        ...     if is_distributed:
        ...         sampler.set_epoch(epoch)
        ...     train(loader)
    """

    def __init__(self, dataset, num_replicas=None, rank=None, shuffle=False, seed=0):
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.epoch = 0
        # self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.num_replicas))
        # self.total_size = self.num_samples * self.num_replicas
        self.total_size = len(self.dataset)         # true value without extra samples
        indices = list(range(self.total_size))
        indices = indices[self.rank:self.total_size:self.num_replicas]
        self.num_samples = len(indices)             # true value without extra samples

        self.shuffle = shuffle
        self.seed = seed

    def __iter__(self):
        if self.shuffle:
            # deterministically shuffle based on epoch and seed
            g = torch.Generator()
            g.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=g).tolist()
        else:
            indices = list(range(len(self.dataset)))


        # # add extra samples to make it evenly divisible
        # indices += indices[:(self.total_size - len(indices))]
        # assert len(indices) == self.total_size

        # subsample
        indices = indices[self.rank:self.total_size:self.num_replicas]
        assert len(indices) == self.num_samples

        return iter(indices)

    def __len__(self):
        return self.num_samples

    def set_epoch(self, epoch):
        r"""
        Sets the epoch for this sampler. When :attr:`shuffle=True`, this ensures all replicas
        use a different random ordering for each epoch. Otherwise, the next iteration of this
        sampler will yield the same ordering.

        Arguments:
            epoch (int): _epoch number.
        """
        self.epoch = epoch

if __name__ == '__main__':
    base = BaseOptions()
    base.initialize()
    opt = base.parser.parse_args()
    bs = 32
    # Distributed training setup
    if opt.cuda_visible_devices is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, opt.cuda_visible_devices))
    util.init_distributed_mode(opt)
    data = [k for j in [[i]*random.randint(bs,bs*5) for i in list(range(100))] for k in j]
    ds = DS(data)
    dl = DataLoader(ds, batch_sampler=DistributedMovieSampler(ds.movies,batch_size=bs))
    batches = list(iter(dl))
    print()