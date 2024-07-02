from typing import Union
import os.path as osp
import scipy.io as sio
import re
import random
import math
import numpy as np
import torch
from torch_geometric.transforms import BaseTransform
from torch_geometric.data import Data, HeteroData
import hashlib

try:
    from .event_filters import *
except ModuleNotFoundError:
    from event_filters import *

class TemporalScaling(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """
    def __init__(self,cfg):
        self.temporal_scale = cfg.temporal_scale

    def __call__(self, data):

        data.pos = torch.mul(data.pos,torch.tensor([[1,1,self.temporal_scale]]))        


        return data

def filter_data(data, indices):
    num_nodes = data.num_nodes
    for key, item in data:
        if key == 'num_nodes':
            data.num_nodes = indices.size(0)
        elif bool(re.search('edge', key)):
            continue
        elif (torch.is_tensor(item) and item.size(0) == num_nodes
                and item.size(0) != 1):
            data[key] = item[indices]
    return data


class TemporalQuantization(BaseTransform):
    def __init__(self,cfg):
        self.temporal_num_bins = cfg.temporal_quantization

    def __call__(self, data):
        t = data.pos[..., -1]
        range_t = (t.max() - t.min())*(1 + 2e-3)
        min_t = t.min() - 1e-3 * range_t
        t = (t - min_t) / range_t * self.temporal_num_bins - 0.5
        data.pos[..., -1] = t.round()/self.temporal_num_bins
        return data

class RemoveOutliers(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """
    def __init__(self, cfg):
        self.thresh_quantile = cfg.thresh_quantile
        assert cfg.conv_vec_path is not None, "'conv_vec_path' cannot be empty!"
        self.conv_vec_path = cfg.conv_vec_path

    def __call__(self, data):

        num_nodes = data.num_nodes
        mat_path = osp.join(self.conv_vec_path,data.label[0],osp.splitext(data.file_id)[0]+'.mat')
        mat = sio.loadmat(mat_path)
        td_conv = torch.squeeze(torch.from_numpy(mat['td_conv']))
        
        if self.thresh_quantile > 1.0:
            indices = torch.argsort(td_conv, descending=True)[:self.thresh_quantile]
        else:            
            indices = torch.zeros((num_nodes,), dtype=torch.bool)
            pol_ind = torch.squeeze(data.x < 0.5)
            thr = torch.quantile(td_conv[pol_ind], self.thresh_quantile)
            indices = indices | (pol_ind & (td_conv > thr))
            pol_ind = torch.squeeze(data.x > 0.5)
            thr = torch.quantile(td_conv[pol_ind], self.thresh_quantile)
            indices = indices | (pol_ind & (td_conv > thr))

        indices = torch.nonzero(indices, as_tuple=True)[0]
        return filter_data(data, indices)
   
   
    
class FilterNodes(BaseTransform):
    r"""
    Keep the nodes that satisfy the condition defined in the filter_nodes function.
    """
    def __init__(self, cfg):

        assert cfg.filter_nodes is not None, "'filter_nodes' cannot be empty!"
        self.filter_nodes = eval(cfg.filter_nodes)

    def get_indices(self,data):
        raise NotImplementedError
    
    def __call__(self, data):

        num_nodes = data.num_nodes
        indices = self.filter_nodes(data)
        indices = torch.nonzero(indices, as_tuple=True)[0]
        return filter_data(data, indices)
    

class SpatialCentering(BaseTransform):
    r"""Centers node positions :obj:`pos` around the origin."""
    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                store.pos[..., -2] = store.pos[..., -2] - store.pos[..., -2].mean()
                store.pos[..., -3] = store.pos[..., -3] - store.pos[..., -3].mean()
        return data
    
class SpatialSubsampling(BaseTransform):
    r"""Subsamples events horizontally and vertically.
    """
    def __init__(
        self,
        subsampling_ratios: tuple,
    ):
        assert len(subsampling_ratios) == 2, 'Subsampling ratios must be a tuple of two integers.'
        assert all([isinstance(ratio, int) for ratio in subsampling_ratios]), 'Subsampling ratios must be integers.'
        assert all([ratio > 0 for ratio in subsampling_ratios]), 'Subsampling ratios must be positive integers.'
        self.subsampling_ratios = subsampling_ratios
        
    def __call__(self, data: Data) -> Data:
        pos = data.pos
        x = pos[..., -3]
        y = pos[..., -2]
        mask = torch.ones_like(x, dtype=torch.bool)
        if self.subsampling_ratios[0] > 1:
            mask = mask & (x % self.subsampling_ratios[0] == 0)
        if self.subsampling_ratios[1] > 1:
            mask = mask & (y % self.subsampling_ratios[1] == 0)
        indices = torch.nonzero(mask, as_tuple=True)[0]
        return filter_data(data, indices)

class TemporalSubsampling(BaseTransform):
    r"""Subsamples events temporally.
    """
    def __init__(
        self,
        subsampling_ratio: int,
        window_size: int,
    ):
        assert isinstance(subsampling_ratio, int), 'Subsampling ratio must be an integer.'
        assert subsampling_ratio > 0, 'Subsampling ratio must be a positive integer.'
        self.subsampling_ratio = subsampling_ratio
        self.window_size = window_size * 1000
        
    def __call__(self, data: Data) -> Data:
        pos = data.pos
        t = pos[..., -1].contiguous()
        min_time = t.min()
        max_time = t.max()
        n_slices = torch.ceil((max_time - min_time) / self.window_size).int()
        window_start_times = torch.arange(n_slices) * self.window_size + min_time
        window_end_times = window_start_times + self.window_size/self.subsampling_ratio
        indices_start = torch.searchsorted(t, window_start_times)
        indices_end = torch.searchsorted(t, window_end_times)
        indices = torch.cat([torch.arange(start, end) for start, end in zip(indices_start, indices_end)])
        return filter_data(data, indices)

class FixedSubsampling(BaseTransform):
    r"""Fixed num subsampling of nodes.
    """
    def __init__(
        self,
        cfg,
        replace: bool = True,
        allow_duplicates: bool = False,
    ):
        self.num = cfg.num_events_per_sample
        self.seed_str = cfg.fixed_sampling.seed_str
        self.replace = replace
        self.allow_duplicates = allow_duplicates


    def create_seed(self, data_str: str) -> int:
        # Create a SHA-256 hash of the input string
        hash_object = hashlib.sha256((self.seed_str + '_' + data_str).encode())
        # Convert the hash to a hexadecimal string
        hex_dig = hash_object.hexdigest()
        # Convert the hexadecimal string to an integer
        seed = int(hex_dig, 16)
        return seed

    def __call__(self, data: Data) -> Data:
        num_nodes = data.num_nodes
        seed = self.create_seed(data.label[0] + '_' + data.file_id)
        rng = np.random.default_rng(seed)
        torch_rng = torch.Generator().manual_seed(seed % (2**32))

        if self.replace:
            choice = rng.choice(num_nodes, self.num, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes, generator=torch_rng)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes, generator=torch_rng)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        return filter_data(data, choice)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.num}, replace={self.replace})'

    
    

class SpatialScaling(BaseTransform):
    r"""Scales node positions by a randomly sampled factor :math:`s` within a
    given interval, *e.g.*, resulting in the transformation matrix

    .. math::
        \begin{bmatrix}
            s & 0 & 0 \\
            0 & s & 0 \\
            0 & 0 & 0 \\
        \end{bmatrix}

    for three-dimensional positions.

    Args:
        scales (tuple): scaling factor interval, e.g. :obj:`(a, b)`, then scale
            is randomly sampled from the range
            :math:`a \leq \mathrm{scale} \leq b`.
    """
    def __init__(self, cfg):
        assert len(cfg.scale_limits) == 2
        self.scales = cfg.scale_limits

    def __call__(self, data):
        scale = random.uniform(*self.scales)
        data.pos[...,-2] = data.pos[...,-2] * scale
        data.pos[...,-3] = data.pos[...,-3] * scale
        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({self.scales})'


class AddEdgeAttr(BaseTransform):

    def __init__(self, cfg):
        # self.norm = norm
        # self.max = max_value
        self.cat = cfg.cat

    def __call__(self, data):
        (row, col), pos, pseudo = data.edge_index, data.pos[:, :2], data.edge_attr
        data.pos = pos
        cart = torch.abs(pos[row] - pos[col])
        cart = cart.view(-1, 1) if cart.dim() == 1 else cart

        # if self.norm and cart.numel() > 0:
        #     max_value = cart.abs().max() if self.max is None else self.max
        #     cart = cart / (2 * max_value) + 0.5

        if pseudo is not None and self.cat:
            pseudo = pseudo.view(-1, 1) if pseudo.dim() == 1 else pseudo
            data.edge_attr = torch.cat([pseudo, cart.type_as(pseudo)], dim=-1)
        else:
            data.edge_attr = cart

        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(norm={self.norm}, '
                f'max_value={self.max})')
        
        
        
class ShiftAndFlip(BaseTransform):

    def __init__(self, cfg):
        self.max_shift = cfg.max_shift
        self.p = cfg.p
        if cfg.resolution is None:
            self.resolution = (180,240)
        else:
            self.resolution = cfg.resolution

    def __call__(self, data):
        H, W = self.resolution
        x_shift, y_shift = torch.randint(-self.max_shift, self.max_shift + 1, (2,))
        data.pos[..., -3] += x_shift
        data.pos[..., -2] += y_shift
        
        indices = (data.pos[..., -3] >= 0) & (data.pos[..., -3] < W) & (data.pos[..., -2] >= 0) & (data.pos[..., -2] < H)
        indices = torch.nonzero(indices, as_tuple=True)[0]
        data =  filter_data(data, indices)
        if torch.rand(1) < self.p:
            data.pos[..., -3] = W - 1 - data.pos[..., -3]
            
        return data

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}(max_shift={self.max_shift}, '
                f'p={self.p})')
        
        
        
class VaryingSamplingPoints(BaseTransform):
    r"""Samples a random number of points and features from the nodes.
    Args:
        range_num (tuple): minimun and maximum for random number of points to sample.
        replace (bool, optional): If set to :obj:`False`, samples points
            without replacement. (default: :obj:`True`)
        allow_duplicates (bool, optional): In case :obj:`replace` is
            :obj`False` and :obj:`num` is greater than the number of points,
            this option determines whether to add duplicated nodes to the
            output points or not.
            In case :obj:`allow_duplicates` is :obj:`False`, the number of
            output points might be smaller than :obj:`num`.
            In case :obj:`allow_duplicates` is :obj:`True`, the number of
            duplicated points are kept to a minimum. (default: :obj:`False`)
    """
    def __init__(
        self,
        range_num: tuple,
        replace: bool = True,
        allow_duplicates: bool = False,
        inverse_sampling: bool = False,
    ):
        assert len(range_num) == 2 and range_num[0] < range_num[1]
        self.num_list = np.arange(*range_num)
        if inverse_sampling:
            weights = 1 / self.num_list
            self.weights = weights / weights.sum()
        else:
            self.weights = None
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def __call__(self, data: Data) -> Data:
        
        self.num = np.random.choice(self.num_list, size=1, p=self.weights)[0]
        num_nodes = data.num_nodes

        if self.replace:
            choice = np.random.choice(num_nodes, self.num, replace=True)
            choice = torch.from_numpy(choice).to(torch.long)
        elif not self.allow_duplicates:
            choice = torch.randperm(num_nodes)[:self.num]
        else:
            choice = torch.cat([
                torch.randperm(num_nodes)
                for _ in range(math.ceil(self.num / num_nodes))
            ], dim=0)[:self.num]

        return filter_data(data, choice)

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(({self.range_num[0]},{self.range_num[1]}), replace={self.replace})'
    
    
    
    

class DropEveryNthEvent(BaseTransform):
    """ From tonic.transforms
    
    Deterministically drops every nth event for every spatial location x (and potentially y).

    Parameters:
        n (int): The event stream for each x/y location is reduced to 1/n.

    Example:
        >>> transform = tonic.transforms.Decimation(n=5)
    """

    def __init__(
        self,
        n: int,
    ):
        assert n > 0, "n has to be an integer greater than zero."
        self.n = n
        
    def __call__(self, data: Data) -> Data:
        
        max_x = data.pos[..., -3].max().int().item()
        max_y = data.pos[..., -2].max().int().item()

        indices = []
        memory = torch.zeros((max_x + 1, max_y + 1))

        # Sort the data.pos[..., -1] array in ascending order
        sorted_indices = torch.argsort(data.pos[..., -1])
        data.pos = data.pos[sorted_indices]
        data.x = data.x[sorted_indices]

        for event_num in range(data.num_nodes):
            event_x, event_y = data.pos[event_num,-3].int(), data.pos[event_num,-2].int()
            memory[event_x, event_y] += data.x[event_num,0]
            if torch.abs(memory[event_x, event_y]) >= self.n:
                memory[event_x, event_y] = 0
                indices.append(event_num)
        
        return filter_data(data, indices)


def DropEventRandomly(BaseTransform):

    def __init__(
        self,
        p: float,
    ):
        self.p = p
        
    def __call__(self, data: Data) -> Data:   
        n_events = data.num_nodes
        indices = torch.where(torch.rand(n_events) > self.p)[0]
        return filter_data(data, indices)   