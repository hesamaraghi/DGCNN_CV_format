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


        for key, item in data:
            if key == 'num_nodes':
                data.num_nodes = torch.sum(indices)
            elif bool(re.search('edge', key)):
                continue
            elif (torch.is_tensor(item) and item.size(0) == num_nodes
                  and item.size(0) != 1):
                data[key] = item[indices]
                
        return data
   
   
    
class FilterNodes(BaseTransform):
    r"""Centers and normalizes node positions to the interval :math:`(-1, 1)`
    (functional name: :obj:`normalize_scale`).
    """
    def __init__(self, cfg):

        assert cfg.filter_nodes is not None, "'filter_nodes' cannot be empty!"
        self.filter_nodes = cfg.filter_nodes

    def __call__(self, data):

        num_nodes = data.num_nodes
        indices = eval(self.filter_nodes)(data)


        for key, item in data:
            if key == 'num_nodes':
                data.num_nodes = torch.sum(indices)
            elif bool(re.search('edge', key)):
                continue
            elif (torch.is_tensor(item) and item.size(0) == num_nodes
                  and item.size(0) != 1):
                data[key] = item[indices]
        return data
    

class SpatialCentering(BaseTransform):
    r"""Centers node positions :obj:`pos` around the origin."""
    def __call__(self, data: Union[Data, HeteroData]):
        for store in data.node_stores:
            if hasattr(store, 'pos'):
                store.pos[..., -2] = store.pos[..., -2] - store.pos[..., -2].mean()
                store.pos[..., -3] = store.pos[..., -3] - store.pos[..., -3].mean()
        return data
    



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

    def __call__(self, data,resolution=(180,240)):
        H, W = resolution
        x_shift, y_shift = torch.randint(-self.max_shift, self.max_shift + 1, (2,))
        data.pos[..., -3] += x_shift
        data.pos[..., -2] += y_shift
        
        indices = (data.pos[..., -3] >= 0) & (data.pos[..., -3] < W) & (data.pos[..., -2] >= 0) & (data.pos[..., -2] < H)
        
        num_nodes = data.num_nodes   
        for key, item in data:
            if key == 'num_nodes':
                data.num_nodes = torch.sum(indices)
            elif bool(re.search('edge', key)):
                continue
            elif (torch.is_tensor(item) and item.size(0) == num_nodes
                  and item.size(0) != 1):
                data[key] = item[indices]
                

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
    ):
        self.range_num = range_num
        self.replace = replace
        self.allow_duplicates = allow_duplicates

    def __call__(self, data: Data) -> Data:
        
        min_num, max_num = self.range_num
        self.num = random.randint(min_num, max_num)
        
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

        for key, item in data:
            if key == 'num_nodes':
                data.num_nodes = choice.size(0)
            elif bool(re.search('edge', key)):
                continue
            elif (torch.is_tensor(item) and item.size(0) == num_nodes
                  and item.size(0) != 1):
                data[key] = item[choice]

        return data

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}(({self.range_num[0]},{self.range_num[1]}), replace={self.replace})'