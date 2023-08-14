import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MLP, EdgeConv, DynamicEdgeConv, global_max_pool

from typing import Callable, Optional, Union
from torch import Tensor

from torch_geometric.typing import OptTensor, PairOptTensor, PairTensor
from torch_cluster import knn




class DynamicEdgeConv2(DynamicEdgeConv):
    r"""The dynamic edge convolutional operator from the `"Dynamic Graph CNN
    for Learning on Point Clouds" <https://arxiv.org/abs/1801.07829>`_ paper
    (see :class:`torch_geometric.nn.conv.EdgeConv`), where the graph is
    dynamically constructed using nearest neighbors in the feature space.

    Args:
        nn (torch.nn.Module): A neural network :math:`h_{\mathbf{\Theta}}` that
            maps pair-wise concatenated node features :obj:`x` of shape
            `:obj:`[-1, 2 * in_channels]` to shape :obj:`[-1, out_channels]`,
            *e.g.* defined by :class:`torch.nn.Sequential`.
        k (int): Number of nearest neighbors.
        aggr (str, optional): The aggregation scheme to use
            (:obj:`"add"`, :obj:`"mean"`, :obj:`"max"`).
            (default: :obj:`"max"`)
        num_workers (int): Number of workers to use for k-NN computation.
            Has no effect in case :obj:`batch` is not :obj:`None`, or the input
            lies on the GPU. (default: :obj:`1`)
        **kwargs (optional): Additional arguments of
            :class:`torch_geometric.nn.conv.MessagePassing`.

    Shapes:
        - **input:**
          node features :math:`(|\mathcal{V}|, F_{in})` or
          :math:`((|\mathcal{V}|, F_{in}), (|\mathcal{V}|, F_{in}))`
          if bipartite,
          batch vector :math:`(|\mathcal{V}|)` or
          :math:`((|\mathcal{V}|), (|\mathcal{V}|))`
          if bipartite *(optional)*
        - **output:** node features :math:`(|\mathcal{V}|, F_{out})` or
          :math:`(|\mathcal{V}_t|, F_{out})` if bipartite
    """
    def __init__(self, nn: Callable, k: int, aggr: str = 'max',
                 num_workers: int = 1, **kwargs):
        super().__init__(nn=nn, k=k, aggr=aggr, num_workers=num_workers, **kwargs)

    def forward(
            self, x: Union[Tensor, PairTensor],
            batch: Union[OptTensor, Optional[PairTensor]] = None) -> Tensor:
        # type: (Tensor, OptTensor) -> Tensor  # noqa
        # type: (PairTensor, Optional[PairTensor]) -> Tensor  # noqa

        if isinstance(x, Tensor):
            x: PairTensor = (x, x)

        if x[0].dim() != 2:
            raise ValueError("Static graphs not supported in DynamicEdgeConv")

        b: PairOptTensor = (None, None)
        if isinstance(batch, Tensor):
            b = (batch, batch)
        elif isinstance(batch, tuple):
            assert batch is not None
            b = (batch[0], batch[1])

        self.edge_index = knn(x[0], x[1], self.k, b[0], b[1]).flip([0])

        # propagate_type: (x: PairTensor)
        return self.propagate(self.edge_index, x=x, size=None)



class Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.model.k
        self.aggr = cfg.model.aggr
        self.conv1 = DynamicEdgeConv2(MLP([2 * 4, 64, 64, 64]), self.k, self.aggr)
        # self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.conv2 = EdgeConv(MLP([2 * 64, 128]), self.aggr)
        self.lin1 = Linear(128 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, cfg.dataset.num_classes], dropout=0.5,
                       batch_norm=False)

    def forward(self, data):
        pos = data.pos
        p = data.x
        if(data.batch is not None):
            batch = data.batch
            x0 = torch.cat([pos,p],dim=1)
            x1 = self.conv1(x0, batch)
            x2 = self.conv2(x1, self.conv1.edge_index)
            out1 = self.lin1(torch.cat([x1, x2], dim=1))
            out2 = global_max_pool(out1, batch)
        else:
            x1 = self.conv1(pos)
            x2 = self.conv2(x1)
            out1 = self.lin1(torch.cat([x1, x2], dim=1))
            out2 = out1.sum(dim=-2, keepdim=out1.dim() == 2)
        out = self.mlp(out2)
        # inters = {'x1': x1, 'x2': x2, 'out1': out1,'out2': out2, 'out': out}
        return F.log_softmax(out, dim=1) #, inters
