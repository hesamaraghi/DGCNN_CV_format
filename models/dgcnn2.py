import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MLP, EdgeConv, global_max_pool

from models.dynamicedgeconv2 import DynamicEdgeConv2


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

        batch = data.batch
        x0 = torch.cat([pos,p],dim=1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, self.conv1.edge_index)
        out1 = self.lin1(torch.cat([x1, x2], dim=1))
        out2 = global_max_pool(out1, batch)
        out = self.mlp(out2)
        # inters = {'x1': x1, 'x2': x2, 'out1': out1,'out2': out2, 'out': out}
        return F.log_softmax(out, dim=1) #, inters
