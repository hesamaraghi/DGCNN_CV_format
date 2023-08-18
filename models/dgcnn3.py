import os.path as osp
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
        self.conv2 = EdgeConv(MLP([2 * 64, 64]), self.aggr)
        self.conv3 = EdgeConv(MLP([2 * 64, 128]), self.aggr)
        self.lin1 = Linear(128 + 64 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, cfg.dataset.num_classes], dropout=0.5,
                       batch_norm=False)

    def forward(self, data):
        pos, p, batch = data.pos, data.x, data.batch
        x0 = torch.cat([pos,p],dim=1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, self.conv1.edge_index)
        x3 = self.conv3(x2, self.conv1.edge_index)
        out1 = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out2 = global_max_pool(out1, batch)
        out = self.mlp(out2)
        return F.log_softmax(out, dim=1)


class DGCNN(Net):
    def __init__(self, out_channels, k=20, aggr='max', path_model = None):
        super().__init__(out_channels, k, aggr)
        if path_model is None:
            self.path_model = osp.join(osp.dirname(osp.realpath(__file__)),'models','model_params.pt')
        else:
            self.path_model = path_model

        self.load_state_dict(torch.load(self.path_model))


