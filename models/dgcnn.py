import os.path as osp

import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MLP, EdgeConv, DynamicEdgeConv, global_max_pool


class Net(torch.nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.k = cfg.model.k
        self.aggr = cfg.model.aggr
        self.conv1 = DynamicEdgeConv(MLP([2 * 4, 64, 64, 64]), self.k, self.aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 128]), self.k, self.aggr)
        # self.conv2 = EdgeConv(MLP([2 * 64, 128]), aggr)
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
            x2 = self.conv2(x1, batch)
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

class DGCNN(Net):
    def __init__(self, out_channels, k=20, aggr='max', path_model = None):
        super().__init__(out_channels, k, aggr)
        if path_model is None:
            self.path_model = osp.join(osp.dirname(osp.realpath(__file__)),'models','model_params_beta_9e-6.pt')
        else:
            self.path_model = path_model
        
        self.load_state_dict(torch.load(self.path_model))
