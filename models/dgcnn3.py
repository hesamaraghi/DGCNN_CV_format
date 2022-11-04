import os.path as osp
import torch
import torch.nn.functional as F
from torch.nn import Linear
from torch_geometric.nn import MLP, DynamicEdgeConv, global_max_pool


class Net(torch.nn.Module):
    def __init__(self, out_channels, k=20, aggr='max'):
        super().__init__()

        self.conv1 = DynamicEdgeConv(MLP([2 * 4, 64, 64, 64]), k, aggr)
        self.conv2 = DynamicEdgeConv(MLP([2 * 64, 64]), k, aggr)
        self.conv3 = DynamicEdgeConv(MLP([2 * 64, 128]), k, aggr)
        self.lin1 = Linear(128 + 64 + 64, 1024)

        self.mlp = MLP([1024, 512, 256, out_channels], dropout=0.5,
                       batch_norm=False)

    def forward(self, data):
        pos, p, batch = data.pos, data.x, data.batch
        x0 = torch.cat([pos,torch.add(-1, p, alpha=2)],dim=1)
        x1 = self.conv1(x0, batch)
        x2 = self.conv2(x1, batch)
        x3 = self.conv3(x2, batch)
        out = self.lin1(torch.cat([x1, x2, x3], dim=1))
        out = global_max_pool(out, batch)
        out = self.mlp(out)
        return F.log_softmax(out, dim=1)


class DGCNN(Net):
    def __init__(self, out_channels, k=20, aggr='max', path_model = None):
        super().__init__(out_channels, k, aggr)
        if path_model is None:
            self.path_model = osp.join(osp.dirname(osp.realpath(__file__)),'models','model_params.pt')
        else:
            self.path_model = path_model

        self.load_state_dict(torch.load(self.path_model))
