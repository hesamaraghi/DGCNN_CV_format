import torch

import torch.nn.functional as F
import torch_geometric.transforms as T
from torch_geometric.nn import SplineConv, voxel_grid, max_pool, max_pool_x, graclus, global_mean_pool, NNConv




class ResidualBlock(torch.nn.Module):
    def __init__(self, in_channel, out_channel):
        super(ResidualBlock, self).__init__()
        self.left_conv1 = SplineConv(in_channel, out_channel, dim=2, kernel_size=5)
        self.left_bn1 = torch.nn.BatchNorm1d(out_channel)
        self.left_conv2 = SplineConv(out_channel, out_channel, dim=2, kernel_size=5)
        self.left_bn2 = torch.nn.BatchNorm1d(out_channel)
        
        self.shortcut_conv = SplineConv(in_channel, out_channel, dim=2, kernel_size=1)
        self.shortcut_bn = torch.nn.BatchNorm1d(out_channel)
        
     
    def forward(self, data):
        
        data.x = F.elu(self.left_bn2(self.left_conv2(F.elu(self.left_bn1(self.left_conv1(data.x, data.edge_index, data.edge_attr))),
                                            data.edge_index, data.edge_attr)) + 
                       self.shortcut_bn(self.shortcut_conv(data.x, data.edge_index, data.edge_attr)))
        
        return data
        
                    
class Net(torch.nn.Module):
    def __init__(self, cfg):
        super(Net, self).__init__()
        out_channels = cfg.dataset.num_classes
        self.conv1 = SplineConv(1, 64, dim=2, kernel_size=5)
        self.bn1 = torch.nn.BatchNorm1d(64)  
        
        self.block1 = ResidualBlock(64, 128)
        self.block2 = ResidualBlock(128, 256)
        self.block3 = ResidualBlock(256, 512)

        self.fc1 = torch.nn.Linear(64 * 512, 1024)
        self.fc2 = torch.nn.Linear(1024, out_channels)
        
    def forward(self, data_org):
        data = data_org.clone()
        data.x = F.elu(self.bn1(self.conv1(data.x, data.edge_index, data.edge_attr)))
        cluster = voxel_grid(pos = data.pos, batch = data.batch, size=[4,4])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data = self.block1(data)
        cluster = voxel_grid(pos = data.pos, batch = data.batch, size=[6,6])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data = self.block2(data)
        cluster = voxel_grid(pos = data.pos, batch = data.batch, size=[20,20])
        data = max_pool(cluster, data, transform=T.Cartesian(cat=False))
        
        data = self.block3(data)
        cluster = voxel_grid(pos = data.pos, batch = data.batch, size=[32,32])
        x,_ = max_pool_x(cluster, data.x, data.batch, size=64)
        
        x = x.view(-1, self.fc1.weight.size(1))
        x = F.elu(self.fc1(x))
        x = F.dropout(x, training=self.training)
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)
        
              
