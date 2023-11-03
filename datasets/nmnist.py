import os.path as osp
import numpy as np
import torch
from torch_geometric.data import Data

try:
    from datasets.base_dataset import BaseDataset
except ModuleNotFoundError:
    from base_dataset import BaseDataset

class NMNIST(BaseDataset):
    
    def __init__(self, root, name='all', transform=None,
            pre_transform=None, pre_filter=None, num_workers=4):
        super().__init__(root, name, transform, pre_transform, pre_filter, num_workers)


    def read_events(self,filename):
        """"Reads in the TD events contained in the N-MNIST/N-CALTECH101 dataset file specified by 'filename'"""
        with open(filename,'rb') as f:
            raw_data = np.fromfile(f, dtype=np.uint8)
            raw_data = np.uint32(raw_data)

        all_y = raw_data[1::5]
        all_x = raw_data[0::5]
        all_p = (raw_data[2::5] & 128) >> 7 #bit 7
        all_ts = ((raw_data[2::5] & 127) << 16) | (raw_data[3::5] << 8) | (raw_data[4::5])

        #Process time stamp overflow events
        time_increment = 2 ** 13
        overflow_indices = np.where(all_y == 240)[0]
        for overflow_index in overflow_indices:
            all_ts[overflow_index:] += time_increment

        #Everything else is a proper td spike
        td_indices = np.where(all_y != 240)[0]

        
        data_x = all_x[td_indices].astype(np.float32)
        width = data_x.max() + 1
        data_y = all_y[td_indices].astype(np.float32)
        height = data_y.max() + 1
        data_ts = all_ts[td_indices].astype(np.float32)
        data_p = all_p[td_indices].astype(np.float32)
        data_p = np.expand_dims(data_p, axis=1) 
        pos = np.array([data_x,data_y,data_ts])
        pos = torch.from_numpy(pos)
        pos = pos.transpose(0,1)
        data_p = torch.from_numpy(data_p)
        data = Data(x=data_p,pos=pos)
        return data


if __name__ == '__main__':
    dir_path = osp.dirname(osp.realpath(__file__))
    dataset_path = osp.join(dir_path,'NMNIST','data')
    print(dataset_path)
    dataset  = NMNIST(dataset_path, transform = None)
    print("Good bye!")

