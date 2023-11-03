import os.path as osp
import numpy as np
import scipy.io as sio
import torch
from torch_geometric.data import  Data

try:
    from datasets.base_dataset import BaseDataset
except ModuleNotFoundError:
    from base_dataset import BaseDataset


class NASL(BaseDataset):

    def __init__(self, root, name='all', transform=None,
            pre_transform=None, pre_filter=None, num_workers=4):
        super().__init__(root, name, transform, pre_transform, pre_filter, num_workers)

    def read_events(self,filename):
        """"Reads in the TD events contained in the NASL dataset file specified by 'filename'

            Python: import scipy.io as sio
                sio.loadmat(filename)

            Each sample contains x, y addresses(x, y), timestamp(ts), polarity(pol).
                x: 0-239    y: 0-179  (dtype=uint8)
                ts: in microsecond  (dtype=int32)
                pol: 1 or 0 (1 means ON polarity, while 0 means OFF polarity.) (dtype=uint8)"""
        
        td = sio.loadmat(filename)
        data_x = td["x"].astype(np.float32)
        data_y = td["y"].astype(np.float32)
        data_ts = td["ts"].astype(np.float32)
        data_p = td["pol"].astype(np.float32) * 2 - 1.0
        pos = np.concatenate([data_x,data_y,data_ts], axis=1)
        pos = torch.from_numpy(pos)
        data_p = torch.from_numpy(data_p)
        data = Data(x=data_p,pos=pos)
        return data
 
if __name__ == '__main__':
    dir_path = osp.dirname(osp.realpath(__file__))
    dataset_path = osp.join(dir_path,'NASL','data')
    dataset  = NASL(dataset_path, transform = None)
    print("Good bye!")