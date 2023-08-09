import glob
import os
import os.path as osp
import sys
import shutil
import numpy as np
import scipy.io as sio
from tqdm import tqdm

import torch


dir_path = osp.dirname(osp.realpath(__file__))
datasets_path = osp.join(dir_path, 'NCARS', 'prophesee-automotive-dataset-toolbox')
sys.path.append(datasets_path)


from src.io.psee_loader import PSEELoader
from torch_geometric.data import InMemoryDataset, Data, download_url, extract_zip


def read_events(filename):
    """"Reads in the TD events contained in the NASL dataset file specified by 'filename'

        Python: import scipy.io as sio
            sio.loadmat(filename)

        Each sample contains x, y addresses(x, y), timestamp(ts), polarity(pol).
            x: 0-239    y: 0-179  (dtype=uint8)
            ts: in microsecond  (dtype=int32)
            pol: 1 or 0 (1 means ON polarity, while 0 means OFF polarity.) (dtype=uint8)"""
    video = PSEELoader(filename)
    events = video.load_n_events(video.event_count()) 
    data_x = events["x"].reshape(-1,1).astype(np.float32)
    data_y = events["y"].reshape(-1,1).astype(np.float32)
    data_ts = events["t"].reshape(-1,1).astype(np.float32)
    data_p = events["p"].reshape(-1,1).astype(np.float32) * 2 - 1.0
    pos = np.concatenate([data_x,data_y,data_ts], axis=1)
    pos = torch.from_numpy(pos)
    data_p = torch.from_numpy(data_p)
    data = Data(x=data_p,pos=pos)
    return data



class NCARS(InMemoryDataset):
    r"""The ModelNet10/40 datasets from the `"3D ShapeNets: A Deep
    Representation for Volumetric Shapes"
    <https://people.csail.mit.edu/khosla/papers/cvpr2015_wu.pdf>`_ paper,
    containing CAD models of 10 and 40 categories, respectively.

    .. note::

        Data objects hold mesh faces instead of edge indices.
        To convert the mesh to a graph, use the
        :obj:`torch_geometric.transforms.FaceToEdge` as :obj:`pre_transform`.
        To convert the mesh to a point cloud, use the
        :obj:`torch_geometric.transforms.SamplePoints` as :obj:`transform` to
        sample a fixed number of points on the mesh faces according to their
        face area.

    Args:
        root (string): Root directory where the dataset should be saved.
        name (string, optional): The name of the dataset (:obj:`"10"` for
            ModelNet10, :obj:`"40"` for ModelNet40). (default: :obj:`"10"`)
        train (bool, optional): If :obj:`True`, loads the training dataset,
            otherwise the test dataset. (default: :obj:`True`)
        transform (callable, optional): A function/transform that takes in an
            :obj:`torch_geometric.data.Data` object and returns a transformed
            version. The data object will be transformed before every access.
            (default: :obj:`None`)
        pre_transform (callable, optional): A function/transform that takes in
            an :obj:`torch_geometric.data.Data` object and returns a
            transformed version. The data object will be transformed before
            being saved to disk. (default: :obj:`None`)
        pre_filter (callable, optional): A function that takes in an
            :obj:`torch_geometric.data.Data` object and returns a boolean
            value, indicating whether the data object should be included in the
            final dataset. (default: :obj:`None`)
    """

    urls = {
        '10':
        'http://vision.princeton.edu/projects/2014/3DShapeNets/ModelNet10.zip',
        '40': 'http://modelnet.cs.princeton.edu/ModelNet40.zip'
    }

    def __init__(self, root, name='all', transform=None,
                 pre_transform=None, pre_filter=None):
        self.dataset_names = ['all' , 'training', 'validation', 'test']
        super().__init__(root, transform, pre_transform, pre_filter)
        assert name in self.dataset_names, "'name' should be chosen from 'all', 'training', 'validation', and 'test'. "
        path = self.processed_paths[self.dataset_names.index(name)]
        self.data, self.slices = torch.load(path)

    @property
    def raw_file_names(self):
        return self.dataset_names 

    @property
    def processed_file_names(self):
        return [name + '_filenames_sign_polarity.pt' for name in self.dataset_names]

    def download(self):
        pass
        # path = download_url(self.urls[self.name], self.root)
        # extract_zip(path, self.root)
        # os.unlink(path)
        # folder = osp.join(self.root, f'ModelNet{self.name}')
        # shutil.rmtree(self.raw_dir)
        # os.rename(folder, self.raw_dir)

        # Delete osx metadata generated during compression of ModelNet10
        # metadata_folder = osp.join(self.root, '__MACOSX')
        # if osp.exists(metadata_folder):
        #     shutil.rmtree(metadata_folder)

    def process(self):
        for i, path in enumerate(self.processed_paths):
            torch.save(self.process_set(self.raw_paths[i]), path)
            print(f"'{self.raw_file_names[i]}' is processed and saved as '{self.processed_file_names[i]}'.")


    def process_set(self, raw_path):
        categories = glob.glob(osp.join(raw_path, '*', ''))
        categories = sorted([x.split(os.sep)[-2] for x in categories])

        data_list = []
        for target, category in enumerate(categories):
            folder = osp.join(raw_path, category)
            paths = glob.glob(f'{folder}{os.sep}*.dat')
            for path in tqdm(paths):
                data = read_events(path)
                data.file_id = osp.basename(path)
                data.label = [category]
                data.y = torch.tensor([target])
                data_list.append(data)
            print(f'category {category} is done!')

        if self.pre_filter is not None:
            data_list = [d for d in data_list if self.pre_filter(d)]

        if self.pre_transform is not None:
            data_list = [self.pre_transform(d) for d in data_list]
        
        return self.collate(data_list)

  
    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'


if __name__ == '__main__':
    dir_path = osp.dirname(osp.realpath(__file__))
    dataset_path = osp.join(dir_path,'NCARS','data')
    dataset  = NCARS(dataset_path, transform = None)
    print("Good bye!")