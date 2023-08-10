import glob
import os
import os.path as osp
import shutil
import numpy as np
from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool, cpu_count

import torch

from torch_geometric.data import Dataset, Data, download_url, extract_zip


def read_events(filename):
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
    data_p = all_p[td_indices].astype(np.float32) * 2 - 1.0
    data_p = np.expand_dims(data_p, axis=1) 
    pos = np.array([data_x,data_y,data_ts])
    pos = torch.from_numpy(pos)
    pos = pos.transpose(0,1)
    data_p = torch.from_numpy(data_p)
    data = Data(x=data_p,pos=pos)
    return data



class NCALTECH101(Dataset):
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
                 pre_transform=None, pre_filter=None, num_workers=4):
        self.dataset_names = ['all' , 'training', 'validation', 'test']
        assert name in self.dataset_names, "'name' should be chosen from 'all', 'training', 'validation', and 'test'. "
        self.num_workers = num_workers
    
        super().__init__(root, transform, pre_transform, pre_filter)
        
        self.output_files = [f for f in self.processed_paths if f.split(os.sep)[-3] == name]

     
    @property
    def raw_file_names(self):
        all_files = glob.glob(osp.join(self.raw_dir, '*', '*', '*'))
        return [osp.join(*f.split(os.sep)[-3:]) for f in all_files if osp.isfile(f)]  

    @property
    def processed_file_names(self):
        return [osp.splitext(osp.join(*f.split(os.sep)[-3:]))[0]+'.pt' for f in self.raw_file_names]

    def download(self):
        pass
    
    # Define a function that processes a single raw_path
    def process_raw_path(self,raw_path):
        data = read_events(raw_path)
        data.file_id = osp.basename(raw_path)
        data.label = [raw_path.split(os.sep)[-2]]
        data.y = torch.tensor([self.categories.index(data.label[0])])

        if self.pre_filter is not None and not self.pre_filter(data):
            return None

        if self.pre_transform is not None:
            data = self.pre_transform(data)

        idx = self.raw_paths.index(raw_path)
 
        torch.save(data, self.processed_paths[idx])
        # print(f"{idx} {self.raw_file_names[idx]}' is processed and saved as '{self.processed_file_names[idx]}'.")

    
    def process_raw_paths_batch(self,raw_paths):
        for idx_and_raw_path in raw_paths:
            _, raw_path = idx_and_raw_path
            data = read_events(raw_path)
            data.file_id = osp.basename(raw_path)
            data.label = [raw_path.split(os.sep)[-2]]
            data.y = torch.tensor([self.categories.index(data.label[0])])
  
            if self.pre_filter is not None and not self.pre_filter(data):
                continue
            
            if self.pre_transform is not None:
                data = self.pre_transform(data)
            
            tail_raw_path = raw_path.split(os.sep)[-3:]
            tail_raw_path[-1] = osp.splitext(tail_raw_path[-1])[0] + '.pt'
            torch.save(data, osp.join(self.processed_dir, *tail_raw_path))


    def divide_list_into_consecutive_groups(self, input_list, num_groups):
        group_size = len(input_list) // num_groups
        remaining = len(input_list) % num_groups

        groups = []
        start_index = 0

        for _ in range(num_groups):
            group_end = start_index + group_size + (1 if remaining > 0 else 0)
            groups.append([(i, item) for i, item in enumerate(input_list[start_index:group_end], start=start_index)])
            start_index = group_end
            remaining -= 1

        return groups
    
       
    @property
    def categories(self):
        return sorted(os.listdir(osp.join(self.raw_dir, 'all')))
    
    def process(self):
        processed_paths_all = [f for f in self.processed_paths if f.split(os.sep)[-3] == 'all']
        if  len(processed_paths_all) == 0 or not all([osp.exists(f) for f in processed_paths_all]):
            raw_paths_all = [f for f in self.raw_paths if f.split(os.sep)[-3] == 'all']
            self.process_all(raw_paths_all)
 
        for name in self.dataset_names:
            if name != 'all':

                for category in self.categories:
                    if not osp.exists(osp.join(self.processed_dir, name, category)):
                        os.makedirs(osp.join(self.processed_dir, name, category))
        
                name_files = [f for f in self.processed_paths if f.split(os.sep)[-3] == name]
                for file in name_files:
                    src_path = osp.join(self.processed_dir, 'all', *file.split(os.sep)[-2:])
                    try:
                        os.symlink(src_path,file)
                    except FileExistsError:
                        os.unlink(file)
                        os.symlink(src_path,file)
                        
                print(f"Sym links for '{name}' dataset are created again.")
        
    
    def process_all(self,raw_paths_all):
        for category in self.categories:
            if not os.path.exists(osp.join(self.processed_dir,'all', category)):
                os.makedirs(osp.join(self.processed_dir,'all', category))
        
        num_workers = self.num_workers
        
        # groups = [[] for _ in range(num_workers)]

        # for index, raw_path in enumerate(self.raw_paths):
        #     group_index = index % num_workers
        #     groups[group_index].append((index, raw_path))
        
        groups = self.divide_list_into_consecutive_groups(raw_paths_all,num_workers)
        
        processes = []
        
        for group_index in range(num_workers):
            process = multiprocessing.Process(target=self.process_raw_paths_batch, args=(groups[group_index],))
            processes.append(process)
            process.start()

        for process in processes:
            process.join()   
                
        # with Pool(processes=num_workers) as pool:
        #     pool.map(self.process_raw_path, all_name_files)

    def len(self):
        return len(self.output_files)
    
    def get(self, idx):
        data = torch.load(self.output_files[idx])
        return data       

    def __repr__(self) -> str:
        return f'{self.__class__.__name__}({len(self)})'

if __name__ == '__main__':
    dir_path = osp.dirname(osp.realpath(__file__))
    dataset_path = osp.join(dir_path,'NCALTECH101','data')
    print(dataset_path)
    dataset  = NCALTECH101(dataset_path, transform = None)
    print("Good bye!")
