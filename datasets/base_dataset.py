import glob
import os
import os.path as osp
import multiprocessing
from tqdm import tqdm
from multiprocessing import Pool, cpu_count

import torch

from torch_geometric.data import Dataset, InMemoryDataset

class BaseDataset(Dataset):
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
    
    def read_events(self):
        raise NotImplementedError("Subclasses must implement this method")
    
    # Define a function that processes a single raw_path
    def process_raw_path(self,raw_path):
        data = self.read_events(raw_path)
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
            data = self.read_events(raw_path)
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
    
    @property
    def num_classes(self) -> int:
        return len(self.categories)
    
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
    
class BaseInMemoryDataset(InMemoryDataset):    
    def __init__(self, root, name='all', transform=None,
                 pre_transform=None, pre_filter=None, num_workers=None):
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

    def read_events(self):
        raise NotImplementedError("Subclasses must implement this method")

    def download(self):
        pass

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
            paths = glob.glob(f'{folder}{os.sep}*')
            for path in tqdm(paths):
                data = self.read_events(path)
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

