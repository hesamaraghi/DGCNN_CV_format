import numpy as np
import os
import os.path as osp
from scipy.ndimage import gaussian_filter

from typing import Union, Iterable, List, Tuple

import torch
import argparse


class FilterDataRecursive():

    def __init__(self, tau: float, filter_size: int, image_size: Tuple[int,int]):
        
        assert filter_size % 2 == 1, "Filter size must be odd"
        self.tau = tau
        self.filter_size = filter_size
        self.image_size = image_size
        self.K = filter_size // 2
        self.H, self.W = image_size
        
        sigma = filter_size / 5.0
        kernel = np.zeros((filter_size, filter_size))
        kernel[filter_size // 2, filter_size // 2] = 1
        self.gaussian_kernel = gaussian_filter(kernel, sigma)
        self.gaussian_kernel = self.gaussian_kernel / np.sum(self.gaussian_kernel)

    def __call__(self, data):
        
        self.last_time_tensor = np.full((2,self.H,self.W), float('0') , dtype=np.float32)
        self.temporal_accumulation_tensor = np.full((2,self.image_size[0],self.image_size[1]), float('0') , dtype=np.float32)

        filter_value_recursive = np.zeros(data.pos.shape[0], dtype=np.float32)

        # sorted_indices = torch.argsort(data.pos[..., -1])
        # data.pos = data.pos[sorted_indices]
        # data.x = data.x[sorted_indices]

        for i ,ts in enumerate(data.pos):
    
            pp = 0 if data.x[i] < 0 else 1
    
            h = ts[-2].int()
            w = ts[-3].int()
            t = ts[-1].numpy() 
    
            h_start = max(h - self.K, 0)
            h_end = min(h + self.K, self.H-1)
            w_start = max(w - self.K, 0)
            w_end = min(w + self.K, self.W-1)

            
            # Compute the temporal lag
            temporal_lag = np.exp(- (t - self.last_time_tensor[pp,h_start:h_end+1,w_start:w_end+1])/self.tau)

            # update the last time tensor
            self.last_time_tensor[pp,h_start:h_end+1,w_start:w_end+1] = t

            # update the temporal accumulation tensor

            self.temporal_accumulation_tensor[pp,h_start:h_end+1,w_start:w_end+1] *= temporal_lag
            self.temporal_accumulation_tensor[pp,h,w] += 1

            # Compute the filter value
            filter_value_recursive[i] = np.sum(self.temporal_accumulation_tensor[pp,h_start:h_end+1,w_start:w_end+1] * self.gaussian_kernel[h_start - h + self.K:h_end + 1 - h + self.K, w_start - w + self.K:w_end +1 - w + self.K])
    
        return filter_value_recursive
    
    
    
class ProcessBatch(FilterDataRecursive):
    
    def __init__(self,batch_file = None, tau = None, filter_size = None, image_size = None) -> None:
        
        assert batch_file is not None, "batch_file must be provided"
        assert tau is not None, "tau must be provided"
        assert filter_size is not None, "filter_size must be provided"
        assert image_size is not None, "image_size must be provided"
        
        super().__init__(tau = tau, filter_size = filter_size, image_size = image_size)
        self.batch_file = batch_file
        # Read the list of sample files
        with open(self.batch_file, 'r') as f:
            self.sample_files = f.read().splitlines()

        dataset_name = batch_file.split(osp.sep)[-4]
        self.filter_value_dir = osp.join("datasets_torch_geometric", dataset_name, "filter_values", f"tau_{tau}_filter_size_{filter_size}")
        if not os.path.exists(self.filter_value_dir):
            os.makedirs(self.filter_value_dir)

    def process(self):
        
        # Process each sample
        for sample_file in self.sample_files:
            class_name = os.path.basename(os.path.dirname(sample_file))        
            output_folder = osp.join(self.filter_value_dir, class_name)
            output_file = osp.join(output_folder, "filter_values_" + osp.basename(sample_file))
            if osp.exists(output_file):
                print(f"Skipping {sample_file}", flush=True)
                continue
            else:
                if not osp.exists(output_folder):
                    os.makedirs(output_folder)
                print(f"Processing {sample_file}", flush=True)
                data = torch.load(sample_file) 
                sorted_indices = torch.argsort(data.pos[..., -1])
                data.pos = data.pos[sorted_indices]
                data.x = data.x[sorted_indices]

                filter_value_recursive = super().__call__(data)

                # To find reverse_indices such that A = B[reverse_indices]
                reverse_indices = torch.zeros_like(sorted_indices)
                reverse_indices[sorted_indices] = torch.arange(len(sorted_indices))

                filter_value_recursive = filter_value_recursive[reverse_indices]
            
                filter_values = torch.tensor(filter_value_recursive)
                torch.save(filter_values, output_file)

    def __call__(self):
        self.process()
        
    def check_batch_file(self):
        
        is_all_processed = True
        for sample_file in self.sample_files:    
            class_name = os.path.basename(os.path.dirname(sample_file))   
            output_file = osp.join(self.filter_value_dir, class_name, "filter_values_" + osp.basename(sample_file))
            if not osp.exists(output_file):
                print(f"Missing {output_file}")
                is_all_processed = False
        
        if is_all_processed:
            print(f"All processed")
            os.remove(self.batch_file)
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-file', type=str, help='Path to the batch file')
    parser.add_argument('--tau', type=float, default = 30 * 1000, help='Value of tau')
    parser.add_argument('--filter-size', type=int, default = 7, help='Size of the filter')
    args = parser.parse_args()
    
    
    dataset_name = args.batch_file.split(osp.sep)[-4]
    
    dataset_names = ['DVSGESTURE_TONIC', 'NCARS', 'NASL', 'NCALTECH101']
    
    assert dataset_name in dataset_names, f"Invalid dataset name: {dataset_name}"
    
    image_size_dict = {'DVSGESTURE_TONIC': (128, 128), 'NCARS': (128, 128), 'NASL': (180, 240), 'NCALTECH101': (180, 240)}
    
    process_batch = ProcessBatch(batch_file=args.batch_file, tau=args.tau, filter_size=args.filter_size, image_size=image_size_dict[dataset_name])
    process_batch()
    process_batch.check_batch_file()