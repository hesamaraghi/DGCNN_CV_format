import os
import os.path as osp

import torch
import argparse

from datatransforms.event_transforms import FilterDataRecursive

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
        self.filter_value_dir = osp.join("datasets_torch_geometric", dataset_name, "filter_values", f"tau_{int(tau)}_filter_size_{filter_size}")
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
    
    image_size_dict = {'DVSGESTURE_TONIC': (128, 128), 'NCARS': (100, 120), 'NASL': (180, 240), 'NCALTECH101': (180, 240)}
    
    process_batch = ProcessBatch(batch_file=args.batch_file, tau=args.tau, filter_size=args.filter_size, image_size=image_size_dict[dataset_name])
    process_batch()
    process_batch.check_batch_file()