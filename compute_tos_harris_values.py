import os
import os.path as osp

import torch
import argparse

from datatransforms.event_transforms import FilterDataTOS2DHarris

class ProcessBatch(FilterDataTOS2DHarris):
    
    def __init__(self, 
                 batch_file = None, 
                 filter_size = None, 
                 TOS_T = None, 
                 Harris_block_size = None, 
                 Harris_ksize = None, 
                 Harris_k = None,
                 image_size = None) -> None:
        
        assert batch_file is not None, "batch_file must be provided"
        assert filter_size is not None, "filter_size must be provided"
        assert TOS_T is not None, "TOS_T must be provided"
        assert Harris_block_size is not None, "Harris_block_size must be provided"
        assert Harris_ksize is not None, "Harris_ksize must be provided"
        assert Harris_k is not None, "Harris_k must be provided"
        assert image_size is not None, "image_size must be provided"
        
        super().__init__(filter_size=filter_size, TOS_T=TOS_T, Harris_block_size=Harris_block_size, Harris_ksize=Harris_ksize, Harris_k=Harris_k, image_size=image_size)
        self.batch_file = batch_file
        # Read the list of sample files
        with open(self.batch_file, 'r') as f:
            self.sample_files = f.read().splitlines()

        dataset_name = batch_file.split(osp.sep)[-4]
        k_vale_in_file = f"{Harris_k:.2e}".replace('.', '_').replace('+', '').replace('-', 'm')
        self.tos_harris_value_dir = osp.join("datasets_torch_geometric", dataset_name, "TOS_Harris_values", f"T_{TOS_T}_filter_size_{filter_size}_Harris_{Harris_block_size}_{Harris_ksize}_{k_vale_in_file}")
        
        if not os.path.exists(self.tos_harris_value_dir):
            os.makedirs(self.tos_harris_value_dir)

    def process(self):
        
        # Process each sample
        for sample_file in self.sample_files:
            class_name = os.path.basename(os.path.dirname(sample_file))        
            output_folder = osp.join(self.tos_harris_value_dir, class_name)
            output_file = osp.join(output_folder, "tos_harris_values_" + osp.basename(sample_file))
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

                tos_harris_value = super().__call__(data)

                # To find reverse_indices such that A = B[reverse_indices]
                reverse_indices = torch.zeros_like(sorted_indices)
                reverse_indices[sorted_indices] = torch.arange(len(sorted_indices))

                tos_harris_value = tos_harris_value[reverse_indices]
            
                tos_harris_value = torch.tensor(tos_harris_value)
                torch.save(tos_harris_value, output_file)

    def __call__(self):
        self.process()
        
    def check_batch_file(self):
        
        is_all_processed = True
        for sample_file in self.sample_files:    
            class_name = os.path.basename(os.path.dirname(sample_file))   
            output_file = osp.join(self.tos_harris_value_dir, class_name, "tos_harris_values_" + osp.basename(sample_file))
            if not osp.exists(output_file):
                print(f"Missing {output_file}")
                is_all_processed = False
        
        if is_all_processed:
            print(f"All processed")
            os.remove(self.batch_file)
            

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-file', type=str, help='Path to the batch file')
    parser.add_argument('--filter-size', type=int, help='Size of the filter', default=7)
    parser.add_argument('--tos-T', type=int, help='Value of TOS_T', default=14)
    parser.add_argument('--Harris-block-size', type=int, help='Value of Harris block_size', default=2)
    parser.add_argument('--Harris-ksize', type=int, help='Value of Harris Sobel kernel size', default=3)
    parser.add_argument('--Harris-k', type=float, help='Value of Harris k value', default=0.04)
    
    args = parser.parse_args()
    
    dataset_name = args.batch_file.split(osp.sep)[-4]
    
    dataset_names = ['DVSGESTURE_TONIC', 'NCARS', 'NASL', 'NCALTECH101']
    
    assert dataset_name in dataset_names, f"Invalid dataset name: {dataset_name}"
    
    image_size_dict = {'DVSGESTURE_TONIC': (128, 128), 'NCARS': (100, 120), 'NASL': (180, 240), 'NCALTECH101': (180, 240)}
    
    process_batch = ProcessBatch(batch_file=args.batch_file, 
                                 filter_size=args.filter_size, 
                                 TOS_T=args.tos_T, 
                                 Harris_block_size=args.Harris_block_size, 
                                 Harris_ksize=args.Harris_ksize, 
                                 Harris_k=args.Harris_k,
                                 image_size=image_size_dict[dataset_name])
    process_batch()
    process_batch.check_batch_file()