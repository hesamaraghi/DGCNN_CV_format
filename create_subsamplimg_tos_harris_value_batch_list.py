import numpy as np
import os
import os.path as osp
from datasets_torch_geometric.dataset_factory import create_dataset
import argparse
from glob import glob

def create_batches(dataset_name, num_jobs, filter_size, TOS_T, Harris_block_size, Harris_ksize, Harris_k):    
    
    dataset = create_dataset(dataset_name = dataset_name,
                            dataset_path = os.path.join("datasets_torch_geometric",dataset_name,"data"), 
                            dataset_type = "all")

    samples = dataset.output_files
    k_vale_in_file = f"{Harris_k:.2e}".replace('.', '_').replace('+', '').replace('-', 'm')
    batch_list_dir = osp.join("datasets_torch_geometric", dataset_name, 
                              "TOS_Harris_values", 
                              f"T_{TOS_T}_filter_size_{filter_size}_Harris_{Harris_block_size}_{Harris_ksize}_{k_vale_in_file}_batch_list")

    if not osp.exists(batch_list_dir):
        os.makedirs(batch_list_dir)

    # Define the number of jobs and divide the samples

    batches = np.array_split(samples, num_jobs)

    
    # Save batch files (optional step to save batches on disk)
    for i, batch in enumerate(batches):
        with open(osp.join(batch_list_dir,f'batch_{i}.txt'), 'w') as f:
            for sample in batch:
                f.write(f"{sample}\n")
                
    batch_files = glob(osp.join(batch_list_dir, '*.txt'))                
    return batch_files

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--num-jobs', type=int, help='Number of jobs', default=100)
    parser.add_argument('--dataset-name', type=str, help='Name of the dataset', default='DVSGESTURE_TONIC')
    parser.add_argument('--filter-size', type=int, help='Size of the filter', default=7)
    parser.add_argument('--tos-T', type=int, help='Value of TOS_T', default=14)
    parser.add_argument('--Harris-block-size', type=int, help='Value of Harris block_size', default=2)
    parser.add_argument('--Harris-ksize', type=int, help='Value of Harris Sobel kernel size', default=3)
    parser.add_argument('--Harris-k', type=float, help='Value of Harris k value', default=0.04)
    
    args = parser.parse_args()
    
    num_jobs = args.num_jobs
    dataset_name = args.dataset_name
    filter_size = args.filter_size
    TOS_T = args.tos_T
    Harris_block_size = args.Harris_block_size
    Harris_ksize = args.Harris_ksize
    Harris_k = args.Harris_k
    
    batch_files = create_batches(dataset_name = dataset_name, 
                                num_jobs = num_jobs, 
                                filter_size = filter_size, 
                                TOS_T = TOS_T,
                                Harris_block_size = Harris_block_size,
                                Harris_ksize = Harris_ksize,
                                Harris_k = Harris_k)
    
    
    
    
    