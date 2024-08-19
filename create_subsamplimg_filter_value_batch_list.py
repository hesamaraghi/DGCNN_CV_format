import numpy as np
import os
import os.path as osp
from datasets_torch_geometric.dataset_factory import create_dataset
import argparse
from glob import glob

def create_batches(dataset_name, num_jobs, filter_size, tau):
    
    dataset = create_dataset(dataset_name = dataset_name,
                            dataset_path = os.path.join("datasets_torch_geometric",dataset_name,"data"), 
                            dataset_type = "all")

    samples = dataset.output_files

    batch_list_dir = osp.join("datasets_torch_geometric", dataset_name, "filter_values", f"tau_{tau}_filter_size_{filter_size}_batch_list")
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
    parser.add_argument('--tau', type=int, help='Value of tau', default=30 * 1000)
    args = parser.parse_args()
    
    num_jobs = args.num_jobs
    dataset_name = args.dataset_name
    filter_size = args.filter_size
    tau = args.tau
    
    batch_files = create_batches(dataset_name = dataset_name, 
                             num_jobs = num_jobs, 
                             filter_size = filter_size, 
                             tau = tau)
    
    
    
    