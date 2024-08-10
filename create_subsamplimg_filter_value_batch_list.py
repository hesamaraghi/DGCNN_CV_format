import numpy as np
import os
import os.path as osp
from datasets_torch_geometric.dataset_factory import create_dataset


num_jobs = 100

image_size = 128,128
filter_size = 7
tau = 30 * 1000

dataset_name = 'DVSGESTURE_TONIC'


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