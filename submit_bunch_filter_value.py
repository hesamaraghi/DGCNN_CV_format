import subprocess
from glob import glob
from os import path as osp
# Define the values for num_events and method


image_size = 128,128

filter_size = 7
tau = 30 * 1000

dataset_name = 'DVSGESTURE_TONIC'

batch_list_dir = osp.join("datasets_torch_geometric", dataset_name, "filter_values", f"tau_{tau}_filter_size_{filter_size}_batch_list")
batch_files = glob(osp.join(batch_list_dir, '*.txt'))

# Define the paths
sbatch_folder = "sbatch_folder"
# sbath_file = 

# Loop over num_events and method
for batch_file in batch_files:   
    sbatch_command = f"sbatch {sbatch_folder}/ python compute_filter_values.py --batch-file {batch_file} --tau {tau} --filter-size {filter_size}"
    subprocess.call(sbatch_command, shell=True)
