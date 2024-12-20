import subprocess
from glob import glob
from os import path as osp
# Define the values for num_events and method



filter_size = 9
tau = 100 * 1000
tau = int(tau)

dataset_name = 'NCARS'

batch_list_dir = osp.join("datasets_torch_geometric", dataset_name, "filter_values", f"tau_{tau}_filter_size_{filter_size}_batch_list")
batch_files = glob(osp.join(batch_list_dir, '*.txt'))

# Define the paths
sbatch_folder = "sbatch_folder"
sbath_file = "small_cpu.sbatch"

# Loop over num_events and method
for batch_file in batch_files:   
    sbatch_command = f"sbatch {sbatch_folder}/{sbath_file} python compute_filter_values.py --batch-file {batch_file} --tau {tau} --filter-size {filter_size}"
    print(sbatch_command)
    subprocess.call(sbatch_command, shell=True)
