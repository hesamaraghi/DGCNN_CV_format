import subprocess
from glob import glob
from os import path as osp
# Define the values for num_events and method



filter_size = 7
TOS_T = 2 * filter_size
Harris_block_size = 2
Harris_ksize = 3 
Harris_k = 0.04
    
dataset_name = 'DVSGESTURE_TONIC'

k_vale_in_file = f"{Harris_k:.2e}".replace('.', '_').replace('+', '').replace('-', 'm')
batch_list_dir = osp.join(  "datasets_torch_geometric", dataset_name, 
                            "TOS_Harris_values", 
                            f"T_{TOS_T}_filter_size_{filter_size}_Harris_{Harris_block_size}_{Harris_ksize}_{k_vale_in_file}_batch_list")
batch_files = glob(osp.join(batch_list_dir, '*.txt'))

# Define the paths
sbatch_folder = "sbatch_folder"
sbath_file = "small_cpu.sbatch"

# Loop over num_events and method
for batch_file in batch_files:   
    
    sbatch_command = (  
        f"sbatch {sbatch_folder}/{sbath_file} "
        f"python compute_tos_harris_values.py "
        f"--batch-file {batch_file} "
        f"--filter-size {filter_size} "
        f"--tos-T {TOS_T} "
        f"--Harris-block-size {Harris_block_size} "
        f"--Harris-ksize {Harris_ksize} "
        f"--Harris-k {Harris_k}"
    )
                                     
    print(sbatch_command)
    subprocess.call(sbatch_command, shell=True)
