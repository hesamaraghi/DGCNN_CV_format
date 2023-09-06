import subprocess

# Define the values for num_events and method
num_events_values = ["64", "128", "256", "512"]
method_values = ["DGCNN"]

# Define the paths
sbatch_folder = "sbatch_folder"
cfg_folder = "cfg_folder"

# Loop over num_events and method
for num_events in num_events_values:
    for method in method_values:
        # Define the job name
        job_name = f"train_{num_events}_{method}"

        # Define the configuration file path
        cfg_file = f"cfg_path={cfg_folder}/{method}_NCALTECH101_{num_events}_500epoch_replicate_previous.yaml"
        experiment_name = f"\"DGCNN (DelftBlue) {num_events} 500 epochs with outlier removal (replicate previous)\""
        # sbatch_command = f"sbatch {sbatch_folder}/run_train.sbatch python train.py {cfg_file} wandb.experiment_name={experiment_name}"
        # Construct the sbatch command
        
        sbatch_command = f"sbatch {sbatch_folder}/run_train.sbatch python train.py {cfg_file}"
        # print(sbatch_command)
        # Submit the job using subprocess
        subprocess.call(sbatch_command, shell=True)
