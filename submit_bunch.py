import subprocess

# Define the values for num_events and method
num_events_values = ["64", "128", "256", "512", "1024", "10000", "20000"]
method_values = ["ShuffleNet", "MobileNet"]

# Define the paths
sbatch_folder = "sbatch_folder"
cfg_folder = "cfg_folder"

# Loop over num_events and method
for num_events in num_events_values:
    for method in method_values:
        # Define the job name
        job_name = f"train_{num_events}_{method}"

        # Define the configuration file path
        cfg_file = f"cfg_path={cfg_folder}/EST_NCALTECH101_{num_events}_{method}_500epoch_batch_32_remove_outliers_not_pretrained.yaml"
        experiment_name = f"\"EST {num_events} {method} 250 epochs batch 32 with outlier removal (not pretrained)\""
        # Construct the sbatch command
        sbatch_command = f"sbatch {sbatch_folder}/run_train.sbatch python train.py {cfg_file} train.epochs=250 wandb.experiment_name={experiment_name}"
        # print(sbatch_command)
        # Submit the job using subprocess
        subprocess.call(sbatch_command, shell=True)
