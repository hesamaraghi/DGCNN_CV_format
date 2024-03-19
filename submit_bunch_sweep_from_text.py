import subprocess

import os
import argparse

def main():
    
    with open(args.sweep_file, "r") as f:
        for line in f:       
            sbatch_command = f"bash {args.autoresume_file} \"{args.sbatch_file} {line.strip()}\" {args.num_repeat}"
            print(sbatch_command)
            # Submit the job using subprocess
            subprocess.call(sbatch_command, shell=True)

        
    
 

if __name__ == "__main__":
    
    # Define the paths
    os.environ["WANDB_CACHE_DIR"] = os.path.join('.', 'cache')
    
    parser = argparse.ArgumentParser(description="Your script description.")

    parser.add_argument("--sbatch-file", type=str, default="sbatch_folder/run_train.sbatch")
    parser.add_argument("--sweep-file", type=str)
    parser.add_argument("--autoresume-file", type=str, default="sbatch_folder/auto_resume.sh")
    parser.add_argument("--num-repeat", type=int, default=2, help="number of repeats (dependent jobs) in auto resume")
    
    
    args = parser.parse_args()

    main()
