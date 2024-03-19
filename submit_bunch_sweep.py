import subprocess

import os
import argparse

def main():

    for _ in range(args.num_run):
        
        sbatch_command = f"sbatch {args.sbatch_folder}/run_train.sbatch wandb agent --count {args.agent_count} {args.sweep_id}"
        print(sbatch_command)
        # Submit the job using subprocess
        subprocess.call(sbatch_command, shell=True)

if __name__ == "__main__":
    
    # Define the paths
    os.environ["WANDB_CACHE_DIR"] = os.path.join('.', 'cache')
    
    parser = argparse.ArgumentParser(description="Your script description.")
    
    parser.add_argument("-n", "--num-run", type=int, default=1, help="number of runs")
    parser.add_argument("--sbatch-folder", type=str, default="sbatch_folder")
    parser.add_argument("--agent-count", type=int, default=1, help="number of agents per call")
    parser.add_argument("--sweep-id", type=str, help="sweep id")
 
    args = parser.parse_args()

    main()
