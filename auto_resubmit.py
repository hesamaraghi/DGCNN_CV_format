import subprocess
import wandb
import os
import argparse
import re
from datetime import datetime, timedelta
import time
import signal
from glob import glob

def extract_cfg_path(job_id):
    try:
        # Run the scontrol command to get detailed information about the job
        output = subprocess.check_output(['scontrol', 'show', 'job', job_id], universal_newlines=True)

        # Use regular expression to find the cfg_path option in the Command field
        match = re.search(r'Command\s*=\s*(.+cfg_path=(.+)\n.+)', output)
        if match:
            cfg_path = match.group(2).strip()
            return cfg_path
        else:
            print("Error: cfg_path option not found in scontrol output.")
            return None
    except subprocess.CalledProcessError:
        # Handle errors when executing scontrol
        print(f"Error: Unable to retrieve information for job {job_id}.")
        return None


def get_jobids(flag = None):
    if flag is None:
        status_list = []
    elif flag == 'PD':
        status_list = ['-t', 'PD']
    elif flag == 'R':
        status_list = ['-t', 'R']
    else:
        raise ValueError("Invalid flag")

    try:
        # Run the squeue command to get information about pending jobs
        output = subprocess.check_output(['squeue', '-u', 'maraghi'] + status_list, universal_newlines=True)
        # Split the output into lines
        lines = output.strip().split('\n')
        # Skip the header line
        lines = lines[1:]
        # Extract job IDs
        job_ids = [line.split()[0] for line in lines]
        return job_ids
    except subprocess.CalledProcessError:
        # Handle errors when executing squeue
        print("Error: Unable to retrieve pending jobs.")
        return []

def main():
    while 1:  
        pending_jobs = get_jobids('PD')
        running_jobs = get_jobids('R')
        run_names_dict = {"running": {}, "pending": {}}
        for job_id in pending_jobs:
            cfg_path_str = extract_cfg_path(job_id)
            if cfg_path_str:
                run_names_dict['pending'][job_id] = (cfg_path_str.split('/')[-2])
        for job_id in running_jobs:
            cfg_path_str = extract_cfg_path(job_id)
            if cfg_path_str:
                run_names_dict['running'][job_id] = (cfg_path_str.split('/')[-2])      
        runs = wandb.Api().runs(args.project_name)
        if args.sweep_name:
            runs = [r for r in runs if r.sweep.name == args.sweep_name]
        if args.tags:
            tags = args.tags.split(",")
            runs = [r for r in runs if all(t in r.tags for t in tags)]
        for run in runs:
            if "resume_config" in run.config:
                if 'test/acc_mean' not in run.summary:
                    if os.path.exists(os.path.join('pl_default_dir',args.project_name,run.name,'config.yml')):
                        if run.state == "running" or run.state == "finished":
                            if run.name not in run_names_dict['running'].values() and run.name not in run_names_dict['pending'].values():
                                sbatch_command = f"bash {args.autoresume_file} \"{args.sbatch_file} python train.py cfg_path={run.config['resume_config']['cfg_path']}\" {args.num_repeat}"
                                print("no test acc.: ",run.name, sbatch_command)
                                # Submit the job using subprocess
                                if not args.dry_run:
                                    subprocess.call(sbatch_command, shell=True)
                else:
                    if os.path.exists(run.config['train']['default_root_dir']):
                        hpc_ckpts_list = glob(os.path.join(run.config['train']['default_root_dir'], 'hpc_ckpt_*.ckpt'))
                        if hpc_ckpts_list:
                            hpc_ckpt_max = max([h.split('_')[-1].split('.')[0] for h in hpc_ckpts_list])
                            if run.name not in run_names_dict['running'].values() and run.name not in run_names_dict['pending'].values():
                                sbatch_command = f"bash {args.autoresume_file} \"{args.sbatch_file} python train.py cfg_path={run.config['resume_config']['cfg_path']}\" {args.num_repeat}"
                                print(f"resume from hpc_ckpt_{hpc_ckpt_max}", run.name, sbatch_command)
                                # Submit the job using subprocess
                                if not args.dry_run:
                                    subprocess.call(sbatch_command, shell=True)
        
        
        dt = datetime.now() + timedelta(hours=1)
        dt = dt.replace(minute=12,second=34)

        print(f"Sleeping until {dt}")

        while datetime.now() < dt:
            time.sleep(1)

def signal_handler(sig, frame):
    print('Signal received:', sig)
    # You can add your custom handling logic here
    # For example, you can gracefully exit the program
    main()


if __name__ == "__main__":
    
    # Define the paths
    os.environ["WANDB_CACHE_DIR"] = os.path.join('.', 'cache')
    
    parser = argparse.ArgumentParser(description="Your script description.")

    parser.add_argument("--project-name", type=str)
    parser.add_argument("--autoresume-file", type=str, default="sbatch_folder/auto_resume.sh")
    parser.add_argument("--num-repeat", type=int, default=1, help="number of repeats (dependent jobs) in auto resume")
    parser.add_argument("--sbatch-file", type=str, default="sbatch_folder/run_train.sbatch")
    parser.add_argument("--sweep-name", type=str)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("tags", nargs="?", help="List of tags separated by commas")

    
    args = parser.parse_args()

    signal.signal(signal.SIGUSR1, signal_handler)
    main()
