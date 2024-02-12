from glob import glob
import os
import argparse
import wandb
import shutil


def main():
    api = wandb.Api()

    runs = api.runs(args.folder)
    valid_run_ids = [r.id for r in runs]
    print(f"Number of runs in wandb server: {len(valid_run_ids)}")
    checkpoint_folders = glob(os.path.join(args.folder, '*'))
    print(f"Number of folders in local: {len(checkpoint_folders)}")

    empty_folders = []
    for folder in checkpoint_folders:
        if not os.listdir(folder):
            empty_folders.append(folder)

    print("Empty folders:")
    print(empty_folders)

    for folder in empty_folders:
        os.rmdir(os.path.join(args.folder, folder))
            
    not_in_wandb = []
    for folder in checkpoint_folders:
        folder = folder.split(os.sep)[-1]
        if folder not in valid_run_ids:
            not_in_wandb.append(folder)
     
    print(f"Number of folders not in wandb: {len(not_in_wandb)}")        
    print("Not in wandb:")
    print(not_in_wandb)
    for folder in not_in_wandb:
        # shutil.rmtree(os.path.join(args.folder, folder))
        print(f"Removed {folder}")


    

    
    




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Your script description.")

    parser.add_argument("--folder", type=str, required=True, help="folder to clear")
    args = parser.parse_args()

    main()