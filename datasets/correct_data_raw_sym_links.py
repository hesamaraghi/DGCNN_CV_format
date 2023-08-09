'''
You should be in an unusual situation to use this code!
Use this code if your sym links in 'data/raw' folder are corrupted (refering to wrong locations).
It reads the sym links in 'data/raw' and redirect them to the coresponding locations in 'downloaded' folder.
NOTE: if the 'data/raw' folder is not in its corrent structure this script may result
in running errors.
'''


# %%

import os
import shutil
import argparse
import os.path as osp
from glob import glob
import numpy as np

# %%

download_folder = {'NCALTECH101':   'Caltech101',
                   'NASL':          ''}

def correct_sym_links(dataset_name):
    assert dataset_name in download_folder.keys(), f"{dataset_name} is not in {download_folder.keys()}"
    raw_path = osp.join(osp.dirname(osp.abspath(__file__)),dataset_name,'data','raw')
    src_path = osp.join(osp.dirname(osp.abspath(__file__)),dataset_name,'downloaded',download_folder[dataset_name])
    assert osp.exists(src_path), f"'download' directory for {dataset_name} dataset is corrupted!"
    all_path = os.path.join(raw_path, 'all')

    # %%
    try:
        os.symlink(src_path,all_path)
    except FileExistsError:
        os.unlink(all_path)
        os.symlink(src_path,all_path)

    # %%

    for folder in ['training', 'validation', 'test']:
        for f in glob(os.path.join(raw_path, folder, '*','*')):
            src = osp.join(src_path,*f.split(os.sep)[-2:])
            try:
                os.symlink(src,f)
            except FileExistsError:
                os.unlink(f)
                os.symlink(src,f)


if __name__ == '__main__':
    # input arguments
    parser = argparse.ArgumentParser(description='get the dataset name')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='name of the dataset')

    args = parser.parse_args()
    correct_sym_links(args.dataset_name)
    print("Good bye!")
