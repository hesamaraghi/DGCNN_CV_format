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
                   'NASL':          '',
                   'NCARS':         ['n-cars_test', 'n-cars_train']}

def correct_sym_links(dataset_name):
    assert dataset_name in download_folder.keys(), f"{dataset_name} is not in {download_folder.keys()}"
    raw_path = osp.join(osp.dirname(osp.abspath(__file__)),dataset_name,'data','raw')
    all_path = os.path.join(raw_path, 'all')
    if isinstance(download_folder[dataset_name],list):  
        src_path_list = [osp.join(osp.dirname(osp.abspath(__file__)),dataset_name,'downloaded',folder) for folder in download_folder[dataset_name]]   
        assert all([osp.exists(src_path) for src_path in src_path_list]), f"'download' directory for {dataset_name} dataset is corrupted!"

    ## creating the 'all' folder
        
        if osp.exists(all_path):
            shutil.rmtree(all_path)
        os.mkdir(all_path)
        
        for src_path in src_path_list:
            for f in glob(os.path.join(src_path, '*','*')):
                class_name = f.split(os.sep)[-2]
                if not osp.exists(os.path.join(all_path, class_name)):
                    os.mkdir(os.path.join(all_path, class_name))
                dst_path = os.path.join(all_path, class_name, f.split(os.sep)[-1])
                try:
                    os.symlink(f,dst_path)
                except FileExistsError:
                    os.unlink(dst_path)
                    os.symlink(f,dst_path)
                    
        for folder in ['training', 'validation', 'test']:
            for f in glob(os.path.join(raw_path, folder, '*','*')):
                all_corresponding_path = osp.join(all_path,*f.split(os.sep)[-2:])
                src = os.readlink(all_corresponding_path)
                try:
                    os.symlink(src,f)
                except FileExistsError:
                    os.unlink(f)
                    os.symlink(src,f)
                    
    elif isinstance(download_folder[dataset_name],str):
        src_path = osp.join(osp.dirname(osp.abspath(__file__)),dataset_name,'downloaded',download_folder[dataset_name])
        assert osp.exists(src_path), f"'download' directory for {dataset_name} dataset is corrupted!"
        try:
            os.symlink(src_path,dst_path)
        except FileExistsError:
            os.unlink(dst_path)
            os.symlink(src_path,dst_path)
            
        for folder in ['training', 'validation', 'test']:
            for f in glob(os.path.join(raw_path, folder, '*','*')):
                src = osp.join(src_path,*f.split(os.sep)[-2:])
                try:
                    os.symlink(src,f)
                except FileExistsError:
                    os.unlink(f)
                    os.symlink(src,f)
            
    else:
        raise "download_folder values should be a list of strings or a string!"

    # %%




if __name__ == '__main__':
    # input arguments
    parser = argparse.ArgumentParser(description='get the dataset name')
    parser.add_argument('--dataset-name', type=str, required=True,
                        help='name of the dataset')

    args = parser.parse_args()
    correct_sym_links(args.dataset_name)
    print("Good bye!")
