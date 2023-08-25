from datasets.nmnist import NMNIST
from datasets.ncaltech101 import NCALTECH101
from datasets.nasl import NASL
from datasets.ncars import NCARS
import os

def create_dataset(dataset_path = None, 
                   dataset_name = None, 
                   dataset_type = None,
                   transform = None, 
                   num_workers = 4):
    dataset_path = os.path.abspath(dataset_path)
    
    if dataset_name == "NMNIST":
        return NMNIST(  root = dataset_path, 
                        name = dataset_type, 
                        transform = transform,
                        num_workers = num_workers)
    elif dataset_name == "NCALTECH101":
        return NCALTECH101( root = dataset_path,
                            name = dataset_type,
                            transform = transform,
                            num_workers = num_workers)
    elif dataset_name == "NASL":
        return NASL(    root = dataset_path,
                        name = dataset_type,
                        transform = transform,
                        num_workers = num_workers)
    elif dataset_name == "NCARS":
        return NCARS(   root = dataset_path,
                        name = dataset_type,
                        transform = transform,
                        num_workers = num_workers)
    # Add more dataset options here as needed
    else:
        raise ValueError("Invalid dataset name")

