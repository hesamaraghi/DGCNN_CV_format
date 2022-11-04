from torch_geometric.loader import DataLoader
import os.path as osp
import sys
from datasets.transform_factory import factory as transforms

dir_path = osp.dirname(osp.realpath(__file__))
parent_dir_path = osp.dirname(dir_path)
parent_dir_path = osp.dirname(parent_dir_path)
datasets_path = osp.join(parent_dir_path,'datasets')
sys.path.append(datasets_path)

from nmnist import NMNIST

# def transforms(cfg):
#     return None

def nmnist_train_loader(cfg):
    if cfg.dataset.dataset_path is None:
        dataset_path = osp.join(datasets_path, cfg.dataset.name, 'data')
    else:
        dataset_path = cfg.dataset.dataset_path
        
    return DataLoader(
        NMNIST(
            dataset_path,
            transform=transforms(cfg.transform.train),
            train=True),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers
    )

def nmnist_test_loader(cfg):
    if cfg.dataset.dataset_path is None:
        dataset_path = osp.join(datasets_path, cfg.dataset.name, 'data')
    else:
        dataset_path = cfg.dataset.dataset_path
        
    return DataLoader(
        NMNIST(
            dataset_path,
            transform=transforms(cfg.transform.test),
            train=False),       
        batch_size=cfg.train.batch_size,
        num_workers=cfg.dataset.num_workers,
        shuffle=False
    )