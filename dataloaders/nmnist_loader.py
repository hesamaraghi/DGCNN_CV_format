from torch_geometric.loader import DataLoader
import os.path as osp
from transform_factory import factory as transforms
from datasets.nmnist import NMNIST

datasets_path = 'datasets'

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