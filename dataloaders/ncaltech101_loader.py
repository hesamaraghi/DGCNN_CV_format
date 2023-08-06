from torch_geometric.loader import DataLoader
import os.path as osp
from transform_factory import factory as transforms
from datasets.ncaltech101 import NCALTECH101

datasets_path = 'datasets'

def ncaltech101_train_loader(cfg):
    if cfg.dataset.dataset_path is None:
        dataset_path = osp.join(datasets_path, cfg.dataset.name, 'data')
    else:
        dataset_path = cfg.dataset.dataset_path
        
    return DataLoader(
        NCALTECH101(
            dataset_path,
            name = 'training',
            transform = transforms(cfg.transform.train)
        ),
        batch_size=cfg.train.batch_size,
        shuffle=True,
        num_workers=cfg.dataset.num_workers,
    )


def ncaltech101_validation_loader(cfg):
    if cfg.dataset.dataset_path is None:
        dataset_path = osp.join(datasets_path, cfg.dataset.name, 'data')
    else:
        dataset_path = cfg.dataset.dataset_path
        
    return DataLoader(
        NCALTECH101(
            dataset_path,
            name = 'validation',
            transform = transforms(cfg.transform.validation),
        ),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers
    )


def ncaltech101_test_loader(cfg):
    if cfg.dataset.dataset_path is None:
        dataset_path = osp.join(datasets_path, cfg.dataset.name, 'data')
    else:
        dataset_path = cfg.dataset.dataset_path
        
    return DataLoader(
        NCALTECH101(
            dataset_path,
            name = 'test',
            transform=transforms(cfg.transform.test),
        ),
        batch_size=cfg.train.batch_size,
        shuffle=False,
        num_workers=cfg.dataset.num_workers
    )