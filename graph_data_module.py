from torch_geometric.loader import DataLoader
import os.path as osp
import pytorch_lightning as pl
from transform_factory import factory as transforms
from datasets.dataset_factory import create_dataset

datasets_path = 'datasets'


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        if cfg.dataset.dataset_path is None:
            self.dataset_path = osp.join(datasets_path, cfg.dataset.name, 'data')
        else:
            self.dataset_path = cfg.dataset.dataset_path
        self.batch_size = cfg.train.batch_size      
        self.dataset_name = cfg.dataset.name          
        self.transform_dict = {}
        self.transform_dict['train'] = transforms(cfg.transform.train)
        self.transform_dict['validation'] = transforms(cfg.transform.validation)
        self.transform_dict['test'] = transforms(cfg.transform.test)
        self.num_workers=cfg.dataset.num_workers
        
    @property
    def num_classes(self):
        return create_dataset(
                dataset_path = self.dataset_path,
                dataset_name  = self.dataset_name,
                dataset_type = 'training',
                transform = self.transform_dict['train'],
                num_workers=self.num_workers
            ).num_classes

    def prepare_data(self):
        # Implement data download or preparation logic if needed
        pass

    def setup(self, stage=None):
        # Implement dataset splitting logic if necessary (train/val/test)
        pass

    def train_dataloader(self):
        # Return a DataLoader for the training dataset
        return DataLoader(
            create_dataset(
                dataset_path = self.dataset_path,
                dataset_name  = self.dataset_name,
                dataset_type = 'training',
                transform = self.transform_dict['train'],
                num_workers=self.num_workers
            ),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # Return a DataLoader for the validation dataset
        return DataLoader(
            create_dataset(
                dataset_path = self.dataset_path,
                dataset_name  = self.dataset_name,
                dataset_type = 'validation',
                transform = self.transform_dict['validation'],
                num_workers=self.num_workers
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )

    def test_dataloader(self):
        # Return a DataLoader for the test dataset
        return DataLoader(
            create_dataset(
                dataset_path = self.dataset_path,
                dataset_name = self.dataset_name,
                dataset_type = 'test',
                transform = self.transform_dict['test'],
                num_workers=self.num_workers
            ),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        )
