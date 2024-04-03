from torch_geometric.loader import DataLoader
import os.path as osp
import pytorch_lightning as pl
from transform_factory import factory as transforms
from datasets_torch_geometric.dataset_factory import create_dataset

datasets_path = 'datasets_torch_geometric'


class GraphDataModule(pl.LightningDataModule):
    def __init__(self, cfg):
        super().__init__()
        if cfg.dataset.dataset_path is None:
            self.dataset_path = osp.join(datasets_path, cfg.dataset.name, 'data')
        else:
            self.dataset_path = cfg.dataset.dataset_path
        self.batch_size = cfg.train.batch_size      
        self.dataset_name = cfg.dataset.name
        self.multi_val_test_num = cfg.train.multi_val_test_num      
        self.transform_dict = {}
        self.transform_dict['train'] = transforms(cfg.transform.train)
        self.transform_dict['validation'] = transforms(cfg.transform.validation)
        self.transform_dict['test'] = transforms(cfg.transform.test)
        self.pre_transform_dict = {'train': None, 'validation': None, 'test': None}
        if cfg.pre_transform.train.transform == True:
            self.pre_transform_dict['train'] = transforms(cfg.pre_transform.train)
        if cfg.pre_transform.validation.transform == True:
            self.pre_transform_dict['validation'] = transforms(cfg.pre_transform.validation)
        if cfg.pre_transform.test.transform == True:
            self.pre_transform_dict['test'] = transforms(cfg.pre_transform.test)
        self.num_workers=cfg.dataset.num_workers
        self.training_dataset = create_dataset(
                dataset_path = self.dataset_path,
                dataset_name  = self.dataset_name,
                dataset_type = 'training',
                transform = self.transform_dict['train'],
                pre_transform=self.pre_transform_dict['train'],
                num_workers=self.num_workers
            )
        self.num_classes = self.training_dataset.num_classes
        print(f"Number of classes: {self.num_classes}", flush=True)
        self.validation_dataset = create_dataset(
                dataset_path = self.dataset_path,
                dataset_name  = self.dataset_name,
                dataset_type = 'validation',
                transform = self.transform_dict['validation'],
                pre_transform=self.pre_transform_dict['validation'],
                num_workers=self.num_workers
            )
        self.test_dataset = create_dataset(
                dataset_path = self.dataset_path,
                dataset_name  = self.dataset_name,
                dataset_type = 'test',
                transform = self.transform_dict['test'],
                pre_transform=self.pre_transform_dict['test'],
                num_workers=self.num_workers
            )
    def prepare_data(self):
        # Implement data download or preparation logic if needed
        pass

    def setup(self, stage=None):
        # Implement dataset splitting logic if necessary (train/val/test)
        pass

    def train_dataloader(self):
        # Return a DataLoader for the training dataset
        return DataLoader(
            self.training_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
        )

    def val_dataloader(self):
        # Return a DataLoader for the validation dataset
        return [DataLoader(
            self.validation_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        ) for _ in range(self.multi_val_test_num)]

    def test_dataloader(self):
        # Return a DataLoader for the test dataset
        return [DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
        ) for _ in range(self.multi_val_test_num)]
