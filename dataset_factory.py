from datasets.nmnist_ import nmnist_train_loader, nmnist_test_loader
from datasets.ncaltech101_ import ncaltech101_train_loader, ncaltech101_validation_loader, ncaltech101_test_loader
# from datasets.nasl_ import NASL


def factory(cfg):
    """
    Returns:
        train_loader: Training dataset loader.
        val_loader: Validation dataset loader. Validation is performed after
            each training epoch. If None, no validation is performed.
        test_loader: Test dataset loader. Testing is performed after fitting is
            done. If None, no testing is performed.
    """
    if cfg.dataset.name == 'NMNIST':
        cfg.dataset.num_classes = nmnist_train_loader(cfg).dataset.num_classes
        return nmnist_train_loader(cfg), nmnist_test_loader(cfg), None
    if cfg.dataset.name == 'NCALTECH101':
        cfg.dataset.num_classes = ncaltech101_train_loader(cfg).dataset.num_classes
        return ncaltech101_train_loader(cfg), ncaltech101_validation_loader(cfg), ncaltech101_test_loader(cfg)
    else:
        raise NotImplementedError(f"Dataset {cfg.dataset.name}")