"""
PyTorch Lightning example code, designed for use in TU Delft CV lab.

Copyright (c) 2022 Robert-Jan Bruintjes, TU Delft.
"""
# Package imports, from conda or pip
import os
os.environ['WANDB__SERVICE_WAIT'] = '1000'
import glob 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor, TQDMProgressBar, ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import torchmetrics
import copy
import numpy as np

# Imports of own files
import model_factory
from graph_data_module import GraphDataModule
from utils import evaluate_manually


class Runner(pl.LightningModule):
    def __init__(self, cfg, model):
        super().__init__()
        self.cfg = cfg
        self.model = model
        self.loss_fn = eval(cfg.train.loss_fn)

        self.train_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes)
        self.val_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes)
        self.test_accuracy = torchmetrics.Accuracy(task='multiclass', num_classes=cfg.dataset.num_classes)

    def forward(self, x):
        # Runner needs to redirect any model.forward() calls to the actual
        # network
        return self.model(x)

    def configure_optimizers(self):
        if self.cfg.optimize.optimizer == 'Adam':
            optimizer = torch.optim.Adam(self.model.parameters(), lr=self.cfg.optimize.lr)
        else:
            raise NotImplementedError(f"Optimizer {self.cfg.optimizer}")
        
        if self.cfg.optimize.lr_scheduler == 'None':
            return optimizer
        elif self.cfg.optimize.lr_scheduler == 'CosineAnnealingLR':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.cfg.optimize.T_max)
        elif self.cfg.optimize.lr_scheduler == 'CosineAnnealingWarmRestarts':
            scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=self.cfg.optimize.T_0)
        elif self.cfg.optimize.lr_scheduler == 'ReduceLROnPlateau':
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode=self.cfg.optimize.mode, factor=self.cfg.optimize.factor, patience=self.cfg.optimize.patience) 
        else:
            raise NotImplementedError(f"Scheduler {self.cfg.optimize.lr_scheduler}")
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/multi_acc/mean",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def _step(self, batch):
        y = batch.y
        y_hat = self.model(batch)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.train_accuracy(preds, batch.y)

        # Log step-level loss & accuracy
        self.log("train/loss_step", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=len(batch.label), sync_dist=True)
        self.log("train/acc_step", self.train_accuracy, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=len(batch.label), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy(preds, batch.y)

        # Log step-level loss & accuracy
        self.log("val/loss_step", loss, on_step=False, on_epoch=True, logger=True, batch_size=len(batch.label), sync_dist=True)
        self.log("val/acc_step", self.val_accuracy, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=len(batch.label), sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.test_accuracy(preds, batch.y)

        # Log test loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=len(batch.label), sync_dist=True)
        self.log('test/acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=len(batch.label), sync_dist=True)
        return loss

    # def on_train_epoch_end(self):
    #     # Log the epoch-level training accuracy
    #     self.log('train/acc', self.train_accuracy.compute())
    #     self.train_accuracy.reset()

    # def on_validation_epoch_end(self):
    #     # Log the epoch-level validation accuracy
    #     self.log('val/acc', self.val_accuracy.compute())
    #     self.val_accuracy.reset()


class MultiValCallback(pl.Callback):
    def __init__(self,gdm,multi_val_test_num):
        super().__init__()
        self.gdm = gdm
        self.multi_val_test_num = multi_val_test_num
    def on_validation_epoch_end(self, trainer, pl_module):
        model = copy.deepcopy(pl_module.model)
        multi_val_acc = []
        for _ in range(self.multi_val_test_num):
            multi_val_acc.append(evaluate_manually(model,self.gdm.val_dataloader()))
        pl_module.log('val/multi_acc/mean', np.mean(multi_val_acc), sync_dist=True)
        pl_module.log('val/multi_acc/std', np.std(multi_val_acc), sync_dist=True)

def main():
    # Load defaults and overwrite by command-line arguments

    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(cmd_cfg.cfg_path)
    
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    cfg_bare = OmegaConf.load("config_bare.yaml")
    cfg = OmegaConf.merge(cfg_bare,cfg)
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Seed everything. Note that this does not make training entirely
    # deterministic.
    pl.seed_everything(cfg.seed, workers=True)


    # Create datasets using factory pattern
    gdm = GraphDataModule(cfg)
    cfg.dataset.num_classes = gdm.num_classes

    # Set cache dir to W&B logging directory
    os.environ["WANDB_CACHE_DIR"] = os.path.join(cfg.wandb.dir, 'cache')
    wandb_logger = WandbLogger(
        save_dir=cfg.wandb.dir,
        project=cfg.wandb.project,
        name=cfg.wandb.experiment_name,
        log_model='all' if cfg.wandb.log else None,
        offline=not cfg.wandb.log,
        # Keyword args passed to wandb.init()
        entity=cfg.wandb.entity,
        config=OmegaConf.to_object(cfg),
    )
    
    # Create model using factory pattern
    model = model_factory.factory(cfg)

    # Tie it all together with PyTorch Lightning: Runner contains the model,
    # optimizer, loss function and metrics; Trainer executes the
    # training/validation loops and model checkpointing.
    runner = Runner(cfg, model)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    bar = TQDMProgressBar(refresh_rate=100)
    checkpoint_callback = ModelCheckpoint(monitor="val/multi_acc/mean", mode="max")
    multi_val_callback = MultiValCallback(gdm,cfg.train.multi_val_test_num)
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        enable_progress_bar=True,
        # Use DDP training by default, even for CPU training
        strategy="ddp_find_unused_parameters_false",
        devices=torch.cuda.device_count(),
        accelerator="auto",
        callbacks=[lr_monitor,bar,checkpoint_callback,multi_val_callback],
        profiler=cfg.train.profiler
    )
    

    
    # torch.set_num_threads(1)
    print(f"Number of threads: {torch.get_num_threads()}")

    # Train + validate (if validation dataset is implemented)
    trainer.fit(model = runner, datamodule=gdm)

    # Test (if test dataset is implemented)
    if gdm.test_dataloader is not None:
        trainer.test(datamodule=gdm)

    model = copy.deepcopy(runner.model)
    multi_test_acc = []
    for _ in range(cfg.train.multi_val_test_num):
        multi_test_acc.append(evaluate_manually(model,gdm.test_dataloader()))
    trainer.logger.experiment.log({'test/multi_acc/mean': np.mean(multi_test_acc)})
    trainer.logger.experiment.log({'test/multi_acc/std': np.std(multi_test_acc)})



if __name__ == '__main__':
    main()