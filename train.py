"""
PyTorch Lightning example code, designed for use in TU Delft CV lab.

Copyright (c) 2022 Robert-Jan Bruintjes, TU Delft.
"""
# Package imports, from conda or pip
import os
import glob 
import torch
import torch.nn as nn
import pytorch_lightning as pl
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import WandbLogger
from omegaconf import OmegaConf
import torchmetrics

# Imports of own files
import model_factory
import dataset_factory


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
                "monitor": "val/loss_step",
                "frequency": 1
                # If "monitor" references validation metrics, then "frequency" should be set to a
                # multiple of "trainer.check_val_every_n_epoch".
            },
        }

    def _step(self, batch):
        y = batch.y
        y_hat,_ = self.model(batch)
        loss = self.loss_fn(y_hat, y)
        return loss, y_hat

    def training_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.train_accuracy(preds, batch.y)

        # Log step-level loss & accuracy
        self.log("train/loss_step", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=self.cfg.train.batch_size, sync_dist=True)
        self.log("train/acc_step", self.train_accuracy, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=self.cfg.train.batch_size, sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.val_accuracy(preds, batch.y)

        # Log step-level loss & accuracy
        self.log("val/loss_step", loss, on_step=False, on_epoch=True, logger=True, batch_size=self.cfg.train.batch_size, sync_dist=True)
        self.log("val/acc_step", self.val_accuracy, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=self.cfg.train.batch_size, sync_dist=True)
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat = self._step(batch)
        preds = torch.argmax(y_hat, dim=1)
        self.test_accuracy(preds, batch.y)

        # Log test loss
        self.log("test/loss", loss, on_step=False, on_epoch=True, logger=True, prog_bar=True, batch_size=self.cfg.train.batch_size, sync_dist=True)
        self.log('test/acc', self.test_accuracy, on_step=False, on_epoch=True, prog_bar=True, logger=True, batch_size=self.cfg.train.batch_size, sync_dist=True)
        return loss

    # def on_train_epoch_end(self):
    #     # Log the epoch-level training accuracy
    #     self.log('train/acc', self.train_accuracy.compute())
    #     self.train_accuracy.reset()

    # def on_validation_epoch_end(self):
    #     # Log the epoch-level validation accuracy
    #     self.log('val/acc', self.val_accuracy.compute())
    #     self.val_accuracy.reset()


def main():
    # Load defaults and overwrite by command-line arguments

    cmd_cfg = OmegaConf.from_cli()
    cfg = OmegaConf.load(cmd_cfg.cfg_path)
    
    cfg = OmegaConf.merge(cfg, cmd_cfg)
    print(OmegaConf.to_yaml(cfg))

    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    # Seed everything. Note that this does not make training entirely
    # deterministic.
    pl.seed_everything(cfg.seed, workers=True)

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
    


    # Create datasets using factory pattern
    loaders = dataset_factory.factory(cfg)
    train_dataset_loader, val_dataset_loader, test_dataset_loader = loaders
    

    # Create model using factory pattern
    model = model_factory.factory(cfg)

    # Tie it all together with PyTorch Lightning: Runner contains the model,
    # optimizer, loss function and metrics; Trainer executes the
    # training/validation loops and model checkpointing.
    runner = Runner(cfg, model)
    
    lr_monitor = LearningRateMonitor(logging_interval='step')
    
    trainer = pl.Trainer(
        max_epochs=cfg.train.epochs,
        logger=wandb_logger,
        enable_progress_bar=False,
        # Use DDP training by default, even for CPU training
        strategy="ddp_find_unused_parameters_false",
        devices=torch.cuda.device_count(),
        accelerator="auto",
        callbacks=[lr_monitor],
        profiler=cfg.train.profiler,
    )

    # Train + validate (if validation dataset is implemented)
    trainer.fit(model = runner, train_dataloaders = train_dataset_loader, val_dataloaders = val_dataset_loader)

    # Test (if test dataset is implemented)
    if test_dataset_loader is not None:
        trainer.test(runner, test_dataset_loader)




if __name__ == '__main__':
    main()