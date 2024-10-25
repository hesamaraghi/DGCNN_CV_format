import argparse
import wandb
from utils.config_utils import get_checkpoint_file, get_config_file
from graph_data_module import GraphDataModule
import model_factory
from train import Runner
import torch
import pytorch_lightning as pl

def main():
    
    api = wandb.Api()
    run = api.run(args.run)
    cfg,_ = get_config_file(*args.run.split("/"), verbose = True)
    cfg.dataset.dataset_path = f'datasets_torch_geometric/{cfg.dataset.name}'
    cfg.dataset.in_memory = False
    cfg.train.multi_test_num = 1
    ckpt = get_checkpoint_file(*args.run.split("/"), remote_root = args.root)
    gdm = GraphDataModule(cfg)
    cfg.dataset.num_classes = gdm.num_classes
    model = model_factory.factory(cfg)
    runner = Runner.load_from_checkpoint(ckpt, cfg=cfg, model=model)
    
    trainer = pl.Trainer(
        enable_progress_bar=True,
        # Use DDP training by default, even for CPU training
        # strategy="ddp_notebook",
        devices=torch.cuda.device_count(),
        accelerator="auto"
    )
    trainer.test(runner, datamodule=gdm)


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Arguments for testing inference deterministic.")
    
    parser.add_argument("--run", type=str, help="The experiments to test.")
    parser.add_argument("--root", type=str, required=True, help="The remote root path.")
    
    args = parser.parse_args()
    
    main()
    