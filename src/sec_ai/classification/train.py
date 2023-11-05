import argparse
from pathlib import Path

from clearml import Task
from omegaconf import OmegaConf

from sec_ai.trainer import VideoTrainer
from sec_ai.settings import logging


if __name__ == "__main__":
    config_file_path = Path("/data/ubuntu/secai/src/sec_ai/config_resnet50.yaml")
    config = OmegaConf.load(config_file_path)

    trainer = VideoTrainer(config=config)
    model, optimizer = trainer.buil_model_and_optimizer()
    train_dataloader, val_dataloader = trainer.get_dataloaders(
        normal_train="/data/ubuntu/secai/normal_train.pickle",
        normal_val="/data/ubuntu/secai/normal_val.pickle",
        shoplift_train="/data/ubuntu/secai/shoplift_train.pickle",
        shoplift_val="/data/ubuntu/secai/shoplift_val.pickle",
    )
    trainer.train(
        model=model,
        optimizer=optimizer,
        train_dataloader=train_dataloader,
        val_dataloader=val_dataloader,
    )
