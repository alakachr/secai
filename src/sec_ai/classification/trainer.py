from sec_ai.VideoDataset import VideoDataset, collate_fn
from sec_ai.BalancedVideoDataset import BalancedVideoDataset, collate_fn
from torch.utils.data import DataLoader
from omegaconf import DictConfig
import torch
from sec_ai.settings import logging
from sec_ai.train_one_epoch import train_one_epoch, validate_one_epoch
from torch.utils.tensorboard.writer import SummaryWriter
from pathlib import Path


class VideoTrainer(object):
    def __init__(self, config: DictConfig):
        self.config = config
        self.device = torch.device("cuda")
        self.writer = SummaryWriter(log_dir=self.config.TRAINING.TENSORBOARD_DIR)

    def get_dataloaders(self, normal_train, normal_val, shoplift_train, shoplift_val):
        train_dataset = BalancedVideoDataset(
            normal_paths=normal_train, shoplift_paths=shoplift_train
        )
        val_dataset = BalancedVideoDataset(normal_paths=normal_val, shoplift_paths=shoplift_val)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.NUM_THREADS,
        )
        val_dataloader = DataLoader(
            val_dataset,
            batch_size=self.config.BATCH_SIZE,
            shuffle=True,
            collate_fn=collate_fn,
            num_workers=self.config.NUM_THREADS,
        )

        return train_dataloader, val_dataloader

    def buil_model_and_optimizer(self):
        # Choose the `slowfast_r50` model
        model = torch.hub.load("facebookresearch/pytorchvideo", "slowfast_r50", pretrained=True)

        modules = [module for module in model.children()]
        modules[-1][-1].proj = torch.nn.Linear(
            in_features=2304, out_features=self.config.MODEL.NB_CLASSES, bias=True
        )
        params = [p for p in model.parameters() if p.requires_grad]

        optimizer = torch.optim.Adam(
            lr=self.config.TRAINING.LEARNING_RATE, weight_decay=0.0005, params=params
        )
        return model, optimizer

    def train(self, model, optimizer, train_dataloader, val_dataloader):
        config_train = self.config.TRAINING

        model.to(self.device)

        max_current_map = 0
        for epoch in range(config_train.NUM_EPOCHS):
            logging.info(f"Epoch: {epoch}")
            train_one_epoch(
                model,
                optimizer,
                device=self.device,
                dataloader=train_dataloader,
                tensorboard_writer=self.writer,
                epoch=epoch,
                weight=self.config.TRAINING.LOSS_WEIGHTS,
            )
            # evaluate and save best model
            # val_loss, max_current_map = self.validate_and_evaluate(
            #     model,
            #     optimizer=optimizer,
            #     val_dataloader=val_dataloader,
            #     rcnn_val_dataloader=rcnn_val_dataloader,
            #     epoch=epoch,
            #     max_current_map=max_current_map,
            #     debug=debug,
            # )
            val_loss = 0

            # save every frequency of epoch
            if epoch % config_train.SAVE_FREQUENCY == 0:
                classes = ["normal", "shoplifing"]
                self.save_checkpoint(
                    model=model,
                    optimizer=optimizer,
                    epoch=epoch,
                    path=Path(self.config.TRAINING.PATH_MODEL) / f"ep{epoch}_model.pt",
                    augment=None,
                    classes=classes,
                    num_classes=self.config.MODEL.NB_CLASSES,
                    config=self.config,
                )

            # learning rate
            learning_rate = optimizer.param_groups[0]["lr"]
            if self.writer is not None:
                self.writer.add_scalar("learning_rate", learning_rate, global_step=epoch)

    @staticmethod
    def save_checkpoint(model, optimizer, epoch, path, augment, num_classes, classes, config):
        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "augmentations": augment,
                "classes": classes,
                "num_classes": num_classes,
                "config": config,
            },
            path,
        )
