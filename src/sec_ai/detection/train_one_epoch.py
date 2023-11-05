import torch
from torch.optim.optimizer import Optimizer
from torch.utils.data import DataLoader
from torch.utils.tensorboard.writer import SummaryWriter
from transformers import Mask2FormerForUniversalSegmentation
from sec_ai.settings import logging
import torch.nn.functional as F
import torchmetrics
from tqdm import tqdm


def train_one_epoch(
    model,
    optimizer: Optimizer,
    device: torch.device,
    dataloader: DataLoader,
    tensorboard_writer: SummaryWriter | None,
    epoch: int,
    frequency: int = 10,
    weight: list[int] = [0.2, 0.8],
):
    model.train()
    running_loss = 0
    num_samples = 0
    nb_batch = len(dataloader)

    accuracy = torchmetrics.Accuracy(task="multiclass", num_classes=2, average=None).to(device)
    precision = torchmetrics.Precision(task="multiclass", num_classes=2, average=None).to(device)
    recall = torchmetrics.Recall(num_classes=2, task="multiclass", average=None).to(device)

    for idx_batch, (slow_path_batch, fast_path_batch, labels) in tqdm(enumerate(dataloader)):
        optimizer.zero_grad()
        slow_path_batch, fast_path_batch, labels = (
            slow_path_batch.to(device),
            fast_path_batch.to(device),
            labels.to(device),
        )
        # Forward pass
        preds = model([slow_path_batch, fast_path_batch])  # batch size, nb_classe

        # Backward propagation
        loss = F.cross_entropy(preds, labels, weight=torch.tensor(weight).cuda())
        loss.backward()

        batch_size = slow_path_batch.size(0)
        running_loss += loss.item()
        num_samples += batch_size

        accuracy(preds, labels)
        precision(preds, labels)
        recall(preds, labels)

        if idx_batch % frequency == 0:
            progress = f"[{idx_batch}/{nb_batch}]"
            logging.info(f"{progress} Loss: {running_loss / num_samples}")

        # Optimization
        optimizer.step()
        step = epoch * len(dataloader) + idx_batch
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("loss_training", running_loss / num_samples, step)

    acc_value = accuracy.compute().cpu().numpy()
    precision_value = precision.compute().cpu().numpy()
    recall_value = recall.compute().cpu().numpy()

    print(f"Accuracy: {acc_value}")
    print(f"Precision: {precision_value}")
    print(f"Recall: {recall_value}")

    # Early stopping for debugging training loop
    return running_loss / num_samples


@torch.no_grad()
def validate_one_epoch(
    model,
    device: torch.device,
    dataloader: DataLoader,
    tensorboard_writer: SummaryWriter | None,
    epoch: float,
    frequency: int = 10,
) -> float:
    """Validation loop for Mask2former"""
    running_loss = 0
    num_samples = 0
    model.eval()
    logging.info("============ Start Validation ===========")
    nb_batch = len(dataloader)
    for idx_batch, (slow_path_batch, fast_path_batch, labels) in enumerate(dataloader):
        # Reset the parameter gradients

        # Forward pass
        preds = model([slow_path_batch, fast_path_batch])  # batch size, nb_classe

        # Backward propagation
        loss = F.cross_entropy(preds, labels)

        batch_size = slow_path_batch.size(0)
        running_loss += loss.item()
        num_samples += batch_size

        if idx_batch % frequency == 0:
            progress = f"[{idx_batch}/{nb_batch}]"
            logging.info(f"{progress} Loss: {running_loss / num_samples}")

        step = epoch * len(dataloader) + idx_batch
        if tensorboard_writer is not None:
            tensorboard_writer.add_scalar("loss_validation", running_loss / num_samples, step)

    validation_loss = running_loss / num_samples
    return validation_loss
