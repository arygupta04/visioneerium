import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from typing import Optional


def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: Optional[int] = None,
    loss: Optional[float] = None,
    filename: str = "checkpoint.pth.tar",
):
    """
    Save the model and optimizer states to a checkpoint file.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int, optional): The current epoch number (if resuming).
        loss (float, optional): The current loss value (if resuming).
        filename (str): Path where to save the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(
    checkpoint_path: str,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    device: str = "cuda",
):
    """
    Loads model and optimizer state from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (str): The device to load the model onto ('cuda' or 'cpu').

    Returns:
        model (nn.Module): Model with loaded weights.
        optimizer (torch.optim.Optimizer, optional): Optimizer with loaded state.
        epoch (int, optional): The epoch number to resume from.
        loss (float, optional): Loss at the checkpoint (if needed).
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    return model, optimizer, epoch, loss


def calculate_val_loss(loader: DataLoader, model: nn.Module, device: str = "cuda"):
    """
    Check the accuracy of the model on the validation/test set.

    Args:
        loader (DataLoader): DataLoader for the validation/test set.
        model (nn.Module): The trained model to evaluate.
        device (str): The device to load the model onto ('cuda' or 'cpu').
    """
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets.long())

            val_loss += loss.item() * data.size(0)

    avg_val_loss = val_loss / len(loader)
    print(f"Loss: {avg_val_loss:.2f}")


def save_model(model: nn.Module, filename: str = "unet_model_trained.pth"):
    """
    Save only the model's state dict after training.

    Args:
        model (nn.Module): The trained model to save.
        filename (str): Path where to save the model state dictionary.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")


def dice_loss(
    pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-6
) -> torch.Tensor:
    """
    Compute the Dice Loss between the predicted and target masks.
    Dice loss is 1 minus the Dice coefficient.

    Args:
        pred (torch.Tensor): Predicted mask from the model.
        target (torch.Tensor): Ground truth mask.
        smooth (float): Smoothing factor to avoid division by zero.

    Returns:
        torch.Tensor: Dice loss value.
    """
    target = target.long()
    pred = torch.softmax(pred, dim=1)  # Convert logits to probabilities

    # Convert target to one-hot encoding
    target_one_hot = (
        torch.nn.functional.one_hot(target, num_classes=pred.size(1))
        .permute(0, 3, 1, 2)
        .float()
    )

    intersection = (pred * target_one_hot).sum(dim=(1, 2, 3))
    union = pred.sum(dim=(1, 2, 3)) + target_one_hot.sum(dim=(1, 2, 3))

    dice_coeff = (2.0 * intersection + smooth) / (
        union + smooth
    )  # Smoothing to avoid division by zero

    return 1 - dice_coeff.mean()


def combined_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    weight_ce: float = 0.5,
    weight_dice: float = 0.5,
) -> torch.Tensor:
    """
    Compute the combined loss: weighted sum of Cross-Entropy Loss and Dice Loss.

    Args:
        pred (torch.Tensor): Predicted mask from the model.
        target (torch.Tensor): Ground truth mask.
        weight_ce (float): Weight for the Cross-Entropy Loss.
        weight_dice (float): Weight for the Dice Loss.

    Returns:
        torch.Tensor: Combined loss value.
    """
    # Compute losses
    ce_loss = F.cross_entropy(pred, target)
    dice_loss_value = dice_loss(pred, target.float())

    # Weighted sum of the two losses
    loss = weight_ce * ce_loss + weight_dice * dice_loss_value

    return loss
