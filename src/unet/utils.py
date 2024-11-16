import torch
import torch.nn as nn
import torch.nn.functional as F

def save_checkpoint(model, optimizer, epoch, loss=None, filename="checkpoint.pth.tar"):
    """
    Save the model and optimizer states to a checkpoint file.

    Args:
        model (nn.Module): The model to save.
        optimizer (torch.optim.Optimizer): The optimizer to save.
        epoch (int, optional): The current epoch number (if resuming).
        loss (float, optional): The current loss value (if resuming).
        filename (str, optional): Path where to save the checkpoint.
    """
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer=None, device="cuda"):
    """
    Loads model and optimizer state from a checkpoint.

    Args:
        checkpoint_path (str): Path to the checkpoint file.
        model (nn.Module): The model to load weights into.
        optimizer (torch.optim.Optimizer, optional): Optimizer to load state into.
        device (str, optional): The device to load the model onto ('cuda' or 'cpu').

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

def calculate_val_loss(loader, model, device="cuda"):
    """
    Check the accuracy of the model on the validation/test set.

    Args:
        loader: DataLoader for validation or test data.
        model: The model to evaluate.
        device: The device to use for evaluation, e.g., 'cuda' or 'cpu'.
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
        
    avg_val_loss = val_loss/len(loader)
    print(f"Loss: {avg_val_loss:.2f}")
    model.train()


def save_model(model, filename="unet_model_trained.pth"):
    """
    Save only the model's state dict after training.

    Args:
        model (nn.Module): The trained model to save.
        filename (str, optional): Path where to save the model state dictionary.
    """
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")

def dice_loss(pred, target, smooth=1e-6):
    """
    Compute the Dice Loss between the predicted and target masks.
    Dice loss is 1 minus the Dice coefficient.
    """
    # Ensure target is of LongTensor type before applying one_hot
    target = target.long()
    pred = torch.softmax(pred, dim=1)
    # Convert target to one-hot encoding
    target_one_hot = torch.nn.functional.one_hot(target, num_classes=pred.size(1)).permute(0, 3, 1, 2).float()

    intersection = (pred * target_one_hot).sum(dim=(1, 2, 3))  # Sum over the spatial dimensions
    union = pred.sum(dim=(1, 2, 3)) + target_one_hot.sum(dim=(1, 2, 3))  # Sum over the spatial dimensions
    
    dice = (2. * intersection + smooth) / (union + smooth)  # Adding smoothing to avoid division by zero
    return 1 - dice.mean()  # Dice loss is 1 - Dice coefficient

def combined_loss(pred, target, weight_ce=1.0, weight_dice=1.0):
    """
    Compute the combined loss: weighted sum of Cross-Entropy Loss and Dice Loss.
    """
    # Cross-Entropy Loss
    ce_loss = F.cross_entropy(pred, target)  # [batch_size, height, width, num_classes] -> (pred), (target)
    
    # Dice Loss (target should be in the same format as the prediction)
    pred_softmax = F.softmax(pred, dim=1)  # Convert logits to probabilities for dice
    dice_loss_value = dice_loss(pred_softmax, target.float())  # target must be float for Dice
    
    # Combined loss (weighted sum of CE and Dice)
    loss = weight_ce * ce_loss + weight_dice * dice_loss_value
    return loss