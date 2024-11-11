import torch
import os
import cv2
import numpy as np
from data import load_data 
from torchvision.utils import save_image

def save_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """
    Saves the model and optimizer checkpoint.

    Args:
        model: The model to save.
        optimizer: The optimizer to save.
        filename: The file name where the checkpoint will be saved.
    """
    print("=> Saving checkpoint")
    torch.save({
        'state_dict': model.state_dict(),
        'optimizer': optimizer.state_dict(),
    }, filename)


def load_checkpoint(model, optimizer, filename="checkpoint.pth.tar"):
    """
    Loads a checkpoint into the model and optimizer.

    Args:
        model: The model to load the checkpoint into.
        optimizer: The optimizer to load the checkpoint into.
        filename: The file name of the checkpoint to load.
    """
    print("=> Loading checkpoint")
    checkpoint = torch.load(filename)
    model.load_state_dict(checkpoint['state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer'])


def get_loaders(image_height, image_width, batch_size, num_workers, pin_memory):
    """
    Returns data loaders for training, validation, and testing.

    Args:
        image_height (int): Image height for resizing.
        image_width (int): Image width for resizing.
        batch_size (int): Batch size for data loading.
        num_workers (int): Number of workers for data loading.
        pin_memory (bool): Whether to pin memory for faster data transfer.

    Returns:
        tuple: train_loader, val_loader, test_loader
    """
    # Assuming the function load_data is correctly importing and calling the TurtleDataset
    path = "dataSet/turtles-data/data/images"  # Adjust path as needed
    train_loader, val_loader, test_loader = load_data(path, batch_size, num_workers)
    return train_loader, val_loader, test_loader


def check_accuracy(loader, model, device="cuda"):
    """
    Check the accuracy of the model on the validation/test set.
    
    Args:
        loader: DataLoader for validation or test data.
        model: The model to evaluate.
        device: The device to use for evaluation, e.g., 'cuda' or 'cpu'.
    """
    model.eval()
    correct_pixels = 0
    total_pixels = 0

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)

            # Apply sigmoid and threshold for binary predictions
            predicted = (torch.sigmoid(outputs) > 0.5).float()

            # Count correctly predicted pixels
            correct_pixels += (predicted == targets).sum().item()
            total_pixels += targets.numel()

    # Calculate accuracy as percentage of correctly predicted pixels
    accuracy = 100 * correct_pixels / total_pixels
    print(f"Accuracy: {accuracy:.2f}%")
    model.train()

def save_predictions_as_imgs(loader, model, epoch, folder="saved_images", device="cuda"):
    """
    Save model predictions as images to a specified folder.

    Args:
        loader: DataLoader for validation data.
        model: The model to make predictions.
        epoch: Current epoch number.
        folder: Folder where the images will be saved.
        device: The device to use for predictions, e.g., 'cuda' or 'cpu'.
    """
    model.eval()
    os.makedirs(folder, exist_ok=True)

    with torch.no_grad():
        for batch_id, (data, targets) in enumerate(loader):
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            predictions = model(data)

            # Convert predictions to binary mask
            predicted_mask = torch.sigmoid(predictions) > 0.5

            # Save images and predictions
            for i in range(data.size(0)):
                save_image(predicted_mask[i], os.path.join(folder, f"epoch{epoch}_batch{batch_id}_image{i}_pred.png"))
                save_image(targets[i], os.path.join(folder, f"epoch{epoch}_batch{batch_id}_image{i}_target.png"))

    model.train()
