import torch
import os
import numpy as np
from src.unet.data import load_data
from torchvision.utils import save_image
from PIL import Image
import torch.nn as nn

from simplecrf import CRF
import numpy as np


def apply_simple_crf(original_image, mask):
    """
    Applies CRF post-processing using SimpleCRF to refine the segmentation mask.
    :param original_image: The original image (H, W, 3)
    :param mask: The predicted mask (classes, H, W)
    :return: Refined mask
    """
    # Convert the mask to a probability map (if not already in probability format)
    mask = torch.sigmoid(mask) if isinstance(mask, torch.Tensor) else mask
    mask = mask.cpu().detach().numpy() if isinstance(mask, torch.Tensor) else mask
    mask = np.moveaxis(mask, 0, -1)  # Change shape from (C, H, W) to (H, W, C)

    # Convert image to the required format
    original_image = (
        original_image.cpu().detach().numpy()
        if isinstance(original_image, torch.Tensor)
        else original_image
    )
    original_image = (
        np.uint8(original_image * 255)
        if original_image.max() <= 1.0
        else np.uint8(original_image)
    )
    original_image = (
        np.moveaxis(original_image, 0, -1)
        if original_image.shape[0] == 3
        else original_image
    )

    # Apply CRF with SimpleCRF
    crf = CRF(original_image, mask)
    refined_mask = crf.run(n_iter=5)  # Number of iterations

    # Convert refined mask back to the original shape
    refined_mask = np.argmax(refined_mask, axis=-1)
    return refined_mask


def save_checkpoint(
    model, optimizer, epoch=None, loss=None, filename="checkpoint.pth.tar"
):
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
    path = r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\images"  # Adjust path as needed
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
    # correct = 0
    # total = 0
    normal_val_loss = 0.0
    crf_val_loss = 0.0
    print("HELLO")
    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)

            # Apply CRF
            probabilities = torch.sigmoid(outputs)

            refined_masks = []
            for i in range(len(data)):
                original_image = (
                    data[i].cpu().permute(1, 2, 0).numpy()
                )  # Convert back to image format
                refined_mask = apply_simple_crf(original_image, probabilities[i])
                refined_masks.append(refined_mask)

            loss_fn = nn.CrossEntropyLoss()
            normal_loss = loss_fn(outputs, targets.long())
            crf_loss = loss_fn(torch.tensor(refined_masks), targets.long())

            normal_val_loss += normal_loss.item() * data.size(0)
            crf_val_loss += crf_loss.item() * data.size(0)
        # For multi-class, we select the class with the highest score for each pixel
        # predicted = torch.argmax(outputs, dim=1)  # This returns shape (batch_size, height, width)
        # correct += (predicted == targets).sum()
        # total += targets.numel()

    # accuracy = 100 * correct / total
    avg_val_loss = normal_val_loss / len(loader)
    crf_val_loss = crf_val_loss / len(loader)
    print(f"Accuracy of outputs: {avg_val_loss:.2f}%")
    print(f"Accuracy of outputs after applying CRF: {crf_val_loss:.2f}%")
    model.train()


def save_predictions_as_imgs(
    loader, model, epoch, folder="saved_images", device="cuda"
):
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
                save_image(
                    predicted_mask[i],
                    os.path.join(
                        folder, f"epoch{epoch}_batch{batch_id}_image{i}_pred.png"
                    ),
                )
                save_image(
                    targets[i],
                    os.path.join(
                        folder, f"epoch{epoch}_batch{batch_id}_image{i}_target.png"
                    ),
                )

    model.train()


def save_masks(
    predicted_masks, ground_truth_masks, output_dir="output_masks", num_images=5
):
    """
    Save the predicted and ground truth masks for the first 'num_images' images.

    Args:
        predicted_masks (list or array): List or array of predicted masks.
        ground_truth_masks (list or array): List or array of ground truth masks.
        output_dir (str): Directory where the masks will be saved. Default is 'output_masks'.
        num_images (int): Number of images for which the masks will be saved. Default is 5.
    """
    # Create the output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Ensure we don't try to access more than the available number of images
    for i in range(min(num_images, len(predicted_masks))):
        # Get the predicted and ground truth mask
        predicted_mask = predicted_masks[i]
        ground_truth_mask = ground_truth_masks[i]

        # Squeeze to remove any extra dimensions (e.g., batch dimension)
        predicted_mask = predicted_mask.squeeze()  # Remove single-dimension entries
        ground_truth_mask = ground_truth_mask.squeeze()

        # Check if it's a multi-class mask (3D tensor, e.g., (num_classes, height, width))
        if predicted_mask.ndimension() == 3:  # Multi-class segmentation
            predicted_mask = predicted_mask.argmax(
                0
            )  # Choose the class with the highest probability
            ground_truth_mask = ground_truth_mask.argmax(
                0
            )  # Similarly for ground truth
        else:  # Binary segmentation (2D tensor, e.g., (height, width))
            predicted_mask = (predicted_mask > 0.5).cpu().numpy().astype(np.uint8)
            ground_truth_mask = (ground_truth_mask > 0.5).cpu().numpy().astype(np.uint8)

        # Ensure the mask is on CPU and convert to NumPy array
        predicted_mask = predicted_mask.cpu().numpy().astype(np.uint8)
        ground_truth_mask = ground_truth_mask.cpu().numpy().astype(np.uint8)

        # Convert numpy arrays to PIL images
        predicted_mask_image = Image.fromarray(predicted_mask)
        ground_truth_mask_image = Image.fromarray(ground_truth_mask)

        # Save the masks to the output directory
        predicted_mask_image.save(os.path.join(output_dir, f"predicted_mask_{i+1}.png"))
        ground_truth_mask_image.save(
            os.path.join(output_dir, f"ground_truth_mask_{i+1}.png")
        )

    print(f"First {num_images} masks saved successfully in '{output_dir}'.")


# def save_masks(predicted_masks, ground_truth_masks, output_dir='output_masks', num_images=5):
#     """
#     Save the predicted and ground truth masks for the first 'num_images' images.

#     Args:
#         predicted_masks (list or array): List or array of predicted masks.
#         ground_truth_masks (list or array): List or array of ground truth masks.
#         output_dir (str): Directory where the masks will be saved. Default is 'output_masks'.
#         num_images (int): Number of images for which the masks will be saved. Default is 5.
#     """
#     # Create the output directory if it doesn't exist
#     os.makedirs(output_dir, exist_ok=True)

#     # Ensure we don't try to access more than the available number of images
#     for i in range(min(num_images, len(predicted_masks))):
#         # Get the predicted and ground truth mask
#         predicted_mask = predicted_masks[i]
#         ground_truth_mask = ground_truth_masks[i]

#         # Squeeze to remove any extra dimensions
#         predicted_mask = predicted_mask.squeeze()  # Remove single-dimension entries
#         ground_truth_mask = ground_truth_mask.squeeze()

#         # Ensure the mask is in the range [0, 1] for binary classification
#         predicted_mask = (predicted_mask > 0.5).cpu().numpy().astype(np.uint8)
#         ground_truth_mask = ground_truth_mask.cpu().numpy().astype(np.uint8)

#         # Convert numpy arrays to PIL images
#         predicted_mask_image = Image.fromarray(predicted_mask)
#         ground_truth_mask_image = Image.fromarray(ground_truth_mask)

#         # Save the masks to the output directory
#         predicted_mask_image.save(os.path.join(output_dir, f'predicted_mask_{i+1}.png'))
#         ground_truth_mask_image.save(os.path.join(output_dir, f'ground_truth_mask_{i+1}.png'))

#     print(f"First {num_images} masks saved successfully in '{output_dir}'.")
