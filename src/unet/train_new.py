import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim
import segmentation_models_pytorch as smp
import torch.cuda.amp as amp  # For mixed precision training
from torch.cuda.amp import GradScaler, autocast

from src.unet.data import load_data
from src.unet.utils import (
    save_checkpoint,
    check_accuracy,
)
import warnings

warnings.filterwarnings("ignore")

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 32
NUM_EPOCHS = 3
NUM_WORKERS = 6
IMAGE_HEIGHT = 2016
IMAGE_WIDTH = 2016
PIN_MEMORY = True
LOAD_MODEL = False
PATH = "data/turtles-data/data"

# Initialize scaler
scaler = GradScaler()


def train_fn(loader, model, optimizer, loss_fn, scaler):
    loop = tqdm(loader)
    for batch_id, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)

        optimizer.zero_grad()

        # Mixed precision training
        with autocast():
            predictions = model(data)
            loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())


def main():

    #  # Initialize pre-built UNET model (from segmentation_models_pytorch)
    # model = smp.Unet(
    #     encoder_name="resnet18",  # Smaller model for lower memory usage
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=4,
    # ).to(DEVICE)

    model = smp.DeepLabV3Plus(
        encoder_name="resnet18",
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.BCEWithLogitsLoss()  # cross entropy loss

    # Get data loaders
    train_loader, val_loader, test_loader = load_data(
        path=PATH,
        batch_size=BATCH_SIZE,
        num_workers=NUM_WORKERS,
    )

    # Initialize the scaler for mixed precision training
    scaler = amp.GradScaler()

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_fn(train_loader, model, optimizer, loss_fn, scaler)

        # # Save model and checkpoint
        if epoch % 10 == 0:
            save_checkpoint(model, optimizer, filename="checkpoint.pth.tar")

        # Check accuracy on validation set
        check_accuracy(val_loader, model, device=DEVICE)

        # Save predictions as images
    # save_predictions_as_imgs(val_loader, model, epoch, folder="saved_images", device=DEVICE)


if __name__ == "__main__":
    main()
