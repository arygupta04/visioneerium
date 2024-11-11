import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim 
import segmentation_models_pytorch as smp

from data import load_data 
from utils import(
    load_checkpoint,
    save_checkpoint,
    get_loaders,
    check_accuracy,
    save_predictions_as_imgs,
)

# Hyperparameters etc.
LEARNING_RATE = 1e-4
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 6
NUM_EPOCHS = 1
NUM_WORKERS = 6
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
PIN_MEMORY = False
LOAD_MODEL = False

def train_fn(loader, model, optimizer, loss_fn):
    loop = tqdm(loader, total=len(loader), desc="Training", leave=True)
    for batch_id, (data, targets) in enumerate(loop):
        data = data.to(device=DEVICE)
        targets = targets.float().to(device=DEVICE)
        targets = targets.squeeze(1)
        targets = targets.long()
        optimizer.zero_grad()

        # Forward pass
        predictions = model(data)
        loss = loss_fn(predictions, targets)

        # Backward pass and optimization
        loss.backward()
        optimizer.step()

        # Update tqdm loop
        loop.set_postfix(loss=loss.item())

def main():

    # Initialize pre-built UNET model (from segmentation_models_pytorch)
    model = smp.Unet(
        encoder_name="efficientnet-b4",  # Smaller model for lower memory usage
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
    ).to(DEVICE)

    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()  # Cross entropy loss

    # Get data loaders
    train_loader, val_loader, test_loader = load_data(
        rf"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\images", 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )

    # Training loop
    for epoch in range(NUM_EPOCHS):
        print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

        train_fn(train_loader, model, optimizer, loss_fn)

        # Check accuracy on validation set
        check_accuracy(val_loader, model, device=DEVICE)
        

if __name__ == "__main__":
    main()