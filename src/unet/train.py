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
IMAGE_HEIGHT = 256
IMAGE_WIDTH = 256
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
#####################################
########TESTING###############
def calculate_iou(prediction, target, class_value):
    # Convert the mask for the specific class
    pred_class = (prediction == class_value).float()
    target_class = (target == class_value).float()
    
    # Calculate IoU
    intersection = (pred_class * target_class).sum()
    union = pred_class.sum() + target_class.sum() - intersection
    if union == 0:
        return float('nan')  # Avoid division by zero if both are empty
    else:
        return intersection / union


def test_fn(loader, model, device="cuda"):
    model.eval()
    head_ious = []
    flippers_ious = []
    carapace_ious = []

    with torch.no_grad():
        for data, targets in loader:
            data = data.to(device)
            targets = targets.to(device)

            # Forward pass
            outputs = model(data)
            predictions = (torch.sigmoid(outputs) > 0.5).float()

            # Calculate IoU for each category
            head_ious.append(calculate_iou(predictions, targets, class_value=1))
            flippers_ious.append(calculate_iou(predictions, targets, class_value=2))
            carapace_ious.append(calculate_iou(predictions, targets, class_value=3))

    # Filter out NaN values and compute mean IoU for each category over the test set
    head_ious = [iou for iou in head_ious if not torch.isnan(iou)]
    flippers_ious = [iou for iou in flippers_ious if not torch.isnan(iou)]
    carapace_ious = [iou for iou in carapace_ious if not torch.isnan(iou)]

    head_miou = sum(head_ious) / len(head_ious) if head_ious else float('nan')
    flippers_miou = sum(flippers_ious) / len(flippers_ious) if flippers_ious else float('nan')
    carapace_miou = sum(carapace_ious) / len(carapace_ious) if carapace_ious else float('nan')

    print(f"Head mIoU: {head_miou:.4f}")
    print(f"Flippers mIoU: {flippers_miou:.4f}")
    print(f"Carapace mIoU: {carapace_miou:.4f}")
    model.train()

###############################################
def main():

    # Initialize pre-built UNET model (from segmentation_models_pytorch)
    model = smp.Unet(
        encoder_name="mobilenet_v2",  # Smaller model for lower memory usage
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
    


    #####
    test_fn(test_loader, model, device=DEVICE)

        

if __name__ == "__main__":
    main()