import torch
from tqdm import tqdm
import torch.nn as nn
import torch.optim as optim 
import segmentation_models_pytorch as smp
from matplotlib import pyplot as plt
from original_unet import UNet
from data import load_data
from utils import(
    combined_loss,
    load_checkpoint,
    save_checkpoint,
    calculate_val_loss,
    save_model
)

# Hyperparameters
LEARNING_RATE = 0.001
DEVICE = "cpu"
BATCH_SIZE = 6
NUM_EPOCHS = 1
NUM_WORKERS = 6
PIN_MEMORY = False
LOAD_MODEL = True

train_losses = []
# Function to train the model
def train_fn(loader, model, optimizer, loss_fn):
    model.train()

    train_loss = 0.0

    loop = tqdm(loader, total=len(loader), desc="Training", leave=True)
    for images, masks in loop:
        images = images.to(DEVICE)
        masks = masks.to(DEVICE)

        # Forward
        predictions = model(images)
        loss = loss_fn(predictions, masks.long())
        train_loss += loss.item() * images.size(0)

        # Backward
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Update progress bar with loss information
        loop.set_postfix(loss=loss.item())
    
    avg_train_loss = train_loss/len(loader.dataset)
    train_losses.append(avg_train_loss)


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
            predictions = torch.argmax(torch.sigmoid(outputs), dim=1)
            predictions_np = predictions.cpu().detach().numpy()

            # Calculate IoU for each category
            carapace_ious.append(torch.tensor(calculate_iou(predictions, targets, class_value=1)))
            flippers_ious.append(torch.tensor(calculate_iou(predictions, targets, class_value=2)))
            head_ious.append(torch.tensor(calculate_iou(predictions, targets, class_value=3)))
            plt.figure(figsize=(10, 5))

            # break
        # Filter out NaN values and compute mean IoU for each category over the test set
        head_ious = [iou for iou in head_ious if not torch.isnan(iou)]
        flippers_ious = [iou for iou in flippers_ious if not torch.isnan(iou)]
        carapace_ious = [iou for iou in carapace_ious if not torch.isnan(iou)]

        mean_head_iou = torch.mean(torch.stack(head_ious))  # Use torch.stack to concatenate the tensors
        mean_flippers_iou = torch.mean(torch.stack(flippers_ious))  # Use torch.stack to concatenate the tensors
        mean_carapace_iou = torch.mean(torch.stack(carapace_ious))  # Use torch.stack to concatenate the tensors

        print(f"Head mIoU: {mean_head_iou:.4f}")
        print(f"Flippers mIoU: {mean_flippers_iou:.4f}")
        print(f"Carapace mIoU: {mean_carapace_iou:.4f}")
    
    model.train()

def plot_loss_graph():
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label="Train Loss", color="blue", marker="o")
    plt.title("Training Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    plt.grid(True)
    plt.show()

###############################################
def main():
    # model = smp.Unet(
    #     encoder_name="mobilenet_v2",  # Smaller model for lower memory usage
    #     encoder_weights="imagenet",
    #     in_channels=3,
    #     classes=4,
    # ).to(DEVICE)

    model = smp.Unet(
        encoder_name="mobilenet_v2",  # Smaller model for lower memory usage
        encoder_weights="imagenet",
        in_channels=3,
        classes=4,
    ).to(DEVICE)

    #model = UNet(num_classes=4, in_channels=3).to(DEVICE)


    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    loss_fn = nn.CrossEntropyLoss()  # Cross entropy loss
    #loss_fn = lambda pred, target: combined_loss(pred, target, weight_ce=1.0, weight_dice=1.0)
    
    train_loader, val_loader, test_loader = load_data(r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\images", 
        batch_size=BATCH_SIZE, 
        num_workers=NUM_WORKERS
    )
    
    if LOAD_MODEL:
        print("continuing...")
        checkpoint_path = "checkpoint_epoch_4.pth.tar"  # Path to your checkpoint
        model, optimizer, start_epoch, _ = load_checkpoint(checkpoint_path, model, optimizer, device=DEVICE)
        
        if start_epoch is None:  # Ensure start_epoch is not None
            start_epoch = 0  # Default to epoch 0 if None
        if start_epoch + 1 <= NUM_EPOCHS:

            print(f"Resuming from epoch {start_epoch + 1}...")

            for epoch in range(start_epoch + 1, NUM_EPOCHS):
                print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

                train_fn(train_loader, model, optimizer, loss_fn)

                save_checkpoint(model, optimizer, epoch, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar")

                # Check accuracy on validation set
                calculate_val_loss(val_loader, model, device=DEVICE)

    else:
    
        # Training loop
        for epoch in range(NUM_EPOCHS):
            print(f"Epoch {epoch + 1}/{NUM_EPOCHS}")

            train_fn(train_loader, model, optimizer, loss_fn)

            save_checkpoint(model, optimizer, filename=f"checkpoint_epoch_{epoch + 1}.pth.tar")

            # Check accuracy on validation set
            calculate_val_loss(val_loader, model, device=DEVICE)
        
        #plot_loss_graph()
        
    # Save the final trained model
    save_model(model, filename="trained_model.pth")
  
    test_fn(test_loader, model, device=DEVICE)

if __name__ == "__main__":
    main()