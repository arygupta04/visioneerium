import torch
import torch.nn as nn

def save_checkpoint(
    model, optimizer, epoch=None, loss=None, filename="checkpoint.pth.tar"
):
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": epoch,
        "loss": loss,
    }
    torch.save(checkpoint, filename)
    print(f"Checkpoint saved to {filename}")


def load_checkpoint(checkpoint_path, model, optimizer=None):
    checkpoint = torch.load(checkpoint_path)

    model.load_state_dict(checkpoint["model_state_dict"])

    if optimizer:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    epoch = checkpoint.get("epoch", 0)
    loss = checkpoint.get("loss", None)

    return model, optimizer, epoch, loss



def calculate_val_loss(loader, model):

    model.eval()
    val_loss = 0.0
   
    with torch.no_grad():
        for data, targets in loader:
        
            # Forward pass
            outputs = model(data)
            loss_fn = nn.CrossEntropyLoss()
            loss = loss_fn(outputs, targets.long())

            val_loss += loss.item() * data.size(0)
        
    avg_val_loss = val_loss/len(loader)
    print(f"Loss: {avg_val_loss:.2f}")
    model.train()



def save_model(model, filename="trained_deeplabv3_model.pth"):
    torch.save(model.state_dict(), filename)
    print(f"Model saved to {filename}")
