# deep learning libraries
import torch
from torch.utils.data import DataLoader

# other libraries
from tqdm.auto import tqdm


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        optimizer: optimizer.
        epoch: epoch of the training.
        device: device for running operations.

    Returns:
        average loss over the training data.
    """

    model.train()
    tqdm_train_data = tqdm(train_data, desc=f"Epoch {epoch}")
    avg_loss = 0.0

    for data, target in tqdm_train_data:
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        loss = model(data, target)
        total_loss = sum(loss.values())

        # Backward pass
        total_loss.backward()
        optimizer.step()

        tqdm_train_data.set_postfix(loss=loss.item())
        avg_loss += loss.item()

    avg_loss /= len(train_data)
    return avg_loss


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        device: device for running operations.

    Returns:
        average loss over the validation data.
    """

    model.eval()
    avg_loss = 0.0

    for data, target in val_data:
        data, target = data.to(device), target.to(device)

        loss = model(data, target)
        total_loss = sum(loss.values())

        avg_loss += total_loss.item()

    avg_loss /= len(val_data)
    return avg_loss


@torch.no_grad()
def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        device: device for running operations.

    Returns:
        average loss over the test data.
    """

    model.eval()
    avg_loss = 0.0

    for data, target in test_data:
        data, target = data.to(device), target.to(device)

        loss = model(data, target)
        total_loss = sum(loss.values())

        avg_loss += total_loss.item()

    avg_loss /= len(test_data)
    return avg_loss
