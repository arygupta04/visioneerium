# deep learning libraries
import torch
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

# other libraries
from tqdm.auto import tqdm


@torch.enable_grad()
def train_step(
    model: torch.nn.Module,
    train_data: DataLoader,
    criterion: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        train_data: dataloader of train data.
        criterion: loss function.
        optimizer: optimizer.
        epoch: epoch of the training.
        device: device for running operations.
    """

    model.train()
    tqdm_train_data = tqdm(train_data, desc=f"Epoch {epoch}")

    for data, target in tqdm_train_data:
        data, target = data.to(device), target.to(device)

        # Zero the gradients
        optimizer.zero_grad()

        # Forward pass
        output = model(data)
        loss = criterion(output, target)

        # Backward pass
        loss.backward()
        optimizer.step()

        tqdm_train_data.set_postfix(loss=loss.item())


@torch.no_grad()
def val_step(
    model: torch.nn.Module,
    val_data: DataLoader,
    criterion: torch.nn.Module,
    device: torch.device,
) -> None:
    """
    This function train the model.

    Args:
        model: model to train.
        val_data: dataloader of validation data.
        criterion: loss function.
        device: device for running operations.

    Returns:
        average loss over the validation data.
    """

    model.eval()
    avg_loss = 0.0
    for data, target in val_data:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        avg_loss += loss.item()

    avg_loss /= len(val_data)
    return avg_loss


@torch.no_grad()
def test_step(
    model: torch.nn.Module,
    test_data: DataLoader,
    device: torch.device,
    criterion: torch.nn.Module,
) -> float:
    """
    This function tests the model.

    Args:
        model: model to make predcitions.
        test_data: dataset for testing.
        device: device for running operations.
        criterion: loss function.

    Returns:
        average loss over the test data.
    """

    model.eval()
    avg_loss = 0.0
    for data, target in test_data:
        data, target = data.to(device), target.to(device)
        output = model(data)
        loss = criterion(output, target)
        avg_loss += loss.item()

    avg_loss /= len(test_data)
    return avg_loss
