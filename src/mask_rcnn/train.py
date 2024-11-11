# deep learning libraries
import torch
from torchvision.models.detection import maskrcnn_resnet50_fpn_v2

# other libraries
from typing import Final
import json
import multiprocessing

# own modules
from src.mask_rcnn.data import load_data
from src.mask_rcnn.train_functions import train_step, val_step, test_step
from src.utils import set_seed, save_model, parameters_to_double

import os

# static variables
DATA_PATH: Final[str] = "data/turtles-data/data"

# set device and seed
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
set_seed(42)


def main() -> None:
    """
    This function is the main program for training.
    """

    HPP_DICT = dict(
        batch_size=16,
        epochs=30,
        lr=0.001,
        weight_decay=0.0,
        patience=5,
        trainable_backbone_layers=3,
        num_workers=0,
    )

    # Load the data
    train_loader, val_loader, test_loader = load_data(DATA_PATH, HPP_DICT)
    print("Data loaded")

    # Create the model
    model = maskrcnn_resnet50_fpn_v2(
        weights="DEFAULT",
        trainable_backbone_layers=HPP_DICT["trainable_backbone_layers"],
    )  # TODO temporary, choose another backbone
    model.name = "maskrcnn_resnet_pt_func"
    model.to(device)

    # parameters_to_double(model)

    # Create the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=HPP_DICT["lr"], weight_decay=HPP_DICT["weight_decay"]
    )

    # Create lr scheduler
    lr_scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer,
        max_lr=HPP_DICT["lr"],
        steps_per_epoch=len(train_loader),
        epochs=HPP_DICT["epochs"],
    )

    best_loss = float("inf")
    no_improvement = 0

    # Train the model
    for epoch in range(1, HPP_DICT["epochs"] + 1):
        # Train the model
        train_loss = train_step(
            model,
            train_loader,
            optimizer,
            epoch,
            device,
        )

        # Evaluate the model
        val_loss = val_step(
            model,
            val_loader,
            device,
        )

        print("------------\nEPOCH: ", epoch)
        print("Training Loss: ", train_loss)
        print("Validation Loss: ", val_loss)

        if val_loss < best_loss:
            best_loss = val_loss
            # Save the model
            i = len(os.listdir("models")) + 1
            save_model(model, f"model_{i}")

            # Save the metadata
            training_metadata = {
                "epoch": epoch,
                "train_loss": train_loss,
                "valid_loss": val_loss,
                "lr": lr_scheduler.get_last_lr(),
                "model_architecture": model.name,
            }
            with open(f"models/model_{i}_metadata.json", "w", encoding="utf-8") as f:
                json.dump(training_metadata, f)

            no_improvement = 0
        else:
            no_improvement += 1
            if no_improvement > HPP_DICT["patience"]:
                break

        # Step the learning rate scheduler
        lr_scheduler.step()

    # Test the model
    i = len(os.listdir("models"))
    best_model = torch.jit.load(f"models/model_{i}.pt")
    test_loss = test_step(best_model, test_loader, device)
    print("Test Loss: ", test_loss)


if __name__ == "__main__":
    main()
