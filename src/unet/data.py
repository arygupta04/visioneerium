# deep learning libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pycocotools.coco import COCO
import json
import numpy as np

# other libraries
import os
from PIL import Image
import pandas as pd

from torchvision.transforms import functional as F


# This class is the Turtle Dataset
class TurtleDataset(Dataset):

    # Global annotations variable
    coco = COCO("data/turtles-data/data/annotations.json")

    # Constructor of TurtleDataset
    def __init__(self, split_type: str, path: str) -> None:
        self.path = path
        self.names = os.listdir(path)
        self.split_type = split_type

        # Load metadata and filter based on split
        metadata = pd.read_csv("data/turtles-data/data/metadata_splits.csv")
        self.img_ids = metadata[metadata["split_open"] == split_type]["id"].tolist()

    # returns the length of the dataset
    def __len__(self) -> int:
        return len(self.img_ids)

    # Loads an item based on the index.

    # Args:
    #  index: index of the element in the dataset.

    # Returns:
    #  tuple with image and label. Image dimensions:
    #  [channels, height, width].
    #
    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:

        # Retrieve the image ID
        print("Loading image: " + str(index))
        img_id = self.img_ids[index]
        image_info = self.coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        image_path = f"data/turtles-data/data/{file_name}"
        image = Image.open(image_path)

        # getting the mask for the images
        print("Generating mask for image: " + str(img_id))
        mask = self.generate_mask(img_id)

        # Get the target size (maximum width and height in the dataset)
        max_width = 2016
        max_height = 2016

        # print("max: " + str(max_width) + ", " + str(max_height))

        # Pad the image and mask to the max dimensions
        image = self.pad_image(image, max_width, max_height)
        mask = self.pad_image(mask, max_width, max_height)

        # Get the dimensions
        width_img, height_img = image.size
        width_mask, height_mask = mask.shape[:2]

        # Transform image and mask
        transformations = transforms.Compose([transforms.ToTensor()])
        image = transformations(image)
        mask = transformations(mask)

        return image, mask

    def pad_image(self, img, target_width, target_height):
        """
        Pad the image to the target width and height.
        Adds black padding (value 0).
        """
        if isinstance(img, Image.Image):  # Check if the image is a PIL Image
            width, height = img.size
            padding = (
                0,
                0,
                target_width - width,
                target_height - height,
            )  # (left, top, right, bottom)
            img = F.pad(img, padding, fill=0)  # Fill padding with 0 (black pixels)
        elif isinstance(img, np.ndarray):  # Check if the image is a numpy array
            height, width = img.shape[:2]
            padded_img = np.zeros((target_height, target_width), dtype=img.dtype)
            padded_img[:height, :width] = img
            img = padded_img
        return img

    def generate_mask(self, img_id):

        cat_ids = self.coco.getCatIds()

        # Load the single image
        img = self.coco.loadImgs([img_id])[0]

        # Load annotations for this image ID
        anns_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)

        # Initialize an empty mask for this image
        mask = np.zeros((img["height"], img["width"]), dtype=np.uint8)

        # Generate the mask by adding each annotation to this image's mask
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))

        # Load the image and pair with its mask
        file_name = f"data/turtles-data/data/{img['file_name']}"
        image = np.array(Image.open(file_name))

        return mask


def load_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns dataloaders for training, validation, and test sets.

    Args:
        path: path of the dataset.
        batch_size: batch size for dataloaders. Default value: 128.
        num_workers: number of workers for loading data. Default value: 0.

    Returns:
        tuple of dataloaders, train, val, and test in respective order.
    """

    # create datasets
    train_dataset = TurtleDataset("train", path)
    val_dataset = TurtleDataset("valid", path)
    test_dataset = TurtleDataset("test", path)

    # define dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data("data/turtles-data/data/images")
