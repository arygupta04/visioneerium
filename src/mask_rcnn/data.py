# Assumption /data is loaded within the mask_rcnn folder

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


# This class is the Turtle Dataset
class TurtleDataset(Dataset):

    # Constructor of TurtleDataset
    # provide a path to the data directory
    def __init__(self, path: str) -> None:
        self.path = path
        self.annotations_filename = f"{path}/annotations.json"
        self.annotations = self.load_annotations(self.annotations_filename)
        self.coco = COCO(self.annotations_filename)

    # Load annotations from JSON file
    def load_annotations(self, annotation_file: str) -> dict:
        with open(annotation_file, "r") as f:
            annotations = json.load(f)
        return annotations

    # returns the length of the dataset
    def __len__(self) -> int:
        return len(self.names)

    # Loads an item based on the index.

    # Args:
    #  index: image index of the element in the dataset.

    # Returns:
    #  tuple with image,mask and bbox. Image dimensions:
    #  [channels, height, width].
    #
    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:

        # retrive the metadata on the image
        image_info = self.coco.loadImgs(index)
        file_name = image_info[0]["file_name"]
        image_path = f"{self.path}/{file_name}"
        image = Image.open(image_path)

        # getting the mask for the images
        mask, bboxes = self.generate_mask(index)

        # Transform image and mask
        transformations = transforms.Compose([transforms.ToTensor()])
        image = transformations(image)
        mask = transformations(mask)
        bboxes = torch.tensor(bboxes, dtype=torch.float32)

        return image, mask, bboxes

    def generate_mask(self, img_id):

        # Initialize COCO object inside __getitem__

        cat_ids = self.coco.getCatIds()

        # Load the single image
        img = self.coco.loadImgs([img_id])[0]

        # Load annotations for this image ID
        anns_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)

        # Initialize an empty mask for this image
        mask = np.zeros((img["height"], img["width"]), dtype=np.uint8)

        # Generate the mask by adding each annotation to this image's mask
        bboxes = []
        for ann in anns:
            mask = np.maximum(mask, self.coco.annToMask(ann))
            bboxes.append(ann["bbox"])

        return mask, bboxes


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
    train_dataset = TurtleDataset(f"{path}/train")
    val_dataset
    train_dataset, val_dataset = random_split(
        train_dataset, [int(0.8 * len(train_dataset)), int(0.2 * len(train_dataset))]
    )
    test_dataset = TurtleDataset(f"{path}/val")

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
    # load_data("data")
    # pass in the path to the dataset
    a = TurtleDataset("data/turtles-data/data")
    a.__getitem__(3)
