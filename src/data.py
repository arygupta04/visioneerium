import os
import cv2
import torch
import numpy as np
from torch.utils.data import Dataset
from torch.utils.data import Dataset, DataLoader
from pycocotools.coco import COCO
import pycocotools.mask as cocoMask
from torchvision import transforms
from PIL import Image
import pandas as pd


class TurtleDataset(Dataset):

    def __init__(
        self, split_type: str, path: str, target_size: tuple[int, int] = (256, 256)
    ):
        self.path = path
        self.coco = COCO(os.path.join(path, "annotations.json"))
        self.names = os.listdir(path)
        self.split_type = split_type

        metadata = pd.read_csv(os.path.join(path, "metadata_splits.csv"))
        self.img_ids = metadata[metadata["split_open"] == split_type]["id"].tolist()
        self.max_width, self.max_height = self.find_max_dimensions()
        self.target_size = target_size

    def generate_mask(self, img_id: int) -> np.ndarray:
        """
        Generates a mask for the given image id.

        Args:
            img_id: image id for which mask is to be generated.

        Returns:
            mask: mask for the given image id.
        """
        img_info = self.coco.loadImgs(img_id)[0]
        img_width = img_info["width"]
        img_height = img_info["height"]

        mask_head = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_carapace = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_flippers = np.zeros((img_height, img_width), dtype=np.uint8)

        cat_ids = self.coco.getCatIds()
        anns_ids = self.coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        anns = self.coco.loadAnns(anns_ids)

        for ann in anns:
            cat_id = ann["category_id"]
            rle = ann["segmentation"]
            segmentation = ann["segmentation"]

            # Check if segmentation is in uncompressed RLE format
            if (
                isinstance(segmentation, dict)
                and "counts" in segmentation
                and isinstance(segmentation["counts"], list)
            ):
                # Convert uncompressed RLE to compressed RLE
                rle = cocoMask.frPyObjects([segmentation], *segmentation["size"])[0]
            else:
                rle = segmentation

            # Set up masks
            rle_mask = cocoMask.decode(rle)
            rle_mask = np.array(Image.fromarray(rle_mask))

            if cat_id == 1:
                mask_carapace[rle_mask == 1] = 1
            elif cat_id == 2:
                mask_flippers[rle_mask == 1] = 2
            elif cat_id == 3:
                mask_head[rle_mask == 1] = 3

        # Combine masks
        mask = np.maximum(mask_carapace, mask_flippers)
        mask = np.maximum(mask, mask_head)

        return mask

    def find_max_dimensions(self):
        """
        Finds the maximum width and height of the images in the dataset.

        Returns:
            max_width: maximum width of the images.
            max_height: maximum height of the images.
        """
        max_width, max_height = 0, 0

        # Iterate over all images to find the maximum width and height
        for img_id in self.img_ids:
            img_info = self.coco.imgs[img_id]
            width, height = img_info["width"], img_info["height"]

            max_width = max(max_width, width)
            max_height = max(max_height, height)

        return max_width, max_height

    def __len__(self) -> int:
        return len(self.img_ids)

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """
        Returns the image and mask for the given index.

        Args:
            index: index of the image to be returned.

        Returns:
            image: image tensor.
            mask: mask tensor.
        """
        # Load image
        img_id = self.img_ids[index]
        image_info = self.coco.loadImgs(img_id)
        file_name = image_info[0]["file_name"]
        image_path = os.path.join(self.path, file_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate mask
        mask = self.generate_mask(img_id)

        # Apply padding to image and mask
        pad_height = self.max_height - image.shape[0]
        pad_width = self.max_width - image.shape[1]
        image = cv2.copyMakeBorder(
            image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0]
        )
        padded_mask = cv2.copyMakeBorder(
            mask, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0
        )

        # Resize image and mask
        image = cv2.resize(image, self.target_size)
        padded_mask = cv2.resize(padded_mask, self.target_size)

        # Convert image and mask to tensor
        transformations = transforms.ToTensor()
        image = transformations(image)
        padded_mask = torch.tensor(padded_mask, dtype=torch.uint8)

        return image, padded_mask


def load_data(
    path: str,
    batch_size: int = 128,
    num_workers: int = 0,
    target_size: tuple[int, int] = (256, 256),
) -> tuple[DataLoader, DataLoader, DataLoader]:
    """
    Returns dataloaders for training, validation, and test sets.

    Args:
        path: path of the dataset.
        batch_size: batch size for dataloaders. Default value: 128.
        num_workers: number of workers for loading data. Default value: 0.
        target_size: target size for the images. Default value: (256, 256).

    Returns:
        train_dataloader: dataloader for training set.
        val_dataloader: dataloader for validation set.
        test_dataloader: dataloader for test set.
    """

    # Create datasets
    train_dataset = TurtleDataset("train", path, target_size=target_size)
    val_dataset = TurtleDataset("valid", path, target_size=target_size)
    test_dataset = TurtleDataset("test", path, target_size=target_size)

    # Define dataloaders
    train_dataloader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    val_dataloader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=True, num_workers=num_workers
    )
    test_dataloader = DataLoader(
        test_dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers
    )

    return train_dataloader, val_dataloader, test_dataloader


if __name__ == "__main__":
    train_loader, val_loader, test_loader = load_data(
        r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data"
    )
