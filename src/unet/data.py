# deep learning libraries
import torch
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from pycocotools.coco import COCO
import numpy as np
import cv2

# other libraries
import os
import pandas as pd

from torchvision.transforms import functional as F

# Global annotations variable
coco = COCO(r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\annotations.json")

# This class is the Turtle Dataset
class TurtleDataset(Dataset):

    # Constructor of TurtleDataset
    def __init__(self, split_type: str ,path: str) -> None:
        self.path = path
        self.names = os.listdir(path)
        self.split_type = split_type

        metadata = pd.read_csv(r"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\metadata_splits.csv")
        self.img_ids = metadata[metadata['split_open'] == split_type]['id'].tolist()
        self.max_width, self.max_height = self.find_max_dimensions()

        ##print
        print(f"max width: {self.max_width} max height: {self.max_height}")
        ##
        

    def find_max_dimensions(self):
        max_width, max_height = 0, 0
        for img_id in self.img_ids:
            img_info = coco.imgs[img_id]
            width, height = img_info['width'], img_info['height']
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        
        return max_width, max_height

    def __len__(self) -> int:
        return len(self.img_ids)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        # Retrieve the image ID
        img_id = self.img_ids[index]
        image_info = coco.loadImgs(img_id)
        file_name = image_info[0]['file_name']

        file_name = image_info[0]['file_name']
        image_path = rf"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\\" + file_name
        
        # Load image with OpenCV and convert BGR to RGB
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Get the mask for the images
        mask = self.generate_mask(img_id)

        if not isinstance(mask, np.ndarray):
            mask = np.array(mask, dtype=np.uint8)
        else:
            mask = mask.astype(np.uint8)

        ### PAD
        pad_height = self.max_height - image.shape[0]
        pad_width = self.max_width - image.shape[1]
        image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        mask = cv2.copyMakeBorder(mask, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)
        ###

        ### RESIZE
        target_size = (512, 512)
        image = cv2.resize(image, target_size)  # Resize to 256x256 (or 128x128)
        mask = cv2.resize(mask, target_size)    # Resize to the same size

        # Transform the image and mask
        transformations = transforms.Compose([
            transforms.ToTensor(),  # Convert to tensor
        ])
        image = transformations(image)
        mask = transformations(mask)

        return image, mask

    def pad_image(self, img, target_width, target_height):
        """
        Pad the image to the target width and height.
        Adds black padding (value 0).
        """
        height, width = img.shape[:2]
        padded_img = np.zeros((target_height, target_width) + img.shape[2:], dtype=img.dtype)
        padded_img[:height, :width] = img
        return padded_img


    # def generate_mask(self, img_id):

    #     cat_ids =  coco.getCatIds()

    #     # Load the single image
    #     img = coco.loadImgs([img_id])[0]
        
    #     # Load annotations for this image ID
    #     anns_ids = coco.getAnnIds(imgIds=[img_id], catIds=cat_ids, iscrowd=None)
    #     anns = coco.loadAnns(anns_ids)

    #     # Initialize an empty mask for this image
    #     mask = np.zeros((img['height'], img['width']), dtype=np.uint8)

    #     # Generate the mask by adding each annotation to this image's mask
    #     for ann in anns:
    #         mask = np.maximum(mask, coco.annToMask(ann))

    #     return mask
    def generate_mask(self, img_id):
        # Vectorized mask generation
        anns = coco.loadAnns(coco.getAnnIds(imgIds=[img_id], catIds=coco.getCatIds(), iscrowd=None))
        mask = np.sum([coco.annToMask(ann) for ann in anns], axis=0)
        mask = np.clip(mask, 0, 1)  # Ensure mask values are 0 or 1
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
    train_dataset = TurtleDataset('train', path)
    val_dataset = TurtleDataset('valid', path)
    test_dataset = TurtleDataset('test', path)

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
    train_loader, val_loader, test_loader = load_data(rf"C:\Users\vedan\Desktop\COMP9517\COMP9517 group project\turtles-data\data\images")