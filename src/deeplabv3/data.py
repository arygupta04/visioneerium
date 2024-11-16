import os
import cv2
import torch
import numpy as np
from pycocotools.coco import COCO
import pycocotools.mask as cocoMask
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import logging

# Suppress pycocotools logging
logging.getLogger("pycocotools").setLevel(logging.ERROR)

# COCO instance
coco = COCO("/Users/aryan/Desktop/COMP9517/grp_project/visioneerium/src/dataSet/turtles-data/data/annotations.json")


class TurtleDataset(Dataset):

    def __init__(self, split_type: str, path: str) -> None:
        self.path = path
        self.names = os.listdir(path)
        self.split_type = split_type

        metadata = pd.read_csv("/Users/aryan/Desktop/COMP9517/grp_project/visioneerium/src/dataSet/turtles-data/data/metadata_splits.csv")
        self.img_ids = metadata[metadata["split_open"] == split_type]["id"].tolist()
        self.max_width, self.max_height = self.find_max_dimensions()


    def generate_mask(self, img_id: int, img: np.ndarray) -> np.ndarray:
        img_info = coco.loadImgs(img_id)[0]
        img_width = img_info["width"]
        img_height = img_info["height"]

        # Masks initialised
        mask_head = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_carapace = np.zeros((img_height, img_width), dtype=np.uint8)
        mask_flippers = np.zeros((img_height, img_width), dtype=np.uint8)

        # Loading in annotations
        cat_ids = coco.getCatIds()
        anns_ids = coco.getAnnIds(imgIds=img_id, catIds=cat_ids, iscrowd=None)
        anns = coco.loadAnns(anns_ids)

        for ann in anns:
            cat_id = ann["category_id"]
            segmentation = ann["segmentation"]

            # Handle segmentation format
            if (isinstance(segmentation, dict) and "counts" in segmentation 
                and isinstance(segmentation["counts"], list)):
                rle = cocoMask.frPyObjects([segmentation], *segmentation["size"])[0]
            else:
                rle = segmentation 

            rle_mask = cocoMask.decode(rle)
        
            # Assign each category to its mask
            # Turtle body
            if cat_id == 1:
                mask_carapace[rle_mask == 1] = 1
            # Flippers
            elif cat_id == 2:
                mask_flippers[rle_mask == 1] = 2
            # Head
            elif cat_id == 3:
                mask_head[rle_mask == 1] = 3

        # Combine masks
        mask = np.maximum.reduce([mask_carapace, mask_flippers, mask_head])
        return mask


    def find_max_dimensions(self) -> tuple[int, int]:
        max_width, max_height = 0, 0
        for img_id in self.img_ids:
            img_info = coco.imgs[img_id]
            width, height = img_info["width"], img_info["height"]
            max_width = max(max_width, width)
            max_height = max(max_height, height)
        return max_width, max_height
    

    def __len__(self) -> int:
        return len(self.img_ids)


    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        img_id = self.img_ids[index]
        image_info = coco.loadImgs(img_id)

        file_name = image_info[0]['file_name']
        image_path = f"/Users/aryan/Desktop/COMP9517/grp_project/visioneerium/src/dataSet/turtles-data/data/{file_name}"
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Generate the combined mask
        mask = self.generate_mask(img_id, image)

        # Adding padding
        pad_height = self.max_height - image.shape[0]
        pad_width = self.max_width - image.shape[1]
        image = cv2.copyMakeBorder(image, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=[0, 0, 0])
        padded_mask = cv2.copyMakeBorder(mask, 0, pad_height, 0, pad_width, cv2.BORDER_CONSTANT, value=0)

        # Resize image and mask
        target_size = (256, 256)
        image = cv2.resize(image, target_size)
        padded_mask = cv2.resize(padded_mask, target_size)

        # Transforming image to tensor
        transformations = transforms.ToTensor() 
        image = transformations(image)

        # Convert the mask to tensor
        padded_mask = torch.tensor(padded_mask, dtype=torch.uint8)

        return image, padded_mask



def load_data(
    path: str, batch_size: int = 128, num_workers: int = 0
) -> tuple[DataLoader, DataLoader, DataLoader]:
   
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
    train_loader, val_loader, test_loader = load_data("/Users/aryan/Desktop/COMP9517/grp_project/visioneerium/src/dataSet/turtles-data/data/images")
